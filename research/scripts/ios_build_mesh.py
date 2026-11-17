"""Сегментированный интраоральный скан -> GLB с отдельным мешем на каждый зуб.

Метки сеть даёт на подвыборке в 24000 точек, а сетка у скана — 100-160 тысяч
вершин. Переносим метки на все вершины ближайшим соседом (cKDTree), затем
метим ГРАНИ по большинству их вершин и режем сетку на части.

Координаты НЕ нормализуем: скан из Exocad хранит верх и низ в одной системе
с сохранённым прикусом, и если положить обе челюсти в одну сцену как есть,
смыкание будет видно сразу.

Сглаживание меша не применяем: trimesh 5.0.0 на упрощённых мешах разносит
геометрию шипами (подробности в cbct_backend._class_mesh).

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_build_mesh.py \
        --upper .../UpperJaw.obj --lower .../LowerJaw.obj --out case.glb
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ios_seg_train import DGCNNSeg, N_CLASSES  # noqa: E402
from ios_seg_eval import predict_scan  # noqa: E402
from ios_seg_infer import load_mesh  # noqa: E402
from ios_canonical import canonical_transform, to_viewer_axes  # noqa: E402
from teeth3ds_preprocess import (N_POINTS, SCALE_MM, pos_to_fdi,
                                 robust_center)  # noqa: E402

# Упрощаем челюсть ЦЕЛИКОМ до этого числа граней, и только потом режем на
# части. Если резать первым, а упрощать каждую часть отдельно, границы зуба и
# десны расползаются каждая по-своему и начинают налезать друг на друга —
# на модели это выглядит как белые пятна поверх розовых и наоборот.
JAW_FACES = 70000
# Голосование по разным разбивкам облака (см. predict_scan). 21, а не 7:
# замер на клиническом скане показал, что при 7 раундах порог качества не
# проходят 5 зубов, при 21 — четыре, и результат одинаков для разных сидов
# (то есть дело не в случайности разбиений, а в их числе). Сборка кейса
# офлайновая и редкая, лишние две минуты терпимы.
TTA_ROUNDS = 21
MIN_FACES = 120         # меньше — это шум сегментации, а не зуб
SPECK_FACES = 25        # компонента мельче — ошмёток, отдаём десне
SPECK_FRACTION = 0.04   # ... или мельче этой доли от крупнейшей у зуба
# Кусок дальше этого от основного тела зуба — не его, каким бы крупным ни был.
# Отсеивать по РАЗМЕРУ нельзя (см. submesh_kept), а по расстоянию можно: замер
# по 776 вторичным компонентам на 30 сканах Teeth3DS с эталонной разметкой —
# у своих кусков зазор до тела медиана 0.23 мм, у чужих 2.28 мм. Порог 2.0
# убирает 55% чужих ценой 9% площади вторичных компонент (основное тело зуба
# не трогается никогда).
COMPONENT_GAP_MM = 2.0


CHUNK_POINTS = 8192      # перезаписывается из чекпойнта в main()


def segment_full(mesh_path: Path, jaw: str, model, device):
    """-> (вершины мм, грани, метка FDI на каждую вершину)"""
    verts, normals = load_mesh(mesh_path)
    center = robust_center(verts)

    # Предсказываем по ВСЕМ вершинам, а не по подвыборке в 24000 с последующим
    # переносом ближайшим соседом: перенос размазывал границу зуб/десна на
    # несколько миллиметров, и на коронках оставались розовые клинья.
    # predict_scan режет облако на куски по 8192 точки — ровно столько сеть
    # видела при обучении, и плотность выборки получается той же.
    pts = ((verts - center) / SCALE_MM).astype(np.float32)
    nrm = normals.astype(np.float32)

    # Ориентация: Teeth3DS хранит челюсть коронками вверх, Exocad и Bits2Bites
    # анатомически, коронками вниз. Разница 180°, аугментация поворотами при
    # обучении узкая (+-20°), поэтому скан надо развернуть. Метки идут по
    # вершинам, порядок не меняется — обратно поворачивать ничего не нужно.
    R = canonical_transform(model, pts, nrm, device, CHUNK_POINTS, jaw=jaw)
    if not np.allclose(R, np.eye(3)):
        print(f"   ориентация: скан развёрнут (diag {np.diag(R).astype(int)})")
    pts, nrm = (pts @ R.T).astype(np.float32), (nrm @ R.T).astype(np.float32)
    # ГЕОМЕТРИЮ ЭТИМ ПОВОРОТОМ ТРОГАТЬ НЕЛЬЗЯ, хотя соблазн есть: разные
    # сканеры кладут скан в разные оси, и для совмещения с КЛКТ хотелось бы
    # общей системы. Но канонизация считается ДЛЯ КАЖДОЙ ЧЕЛЮСТИ ОТДЕЛЬНО и
    # даёт им разные матрицы (у нашего клинического скана верхняя получает
    # разворот, нижняя — тождество). Повернув геометрию, мы развалим взаимное
    # положение челюстей, а на нём держится вся карта окклюзионных контактов.
    # Ориентацию под КЛКТ ищет Kabsch при совмещении — ему поворот не помеха.

    lab_all = predict_scan(model, pts, nrm, device,
                           chunk=CHUNK_POINTS, rounds=TTA_ROUNDS)
    fdi = np.array([pos_to_fdi(int(p), jaw) if p > 0 else 0 for p in range(N_CLASSES)],
                   dtype=np.int32)[lab_all]

    m = trimesh.load(str(mesh_path), process=False, force="mesh")
    faces = np.asarray(m.faces)

    # OBJ из Exocad идёт с текстурой, а такие файлы дублируют вершины по швам
    # развёртки: у этого скана 158 755 вершин при 138 293 уникальных позициях.
    # Топологически поверхность оказывается разрезана на 1205 кусков, и любой
    # анализ связности рассыпается. trimesh.merge_vertices() швы бережёт (ему
    # важны UV), поэтому сливаем по координатам сами.
    key = np.round(verts, 5)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    if len(first) < len(verts):
        print(f"   слито дублей вершин: {len(verts) - len(first):,}")
        verts, fdi, faces = verts[first], fdi[first], inv[faces]
    return verts, faces, fdi


def decimate_jaw(verts, faces, fdi, target):
    """Упрощает челюсть целиком и переносит метки на новые вершины.

    Именно здесь, ДО разрезания на зубы и десну. Тогда все куски выходят из
    одной упрощённой поверхности и делят общие вершины на стыках.
    """
    if len(faces) <= target:
        return verts, faces, fdi
    import fast_simplification
    v2, f2 = fast_simplification.simplify(
        verts, faces, target_reduction=1.0 - target / len(faces))
    v2 = np.asarray(v2, np.float64)
    f2 = np.asarray(f2, np.int64)
    _, nn = cKDTree(verts).query(v2, k=1, workers=-1)
    return v2, f2, fdi[nn]


def submesh(verts, faces, mask):
    """Кусок сетки по маске граней. Без упрощения — оно уже сделано."""
    f = faces[mask]
    if len(f) < MIN_FACES:
        return None
    used = np.unique(f)
    remap = np.full(verts.shape[0], -1, np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(verts[used], remap[f], process=False)


CREASE_COS = 0.86       # нормали расходятся больше ~30° — это перегиб
CREASE_W = 0.05         # вес ребра через вогнутый перегиб


def crease_weights(verts, normals, e):
    """Вес ребра: маленький, если ребро идёт через ВОГНУТЫЙ перегиб.

    Шейка зуба — это выраженная вогнутая борозда там, где коронка выходит из
    десны. Если запретить меткам перетекать через такие рёбра, граница
    сегментации сама садится в анатомическую шейку вместо того, чтобы
    ползти по коронке.
    """
    d = verts[e[:, 1]] - verts[e[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    # нормали сходятся вдоль ребра -> впадина, расходятся -> выпуклость
    concave = ((normals[e[:, 1]] - normals[e[:, 0]]) * d).sum(1) < 0
    cosang = (normals[e[:, 0]] * normals[e[:, 1]]).sum(1).clip(-1, 1)
    w = np.ones(len(e), np.float32)
    w[concave & (cosang < CREASE_COS)] = CREASE_W
    return w


def smooth_labels(faces, fdi, n_verts, iters=4, verts=None, normals=None):
    """Голосование по соседям на сетке: убирает крапчатость меток.

    Сеть метит вершины независимо, и на границах зубов получается соль с
    перцем. Если резать сетку по таким меткам, куски выходят рваные, а после
    упрощения обрастают шипами. Сглаживаем метки ДО резки.

    Если переданы verts и normals, рёбра через вогнутые перегибы получают
    малый вес — тогда сглаживание не размывает границу, а наоборот подтягивает
    её к шейке зуба (см. crease_weights).
    """
    from scipy import sparse
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    w = (crease_weights(verts, normals, e)
         if verts is not None and normals is not None
         else np.ones(len(e), np.float32))
    adj = sparse.coo_matrix((w, (e[:, 0], e[:, 1])),
                            shape=(n_verts, n_verts)).tocsr()
    adj = adj + sparse.eye(n_verts, format="csr", dtype=np.float32)

    codes, inv = np.unique(fdi, return_inverse=True)
    cur = inv
    for _ in range(iters):
        onehot = sparse.csr_matrix(
            (np.ones(n_verts, np.float32), (np.arange(n_verts), cur)),
            shape=(n_verts, len(codes)))
        cur = np.asarray((adj @ onehot).todense()).argmax(axis=1)
    return codes[cur]


def vertex_normals(verts, faces):
    return np.asarray(trimesh.Trimesh(verts, faces, process=False).vertex_normals)


def paint_natural(scene, sources):
    """Положить в сегментированную модель НАСТОЯЩИЙ цвет со скана.

    ЗАЧЕМ. Во вкладке сегментации зубы и десна раскрашены условно — по номерам
    и ролям. Это нужно, чтобы отличать зубы друг от друга, но врач хочет
    смотреть на реальный вид, не теряя фильтров (скрыть верхнюю челюсть,
    скрыть десну, погасить отдельный зуб). Поэтому цвет кладётся в вершины
    ЗДЕСЬ, а вьюер переключает: показывать условную раскраску или натуральную.

    ПОЧЕМУ ПЕРЕНОСОМ, А НЕ ПРЯМО ИЗ ТЕКСТУРЫ. Сегментированный меш упрощён
    (decimate) и разрезан на зубы, его вершины не совпадают с вершинами
    исходного OBJ, и своих UV у них нет. Берём цвет ближайшей вершины
    исходника: они плотные (медиана расстояния между соседями заметно меньше
    размера фрагмента текстуры), поэтому перенос не размывает границы.

    Цвет исходника считается тем же способом, что и для вкладки «Исходный
    скан» — средним по площади фрагмента атласа, иначе на мелкой развёртке
    выборка шумит (см. `ios_build_raw._face_average_colors`).
    """
    from scipy.spatial import cKDTree
    import ios_build_raw as ibr

    pts, cols = [], []
    for path in sources:
        if not path:
            continue
        m = trimesh.load(str(path), process=False, force="mesh")
        img = getattr(getattr(m.visual, "material", None), "image", None)
        uv = getattr(m.visual, "uv", None)
        if img is None or uv is None:
            print(f"   {Path(path).name}: текстуры нет, натуральный цвет пропущен")
            continue
        tex = np.asarray(img.convert("RGB"))
        c = ibr._face_average_colors(tex, np.asarray(uv, np.float64),
                                     np.asarray(m.faces))
        c = ibr._smooth_vertex_colors(c, np.asarray(m.faces), len(m.vertices),
                                      iters=5)
        pts.append(np.asarray(m.vertices))
        cols.append(c)
    if not pts:
        return False

    tree = cKDTree(np.vstack(pts))
    src_col = np.vstack(cols)
    for geom in scene.geometry.values():
        _, idx = tree.query(np.asarray(geom.vertices), workers=-1)
        rgba = np.empty((len(idx), 4), np.uint8)
        rgba[:, :3] = src_col[idx]
        rgba[:, 3] = 255
        geom.visual = trimesh.visual.ColorVisuals(geom, vertex_colors=rgba)
    print(f"   натуральный цвет перенесён на {len(scene.geometry)} узлов")
    return True


def add_jaw(scene, verts, faces, fdi, jaw, info):
    fdi = smooth_labels(faces, fdi, len(verts), iters=6, verts=verts,
                        normals=vertex_normals(verts, faces))
    verts, faces, fdi = decimate_jaw(verts, faces, fdi, JAW_FACES)
    fdi = smooth_labels(faces, fdi, len(verts), iters=2, verts=verts,
                        normals=vertex_normals(verts, faces))
    face_lab = fdi[faces]                       # (F, 3)
    # грань относим к зубу, если хотя бы два её угла на этом зубе
    lab = np.zeros(len(faces), np.int32)
    for col in range(3):
        same = ((face_lab[:, col] == face_lab[:, (col + 1) % 3]) &
                (face_lab[:, col] != 0))
        lab[same] = face_lab[same, col]

    # Копим грани, реально ушедшие в зубы. Всё остальное обязано достаться
    # десне — иначе выброшенные мелкие компоненты оставляют в сетке дыры.
    taken = np.zeros(len(faces), bool)
    for tooth in sorted(set(int(x) for x in np.unique(lab) if x != 0)):
        m = lab == tooth
        mesh, kept = submesh_kept(verts, faces, m)
        if mesh is None:
            continue
        taken |= kept
        scene.add_geometry(mesh, geom_name=f"tooth_{tooth}")
        info["teeth"].append({"fdi": tooth, "jaw": jaw,
                              "faces": int(len(mesh.faces))})
    gum = submesh(verts, faces, ~taken)
    if gum is not None:
        scene.add_geometry(gum, geom_name=f"gum_{jaw}")
        info["gum"][jaw] = int(len(gum.faces))


def submesh_kept(verts, faces, mask):
    """Меш зуба + маска взятых граней (остаток уходит десне, без дыр).

    Оставлять ТОЛЬКО крупнейшую компоненту нельзя: поверхность зуба на скане
    распадается на несколько кусков, и жёсткий фильтр съедал до 70% зуба —
    у зуба 11 из 1612 граней оставалось 490, а остальное уходило в десну и
    коронка получалась розовой. Крапчатость уже снята сглаживанием меток,
    поэтому здесь достаточно отсечь совсем мелкие ошмётки.
    """
    idx_all = np.flatnonzero(mask)
    if len(idx_all) < MIN_FACES:
        return None, np.zeros(len(faces), bool)
    f = faces[idx_all]
    used = np.unique(f)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    mesh = trimesh.Trimesh(verts[used], remap[f], process=False)

    comp = trimesh.graph.connected_component_labels(mesh.face_adjacency,
                                                    node_count=len(mesh.faces))
    sizes = np.bincount(comp)
    keep_min = max(SPECK_FACES, int(sizes.max() * SPECK_FRACTION))
    ok = set(np.flatnonzero(sizes >= keep_min).tolist())

    # Отсев по расстоянию: крупный кусок в стороне от зуба — это соседний зуб
    # или десна, которые сегментация прицепила по ошибке. На клиническом скане
    # так к зубу 45 попал лоскут в 2 мм от тела, раздувавший его габарит на
    # 4.4 мм. Размер тут не помощник: свой кусок бывает не меньше чужого.
    main = int(np.argmax(sizes))
    MV, MF = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    main_pts = MV[np.unique(MF[comp == main])]
    tree = cKDTree(main_pts)
    for l in list(ok):
        if l == main:
            continue
        pts = MV[np.unique(MF[comp == l])]
        if float(tree.query(pts)[0].min()) > COMPONENT_GAP_MM:
            ok.discard(l)
    sel = np.isin(comp, sorted(ok))
    if sel.sum() < MIN_FACES:
        return None, np.zeros(len(faces), bool)

    kept = np.zeros(len(faces), bool)
    kept[idx_all[sel]] = True
    mesh = submesh(verts, faces, kept)
    return mesh, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper")
    ap.add_argument("--lower")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="research/output/ios_runs/dgcnn_v3/best.pt")
    args = ap.parse_args()

    # Разбивка облака на куски в predict_scan случайна, поэтому две сборки
    # одного и того же скана давали слегка разную сегментацию. Кейс — это
    # клинический артефакт: пересборка обязана повторяться в точности.
    np.random.seed(0)

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DGCNNSeg(6, N_CLASSES, ck["args"]["k"], ck["args"].get("width", 1)).to(device).eval()
    model.load_state_dict(ck["model"])
    global CHUNK_POINTS
    CHUNK_POINTS = int(ck["args"].get("points", CHUNK_POINTS))

    scene = trimesh.Scene()
    info = {"teeth": [], "gum": {}}
    for jaw, path in (("upper", args.upper), ("lower", args.lower)):
        if not path:
            continue
        verts, faces, fdi = segment_full(Path(path), jaw, model, device)
        print(f"{jaw}: вершин={len(verts):,} граней={len(faces):,} "
              f"зубов={len(set(int(x) for x in np.unique(fdi) if x))}")
        add_jaw(scene, verts, faces, fdi, jaw, info)

    # Натуральный цвет переносим ДО поворота осей: исходный OBJ лежит в
    # координатах сканера, и дерево ближайших точек должно строиться в них же.
    paint_natural(scene, (args.upper, args.lower))

    # оси сканера -> оси three.js: поворот, а не зеркало (см. to_viewer_axes)
    for geom in scene.geometry.values():
        geom.vertices = to_viewer_axes(geom.vertices)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    info["glb_mb"] = round(out.stat().st_size / 1e6, 2)
    (out.with_suffix(".json")).write_text(
        json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{out.name}: {info['glb_mb']} МБ, мешей={len(scene.geometry)}")
    print(f"зубы: {[t['fdi'] for t in info['teeth']]}")


if __name__ == "__main__":
    main()
