"""Совмещение интраорального скана с КЛКТ (superimposition).

Зачем. У скана точная коронка без металлических артефактов, у КЛКТ — корни,
кость и нерв. Планирование имплантации и ортодонтии идёт по обоим сразу:
коронка задаёт, куда ставить, кость — можно ли туда поставить. Врач сейчас
сводит их глазами на двух экранах.

Почему это НЕ слепой ICP. У нас с обеих сторон уже есть сегментация с
номерами зубов, то есть СООТВЕТСТВИЕ ИЗВЕСТНО. Это меняет задачу: вместо
поиска совпадения по форме считается преобразование по парам одноимённых
зубов (Кабш), а ICP остаётся только уточнением. Слепой ICP на симметричной
дуге любит сойтись в зеркало — здесь такой развилки нет вовсе.

Ловушка, из-за которой пары нельзя брать целиком: у скана есть только
КОРОНКА, у КЛКТ — коронка вместе с корнем, и центр масс зуба на КЛКТ сидит
на несколько миллиметров глубже. Пары центроидов «как есть» дают
систематический сдвиг по оси зуба. Поэтому с КЛКТ берётся только коронковая
часть — доля CROWN_FRAC от длины зуба со стороны смыкания.

Пространство одно — ЭКРАННОЕ (то же, в котором лежат оба GLB): результат
прикладывается во вьюере матрицей как есть, без ещё одной перестановки осей
(на перестановках проект уже терял знак определителя, см. CLAUDE.md).

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_cbct_register.py --selftest client4
    .venv/Scripts/python.exe research/scripts/ios_cbct_register.py --cbct client4 --ios batir
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

IOS_CASES = Path(__file__).resolve().parents[2] / "research_ios" / "web_cases"

CROWN_FRAC = 0.45      # какая доля зуба на КЛКТ считается коронкой
MIN_PAIRS = 3          # меньше трёх пар — жёсткое преобразование не определено
ICP_ITERS = 30
ICP_TOL = 1e-4         # сходимость по изменению RMS, мм
MAX_PTS = 40000        # потолок точек на облако: больше не нужно, дольше в 10 раз
# Порог доверия к результату. У скана точность порядка 0.05 мм, у КЛКТ воксель
# 0.3 мм; совмещение точнее вокселя физически невозможно, а хуже 0.5 мм уже
# видно глазом на границе коронки.
GOOD_RMS_MM = 0.5
# Второй критерий, более честный: медиана расхождения ПО ЗУБАМ. Общий остаток
# ICP занижен отсечением худших пар, а зуб на своём месте либо стоит, либо нет.
GOOD_TOOTH_MM = 0.5

# Порча «скана» в самопроверке: шум сканера порядка 0.05 мм (паспортная
# точность интраоральных сканеров), и на скан попадают не все зубы.
SELFTEST_NOISE_MM = 0.05
SELFTEST_KEEP_FRAC = 0.7


def _to_viewer(pts_zyx: np.ndarray) -> np.ndarray:
    """(z, y, x) мм -> экранные (x, z, y). Та же перестановка, что в build_mesh
    (`[2, 0, 1]`, определитель +1 — это поворот, а не зеркало)."""
    return pts_zyx[:, [2, 0, 1]]


def _from_viewer(pts_viewer: np.ndarray) -> np.ndarray:
    """Обратная перестановка: экранные (x, z, y) -> (z, y, x).

    Живёт рядом с прямой намеренно. Пара «туда-обратно», разнесённая по
    файлам, — верный способ однажды перепутать порядок осей и получить
    зеркало: в этом проекте на перестановках осей уже дважды теряли стороны.
    """
    return pts_viewer[:, [1, 2, 0]]


def _surface(bm: np.ndarray) -> np.ndarray:
    """Воксели ПОВЕРХНОСТИ структуры (у которых есть сосед снаружи).

    ICP обязан сводить поверхность с поверхностью. Со сплошным облаком точек
    ближайшим соседом для точки скана оказывается воксель ВНУТРИ зуба, а не на
    его границе, и притяжение к правильному месту слабеет. Замер (самопроверка
    на client4, 5 сидов): по сплошным облакам ошибка сдвига доходила до
    0.60 мм при вокселе 0.31 — то есть ICP УХУДШАЛ и без того точное
    совмещение по центроидам.
    """
    from scipy import ndimage as ndi
    if not bm.any():
        return bm
    idx = np.nonzero(bm)
    lo = [max(int(i.min()) - 1, 0) for i in idx]
    hi = [min(int(i.max()) + 2, s) for i, s in zip(idx, bm.shape)]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = bm[box]
    inner = ndi.binary_erosion(sub, ndi.generate_binary_structure(3, 1))
    out = np.zeros_like(bm)
    out[box] = sub & ~inner
    return out


def cbct_crowns(case_id: str, model: str | None = None) -> dict:
    """Точки ПОВЕРХНОСТИ коронок зубов КЛКТ по номерам FDI, в экранных мм."""
    c = cb.get_case(case_id)
    if c is None:
        raise SystemExit(f"нет кейса КЛКТ {case_id}")
    model = model or cb.mask_model_for(c)
    if model is None:
        raise SystemExit(f"у кейса {case_id} нет готовой маски")
    _vol, mask = cb.arrays_for(c, model)
    spacing = np.array(c["spacing"], float)
    z_up = cb.occlusal_up(mask)
    out = {}
    for t in cb.TEETH_CLASSES:
        bm = mask == t
        if int(bm.sum()) < 300:
            continue
        idx = np.array(np.nonzero(_surface(bm)))
        if idx.shape[1] < 100:
            continue
        pts = idx.T * spacing                    # (N, 3) мм в (z, y, x)
        # Коронковый конец по оси z: сторона зависит от ориентации тома и от
        # того, верхний это зуб или нижний.
        upper = t // 10 in (1, 2)
        crown_up = upper != z_up                 # коронка в сторону БОЛЬШИХ z?
        zc = pts[:, 0]
        lo, hi = zc.min(), zc.max()
        cut = (hi - CROWN_FRAC * (hi - lo)) if crown_up \
            else (lo + CROWN_FRAC * (hi - lo))
        sel = zc >= cut if crown_up else zc <= cut
        if sel.sum() < 100:
            continue
        out[t] = _to_viewer(pts[sel])
    return out


def ios_crowns(case_id: str) -> dict:
    """Точки зубов со скана по номерам FDI, в экранных мм (как лежит в GLB)."""
    import trimesh
    path = IOS_CASES / case_id / "scan.glb"
    if not path.is_file():
        raise SystemExit(f"нет скана {path}")
    scene = trimesh.load(str(path), process=False)
    out = {}
    for name, geom in scene.geometry.items():
        if not name.startswith("tooth_"):
            continue
        try:
            fdi = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        v = np.asarray(geom.vertices, dtype=float)
        if len(v) >= 50:
            out[fdi] = v
    return out


def _kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Жёсткое преобразование src -> dst по парам точек (4x4).

    Отражение запрещаем явно: при почти плоском наборе точек SVD легко даёт
    решение с определителем -1, и скан встаёт зеркально. На дуге зубов это
    выглядит правдоподобно — левая сторона садится на правую.
    """
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cd - R @ cs
    return T


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def _subsample(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), n, replace=False)]


def icp(src: np.ndarray, dst: np.ndarray, T0: np.ndarray,
        iters: int = ICP_ITERS) -> tuple:
    """Уточнение совмещения: ближайшая точка -> Кабш, до сходимости RMS.

    Отсекаем 20% худших пар на каждой итерации: у скана есть десна, у КЛКТ её
    нет вовсе, и эти точки тянут решение на себя.
    """
    T = T0.copy()
    tree = cKDTree(dst)
    prev = None
    for _ in range(iters):
        dist, j = tree.query(_apply(T, src), k=1)
        keep = dist <= np.percentile(dist, 80)
        if keep.sum() < MIN_PAIRS:
            break
        T = _kabsch(src[keep], dst[j[keep]])
        rms = float(np.sqrt(np.mean(dist[keep] ** 2)))
        if prev is not None and abs(prev - rms) < ICP_TOL:
            break
        prev = rms
    # Остаток считаем ПОСЛЕ последнего обновления: внутри цикла он относится к
    # предыдущей матрице, и отчёт врал бы на одну итерацию в худшую сторону.
    dist, _ = tree.query(_apply(T, src), k=1)
    keep = dist <= np.percentile(dist, 80)
    return T, float(np.sqrt(np.mean(dist[keep] ** 2)))


def register(cbct_case: str, ios_case: str, model: str | None = None) -> dict:
    """Совмещение скана с КЛКТ, ПО КАЖДОЙ ЧЕЛЮСТИ ОТДЕЛЬНО.

    Раздельно — потому что челюсти подвижны друг относительно друга. Верхняя
    жёстко связана с черепом, нижняя нет: в КЛКТ пациент лежит с одним
    положением, скан снимают с другим, и одно преобразование на оба ряда ищет
    компромисс между двумя РАЗНЫМИ положениями. Ошибка при этом размазывается
    по всем зубам, и совмещение не проходит порог даже когда пациент один.

    Замер на клиническом кейсе (30 общих зубов, пациент заведомо один):

        обе челюсти вместе   RMS 0.61   медиана по зубам 2.29 мм
        только верхняя       RMS 0.48   медиана по зубам 0.51 мм
        только нижняя        RMS 0.56   медиана по зубам 0.71 мм

    То есть раздельный счёт точнее втрое-вчетверо, и он же физически верен.
    Вместе с этим уходит ложный вывод «кейсы от разных пациентов», который
    прежняя версия выдавала на СВОИХ данных.
    """
    tgt = cbct_crowns(cbct_case, model)
    src = ios_crowns(ios_case)
    common = sorted(set(tgt) & set(src))
    if len(common) < MIN_PAIRS:
        return {"ok": False, "common_teeth": common,
                "error": f"общих зубов {len(common)}, нужно минимум {MIN_PAIRS}"}

    jaws = {}
    for name, quads in (("upper", (1, 2)), ("lower", (3, 4))):
        teeth = [t for t in common if t // 10 in quads]
        if len(teeth) >= MIN_PAIRS:
            jaws[name] = _fit(src, tgt, teeth)

    out = _combined(cbct_case, ios_case, common, jaws)
    if out is not None:
        return out
    # Челюстей по отдельности не хватило (мало зубов) — считаем как раньше,
    # по всем общим сразу: лучше приблизительный ответ, чем никакого.

    # 1) грубо: по центроидам одноимённых зубов
    s_cen = np.array([src[t].mean(0) for t in common])
    t_cen = np.array([tgt[t].mean(0) for t in common])
    T = _kabsch(s_cen, t_cen)
    coarse = float(np.sqrt(np.mean(
        np.sum((_apply(T, s_cen) - t_cen) ** 2, axis=1))))

    # 2) точно: ICP по поверхностям только общих зубов
    s_pts = _subsample(np.vstack([src[t] for t in common]), MAX_PTS)
    # Цель НЕ прореживаем: это дерево поиска, и чем оно плотнее, тем точнее
    # ближайший сосед. Прореживание цели давало лишнюю ошибку на ровном месте.
    t_pts = np.vstack([tgt[t] for t in common])
    T, rms = icp(s_pts, t_pts, T)

    # 3) по каждому зубу отдельно — чтобы видеть, где именно разошлось
    per_tooth = {}
    for t in common:
        tree = cKDTree(tgt[t])
        d, _ = tree.query(_apply(T, _subsample(src[t], 4000)), k=1)
        per_tooth[t] = round(float(np.median(d)), 2)

    # Медиана ПО ЗУБАМ — главный критерий, а не общий RMS. Замер на заведомо
    # разных пациентах (client4 + batir): общий RMS вышел 1.02 мм, всего вдвое
    # выше порога, потому что ICP отсекает 20% худших пар и промахи прячутся.
    # По зубам в том же прогоне: 1.9, 2.5, ... 90.9, 109.6 мм. Дуги у людей
    # похожи, и слепо доверять общему остатку здесь нельзя.
    tooth_med = float(np.median(list(per_tooth.values()))) if per_tooth else 1e9
    return {
        "ok": True,
        "cbct_case": cbct_case, "ios_case": ios_case,
        "common_teeth": common,
        "matrix": [round(float(v), 6) for v in T.flatten()],
        "coarse_rms_mm": round(coarse, 2),
        "rms_mm": round(rms, 2),
        "per_tooth_median_mm": per_tooth,
        "tooth_median_mm": round(tooth_med, 2),
        "good": bool(rms <= GOOD_RMS_MM and tooth_med <= GOOD_TOOTH_MM),
        "note": ("Совмещение по одноимённым зубам: соответствие берётся из "
                 "сегментации, а не ищется по форме. С КЛКТ используется "
                 f"только коронковая часть ({CROWN_FRAC:.0%} длины зуба) — "
                 "у скана корня нет."),
        "draft": True,
    }


def _fit(src: dict, tgt: dict, teeth: list) -> dict:
    """Одно преобразование для одного набора зубов (одной челюсти)."""
    s_cen = np.array([src[t].mean(0) for t in teeth])
    t_cen = np.array([tgt[t].mean(0) for t in teeth])
    T = _kabsch(s_cen, t_cen)
    coarse = float(np.sqrt(np.mean(
        np.sum((_apply(T, s_cen) - t_cen) ** 2, axis=1))))
    s_pts = _subsample(np.vstack([src[t] for t in teeth]), MAX_PTS)
    t_pts = np.vstack([tgt[t] for t in teeth])
    T, rms = icp(s_pts, t_pts, T)
    per = {}
    for t in teeth:
        tree = cKDTree(tgt[t])
        d, _ = tree.query(_apply(T, _subsample(src[t], 4000)), k=1)
        per[t] = round(float(np.median(d)), 2)
    med = float(np.median(list(per.values()))) if per else 1e9
    return {
        "teeth": teeth,
        "matrix": [round(float(v), 6) for v in T.flatten()],
        "coarse_rms_mm": round(coarse, 2),
        "rms_mm": round(rms, 2),
        "per_tooth_median_mm": per,
        "tooth_median_mm": round(med, 2),
        "good": bool(rms <= GOOD_RMS_MM and med <= GOOD_TOOTH_MM),
    }


def _combined(cbct_case: str, ios_case: str, common: list, jaws: dict):
    """Свести результаты челюстей в один ответ.

    Общие поля (`matrix`, `rms_mm`, `tooth_median_mm`) остаются: их читают
    вьюер и API суперимпозиции. В них кладём ЛУЧШУЮ челюсть — для планирования
    имплантации работают с одной челюстью за раз, и брать среднее по двум
    разным положениям бессмысленно.
    """
    if not jaws:
        return None
    best_name = min(jaws, key=lambda k: jaws[k]["tooth_median_mm"])
    best = jaws[best_name]
    return {
        "ok": True,
        "cbct_case": cbct_case, "ios_case": ios_case,
        "common_teeth": common,
        "jaws": jaws,
        "best_jaw": best_name,
        "matrix": best["matrix"],
        "coarse_rms_mm": best["coarse_rms_mm"],
        "rms_mm": best["rms_mm"],
        "per_tooth_median_mm": best["per_tooth_median_mm"],
        "tooth_median_mm": best["tooth_median_mm"],
        "good": any(j["good"] for j in jaws.values()),
        "note": ("Совмещение считается ПО КАЖДОЙ ЧЕЛЮСТИ ОТДЕЛЬНО: челюсти "
                 "подвижны друг относительно друга, и в КЛКТ рот стоит не "
                 "так, как при сканировании. Соответствие берётся из "
                 "сегментации по одноимённым зубам, а не ищется по форме; "
                 f"с КЛКТ используется коронковая часть ({CROWN_FRAC:.0%} "
                 "длины зуба) — у скана корня нет."),
        "draft": True,
    }


def _spacing_of(case_id: str) -> np.ndarray:
    """Размер вокселя кейса: точнее него совместить физически невозможно."""
    c = cb.get_case(case_id)
    return np.array(c["spacing"], float)


def selftest(cbct_case: str, model: str | None = None, seed: int = 0) -> dict:
    """Проверка без пары «тот же пациент»: коронки КЛКТ вместо скана.

    Пары «скан + КЛКТ одного пациента» у клиники ещё нет, поэтому настоящую
    точность измерить НЕ НА ЧЕМ. Проверяем то, что проверить можно: берём
    коронки самой томограммы, сдвигаем и поворачиваем на известную величину и
    смотрим, восстановит ли метод преобразование обратно.

    Копию «как есть» подсовывать бессмысленно — ICP тогда сводит каждую точку
    саму с собой и выдаёт ровный ноль (проверено: 0.000 мм на пяти сидах).
    Поэтому «скан» портится тремя способами, как портит его реальность:
    шумом сканера (SELFTEST_NOISE_MM), потерей части зубов (у скана они
    попадают не все) и прореживанием точек. Это проверка МАТЕМАТИКИ под
    реалистичной порчей, а не работы на настоящем скане: там добавятся другой
    аппарат, десна и ошибки обеих сегментаций.
    """
    tgt = cbct_crowns(cbct_case, model)
    if len(tgt) < MIN_PAIRS:
        return {"ok": False, "error": "мало зубов в маске"}
    rng = np.random.default_rng(seed)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = np.deg2rad(rng.uniform(5, 20))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    shift = rng.uniform(-8, 8, size=3)
    T_true = np.eye(4)
    T_true[:3, :3] = R
    T_true[:3, 3] = shift

    # «скан» — коронки, сдвинутые известным преобразованием (обратным),
    # с шумом и неполным набором зубов
    T_inv = np.linalg.inv(T_true)
    teeth = sorted(tgt)
    keep_n = max(MIN_PAIRS, int(round(len(teeth) * SELFTEST_KEEP_FRAC)))
    keep = sorted(rng.choice(teeth, size=keep_n, replace=False).tolist())
    src = {}
    for t in keep:
        p = _apply(T_inv, tgt[t])
        p = p + rng.normal(scale=SELFTEST_NOISE_MM, size=p.shape)
        src[t] = _subsample(p, max(len(p) // 3, 100), seed=seed)
    common = sorted(set(tgt) & set(src))
    s_cen = np.array([src[t].mean(0) for t in common])
    t_cen = np.array([tgt[t].mean(0) for t in common])
    T = _kabsch(s_cen, t_cen)
    s_pts = _subsample(np.vstack([src[t] for t in common]), MAX_PTS)
    t_pts = np.vstack([tgt[t] for t in common])
    T, rms = icp(s_pts, t_pts, T)

    err = T @ np.linalg.inv(T_true)
    ang_err = float(np.degrees(np.arccos(
        np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))))
    shift_err = float(np.linalg.norm(err[:3, 3]))
    return {"ok": True, "teeth": len(common), "of_teeth": len(tgt),
            "rms_mm": round(rms, 3),
            "voxel_mm": round(float(np.max(_spacing_of(cbct_case))), 2),
            "angle_error_deg": round(ang_err, 3),
            "shift_error_mm": round(shift_err, 3),
            "true_angle_deg": round(float(np.degrees(ang)), 1),
            "true_shift_mm": round(float(np.linalg.norm(shift)), 1)}


def save(res: dict) -> Path:
    """Кладём результат к кейсу скана: вьюер КЛКТ читает его оттуда."""
    d = IOS_CASES / res["ios_case"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f'register_{res["cbct_case"]}.json'
    p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cbct")
    ap.add_argument("--ios")
    ap.add_argument("--model", default=None)
    ap.add_argument("--selftest", metavar="CBCT_CASE",
                    help="проверка математики на коронках самой томограммы")
    ap.add_argument("--seeds", type=int, default=5, help="прогонов самопроверки")
    a = ap.parse_args()

    if a.selftest:
        print(f"САМОПРОВЕРКА на {a.selftest}: коронки КЛКТ сдвигаются известным "
              f"преобразованием и совмещаются обратно\n")
        print(f"порча «скана»: шум {SELFTEST_NOISE_MM} мм, остаётся "
              f"{SELFTEST_KEEP_FRAC:.0%} зубов и треть точек\n")
        print(f"{'сид':>4} {'зубов':>7} {'RMS мм':>8} {'ошибка угла':>12} "
              f"{'ошибка сдвига':>14}   истинные")
        bad, vox = 0, None
        for s in range(a.seeds):
            r = selftest(a.selftest, a.model, seed=s)
            if not r["ok"]:
                raise SystemExit(r["error"])
            vox = r["voxel_mm"]
            # Судим по ФИЗИЧЕСКОМУ пределу: точнее вокселя КЛКТ совмещать
            # нечем, поэтому порог — воксель, а не произвольная цифра.
            if r["shift_error_mm"] > vox or r["angle_error_deg"] > 1.0:
                bad += 1
            print(f"{s:>4} {r['teeth']:>3}/{r['of_teeth']:<3} "
                  f"{r['rms_mm']:>8.3f} "
                  f"{r['angle_error_deg']:>11.3f}° {r['shift_error_mm']:>13.3f} "
                  f"мм   {r['true_angle_deg']}° / {r['true_shift_mm']} мм")
        print(f"\nхуже вокселя ({vox} мм) или 1°: {bad} из {a.seeds}")
        return

    if not (a.cbct and a.ios):
        raise SystemExit("нужны --cbct и --ios, либо --selftest CBCT_CASE")
    res = register(a.cbct, a.ios, a.model)
    if not res["ok"]:
        raise SystemExit(res["error"])
    print(f"общих зубов: {len(res['common_teeth'])} — {res['common_teeth']}")
    print(f"по центроидам: {res['coarse_rms_mm']} мм -> после ICP: "
          f"{res['rms_mm']} мм; медиана по зубам {res['tooth_median_mm']} мм")
    print("по зубам (медиана расстояния, мм):")
    for t, v in sorted(res["per_tooth_median_mm"].items()):
        print(f"  {t}: {v}")
    if res.get("jaws"):
        print("\nпо челюстям (считаются раздельно — они подвижны друг "
              "относительно друга):")
        for name, j in res["jaws"].items():
            ru = "верхняя" if name == "upper" else "нижняя"
            print(f"  {ru}: зубов {len(j['teeth'])}, RMS {j['rms_mm']} мм, "
                  f"медиана по зубам {j['tooth_median_mm']} мм — "
                  f"{'принято' if j['good'] else 'не принято'}")
    if not res["good"]:
        # Прежняя формулировка обвиняла в «разных пациентах» при ЛЮБОМ
        # непрохождении порога, и врач получил это на своих данных при
        # расхождении 0.51 мм против порога 0.50. Порог здесь — защита: при
        # 0.5 проходит лишь четверть верных пар, зато ни одной чужой. Значит
        # само непрохождение ничего не доказывает, важно НАСКОЛЬКО не прошли.
        med, rms = res["tooth_median_mm"], res["rms_mm"]
        print(f"\nНЕ ПРИНЯТО порогом: RMS {rms} мм (порог {GOOD_RMS_MM}), "
              f"медиана по зубам {med} мм (порог {GOOD_TOOTH_MM}).")
        if med <= GOOD_TOOTH_MM * 2:
            print("Расхождение близко к порогу — это НЕ признак разных "
                  "пациентов. Порог намеренно строгий: пропускает лишь "
                  "четверть верных совмещений, зато не принимает чужие.")
        else:
            print("Расхождение велико. Возможные причины: кейсы от разных "
                  "пациентов, развалившаяся сегментация одного из них, либо "
                  "мало общих зубов.")
    print(f"\nсохранено: {save(res)}")


if __name__ == "__main__":
    main()
