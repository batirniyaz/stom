"""Проверка: какая вертикаль дуги даёт измерения прикуса, различающие диагнозы.

Опыт с одной переменной. Измерения прикуса (перекрытие резцов, смещение
моляров, средняя линия) считаются по осям дуги; оси строятся одинаково, кроме
ВЕРТИКАЛИ, а она берётся двумя способами:

    old   разница центров верхних и нижних зубов — как в ios_occlusion.build_frame
    new   перпендикуляр к плоскости дуги (PCA центров) — как в ios_geometry

На клиническом скане второй способ отклоняется от нормали окклюзионной
плоскости на 0.8°, а первый на 28.8°. Здесь проверяем, влияет ли это на то,
разделяют ли измерения категории врача из Bits2Bites.

Мера разделения — AUC (доля пар «больной/здоровый», упорядоченных верно):
0.5 — измерение бесполезно, 1.0 — разделяет идеально. Медианы, которыми
меряли в прошлый раз, этого не показывают: они могут совпадать при хорошем
разделении и расходиться при плохом.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_frame_validate.py --limit 8
    .venv/Scripts/python.exe research/scripts/ios_frame_validate.py --json out.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ios_geometry as G  # noqa: E402
from ios_seg_train import DGCNNSeg, N_CLASSES  # noqa: E402
from ios_occlusion import segment_jaw, tooth_points, MIN_PTS  # noqa: E402
import ios_build_mesh as BM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "research_ios" / "datasets" / "Bits2Bites"
DATA = ROOT / "data"
CSV = DATA / "Annotations.csv"


def case_pipeline_points(path: Path, jaw: str, model, device) -> dict:
    """Точки зубов ровно так, как их видит кейс: -> {fdi: вершины}.

    Повторяет ios_build_mesh.add_jaw: сглаживание меток, упрощение челюсти
    целиком, разрезание на зубы с отсевом ошмётков. Нужно потому, что пороги
    прикуса калибровались на СЫРОЙ сегментации, а demo считает по очищенному
    мешу, и на клиническом скане смещение моляров расходилось на 1.5 мм —
    достаточно, чтобы сменился класс по Энглю.
    """
    verts, faces, fdi = BM.segment_full(path, jaw, model, device)
    nrm = BM.vertex_normals(verts, faces)
    fdi = BM.smooth_labels(faces, fdi, len(verts), iters=6, verts=verts,
                           normals=nrm)
    verts, faces, fdi = BM.decimate_jaw(verts, faces, fdi, BM.JAW_FACES)
    nrm = BM.vertex_normals(verts, faces)
    fdi = BM.smooth_labels(faces, fdi, len(verts), iters=2, verts=verts,
                           normals=nrm)
    face_lab = fdi[faces]
    lab = np.zeros(len(faces), np.int32)
    for col in range(3):
        same = ((face_lab[:, col] == face_lab[:, (col + 1) % 3])
                & (face_lab[:, col] != 0))
        lab[same] = face_lab[same, col]
    out = {}
    for tooth in sorted(int(x) for x in np.unique(lab) if x != 0):
        mesh, _ = BM.submesh_kept(verts, faces, lab == tooth)
        if mesh is not None:
            out[int(tooth)] = np.asarray(mesh.vertices)
    return out


def centroids(verts, fdi, lo, hi) -> dict:
    out = {}
    for t in range(lo, hi):
        p = tooth_points(verts, fdi, t)
        if p is not None:
            out[t] = p.mean(0)
    return out


def frame_axes(cu: dict, cl: dict, how: str):
    """Оси дуги. Различается ТОЛЬКО вертикаль, всё остальное одинаково."""
    cen = {**cu, **cl}
    if how == "old":
        up = np.mean(list(cu.values()), axis=0) - np.mean(list(cl.values()), axis=0)
        up = up / np.linalg.norm(up)
    else:
        P = np.array(list(cen.values()))
        _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
        up = Vt[2]
        if (np.mean(list(cu.values()), axis=0)
                - np.mean(list(cl.values()), axis=0)) @ up < 0:
            up = -up

    right = [c for t, c in cen.items() if t // 10 in (1, 4)]
    left = [c for t, c in cen.items() if t // 10 in (2, 3)]
    lat = np.mean(left, axis=0) - np.mean(right, axis=0)
    lat -= up * (lat @ up)
    lat /= np.linalg.norm(lat)

    fwd = np.cross(lat, up)
    fwd /= np.linalg.norm(fwd)
    inc = [c for t, c in cen.items() if t % 10 <= 2]
    mol = [c for t, c in cen.items() if t % 10 >= 6]
    if inc and mol and (np.mean(inc, axis=0) - np.mean(mol, axis=0)) @ fwd < 0:
        fwd = -fwd
    return np.stack([lat, fwd, up])


def measure(F, vu, fu, vl, fl) -> dict:
    """Те же три измерения, что в ios_occlusion.measure, но с заданными осями."""
    lat, fwd, up = F
    proj = lambda p: np.stack([p @ lat, p @ fwd, p @ up], axis=-1)
    out = {}

    ui = [p for t in (11, 21) if (p := tooth_points(vu, fu, t)) is not None]
    li = [p for t in (41, 31) if (p := tooth_points(vl, fl, t)) is not None]
    if ui and li:
        pu, pl = proj(np.vstack(ui)), proj(np.vstack(li))
        out["overbite"] = float(np.percentile(pl[:, 2], 98)
                                - np.percentile(pu[:, 2], 2))
        out["overjet"] = float(np.percentile(pu[:, 1], 98)
                               - np.percentile(pl[:, 1], 98))
        out["midline"] = float(pu[:, 0].mean() - pl[:, 0].mean())

    for side, u6, l6 in (("right", 16, 46), ("left", 26, 36)):
        pu, pl = tooth_points(vu, fu, u6), tooth_points(vl, fl, l6)
        if pu is not None and pl is not None:
            out[f"molar_{side}"] = float(proj(pu).mean(0)[1] - proj(pl).mean(0)[1])
    return out


def measure_landmarks(frame, pts: dict, seq: dict) -> dict:
    """Класс Энгля и средняя линия по АНАТОМИЧЕСКИМ ориентирам, не центроидам.

    Класс: мезиально-щёчный бугор верхней шестёрки против щёчной борозды
    нижней. В норме первый стоит ровно во второй, поэтому база измерения по
    построению НУЛЕВАЯ — в отличие от разницы центроидов, где норма оказалась
    −1.9 мм. Знак тот же: плюс — бугор впереди борозды (II класс).

    Средняя линия: по мезиальным контактным пунктам центральных резцов.
    """
    out = {}
    for side, u6, l6 in (("right", 16, 46), ("left", 26, 36)):
        if u6 not in pts or l6 not in pts:
            continue
        try:
            au = G.tooth_axes(u6, frame, seq)
            al = G.tooth_axes(l6, frame, seq)
        except (ValueError, KeyError, IndexError):
            continue
        mb = G.landmark(pts[u6], au, u6, "MBCusp")
        fa = G.landmark(pts[l6], al, l6, "FacialPoint")
        if mb is None or fa is None:
            continue
        out[f"lm_molar_{side}"] = float((mb - fa) @ frame.fwd)

    up_pts, lo_pts = [], []
    for t, bag in ((11, up_pts), (21, up_pts), (41, lo_pts), (31, lo_pts)):
        if t not in pts:
            continue
        try:
            ax = G.tooth_axes(t, frame, seq)
        except (ValueError, KeyError, IndexError):
            continue
        c = G.landmark(pts[t], ax, t, "Mesial")
        if c is not None:
            bag.append(c)
    if len(up_pts) == 2 and len(lo_pts) == 2:
        d = (np.mean(up_pts, axis=0) - np.mean(lo_pts, axis=0)) @ frame.lat
        out["lm_midline"] = float(d)
        out["lm_midline_abs"] = abs(float(d))
    return out


def auc(pos: list, neg: list) -> float:
    """Доля пар, упорядоченных верно. 0.5 — измерение не различает группы."""
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    a, b = np.asarray(pos, float), np.asarray(neg, float)
    wins = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return float(wins / (len(a) * len(b)))


# Что с чем сравниваем: (подпись, измерение, категория, «больные», «здоровые»,
# знак). Знак +1 — у «больных» величина должна быть больше, -1 — меньше.
CONTRASTS = [
    ("Глубокий прикус против нормы", "overbite", "Anterior Bite",
     ["Deep Bite"], ["Normal"], +1),
    ("Открытый прикус против нормы", "overbite", "Anterior Bite",
     ["Open Bite"], ["Normal"], -1),
    ("II класс справа против I", "molar_right", "Right Class",
     ["Class II Full", "Class II Edge to Edge"], ["Class I"], +1),
    ("III класс справа против I", "molar_right", "Right Class",
     ["Class III"], ["Class I"], -1),
    ("II класс слева против I", "molar_left", "Left Class",
     ["Class II Full", "Class II Edge to Edge"], ["Class I"], +1),
    ("III класс слева против I", "molar_left", "Left Class",
     ["Class III"], ["Class I"], -1),
    ("Смещение средней линии", "midline_abs", "Median Lines",
     ["Deviated"], ["Centered"], +1),
    # то же самое, но по анатомическим ориентирам вместо центроидов
    ("II класс справа, ориентиры", "lm_molar_right", "Right Class",
     ["Class II Full", "Class II Edge to Edge"], ["Class I"], +1),
    ("III класс справа, ориентиры", "lm_molar_right", "Right Class",
     ["Class III"], ["Class I"], -1),
    ("II класс слева, ориентиры", "lm_molar_left", "Left Class",
     ["Class II Full", "Class II Edge to Edge"], ["Class I"], +1),
    ("III класс слева, ориентиры", "lm_molar_left", "Left Class",
     ["Class III"], ["Class I"], -1),
    ("Средняя линия, ориентиры", "lm_midline_abs", "Median Lines",
     ["Deviated"], ["Centered"], +1),
]


def run(limit: int, ckpt: str, out_json: str | None,
        pipeline: str = "raw") -> None:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    device = torch.device("cuda")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = DGCNNSeg(6, N_CLASSES, ck["args"]["k"],
                     ck["args"].get("width", 1)).to(device).eval()
    model.load_state_dict(ck["model"])
    chunk = int(ck["args"].get("points", 8192))
    BM.CHUNK_POINTS = chunk
    print(f"путь расчёта: {pipeline}")

    done, skipped, t0 = [], 0, time.perf_counter()
    for row in rows:
        if limit and len(done) >= limit:
            break
        pid = row["Patient"]
        d = DATA / "train" / pid
        up_f, lo_f = d / "upper.stl", d / "lower.stl"
        if not (up_f.is_file() and lo_f.is_file()):
            skipped += 1
            continue
        try:
            if pipeline == "case":
                pu = case_pipeline_points(up_f, "upper", model, device)
                pl = case_pipeline_points(lo_f, "lower", model, device)
                cu = {t: v.mean(0) for t, v in pu.items() if t // 10 in (1, 2)}
                cl = {t: v.mean(0) for t, v in pl.items() if t // 10 in (3, 4)}
                # для measure() восстанавливаем вид «вершины + метки»
                vu = np.vstack(list(pu.values())) if pu else np.zeros((0, 3))
                fu = np.concatenate([np.full(len(v), t) for t, v in pu.items()]) \
                    if pu else np.zeros(0, int)
                vl = np.vstack(list(pl.values())) if pl else np.zeros((0, 3))
                fl = np.concatenate([np.full(len(v), t) for t, v in pl.items()]) \
                    if pl else np.zeros(0, int)
            else:
                vu, fu = segment_jaw(up_f, "upper", model, device, chunk)
                vl, fl = segment_jaw(lo_f, "lower", model, device, chunk)
                cu = centroids(vu, fu, 11, 29)
                cl = centroids(vl, fl, 31, 49)
            if len(cu) < 5 or len(cl) < 5 or len(cu) + len(cl) < 12:
                skipped += 1
                continue
            rec = {"patient": pid, "label": row}
            for how in ("old", "new"):
                rec[how] = measure(frame_axes(cu, cl, how), vu, fu, vl, fl)
                rec[how]["midline_abs"] = abs(rec[how].get("midline", np.nan))
            # ориентиры считаем в осях «new»: они уже проверены
            try:
                pts_all = {}
                for verts, fdi_arr, lo, hi in ((vu, fu, 11, 29), (vl, fl, 31, 49)):
                    for tt in range(lo, hi):
                        pp = tooth_points(verts, fdi_arr, tt)
                        if pp is not None:
                            pts_all[tt] = pp
                fr = G.arch_frame_from_centroids({**cu, **cl})
                rec["new"].update(measure_landmarks(
                    fr, pts_all, G.arch_sequence(pts_all)))
            except Exception:  # noqa: BLE001 — ориентиры не должны ронять прогон
                pass
            # угол между двумя вертикалями — насколько вообще различаются оси
            rec["tilt_deg"] = float(np.degrees(np.arccos(np.clip(abs(
                frame_axes(cu, cl, "old")[2] @ frame_axes(cu, cl, "new")[2]),
                -1, 1))))
            done.append(rec)
        except Exception as exc:  # noqa: BLE001 — один плохой скан не должен ронять прогон
            print(f"  пациент {pid}: пропуск ({exc})")
            skipped += 1
        if len(done) % 10 == 0 and done:
            el = time.perf_counter() - t0
            print(f"  обработано {len(done)}  ({el/len(done):.1f} с на пациента)",
                  flush=True)

    print(f"\nпациентов посчитано: {len(done)}, пропущено: {skipped}")
    tilts = [r["tilt_deg"] for r in done]
    print(f"угол между двумя вертикалями: медиана {np.median(tilts):.1f}°, "
          f"размах {min(tilts):.1f}..{max(tilts):.1f}°")

    print(f"\n{'сравнение':<34} {'n+':>4} {'n-':>4} {'AUC old':>9} "
          f"{'AUC new':>9} {'сдвиг':>8}")
    results = []
    for label, key, col, pos_lab, neg_lab, sign in CONTRASTS:
        line = {"contrast": label}
        for how in ("old", "new"):
            pos = [r[how][key] * sign for r in done
                   if r["label"][col] in pos_lab and key in r[how]]
            neg = [r[how][key] * sign for r in done
                   if r["label"][col] in neg_lab and key in r[how]]
            line[how] = auc(pos, neg)
            line["n_pos"], line["n_neg"] = len(pos), len(neg)
        d = (line["new"] - line["old"]) if line["new"] == line["new"] else float("nan")
        line["delta"] = d
        results.append(line)
        f = lambda v: "—" if v != v else f"{v:.3f}"
        print(f"{label:<34} {line['n_pos']:>4} {line['n_neg']:>4} "
              f"{f(line['old']):>9} {f(line['new']):>9} "
              f"{('' if d != d else f'{d:+.3f}'):>8}")

    print("\nAUC 0.5 — измерение категории врача не различает; "
          "0.7 и выше — различает уверенно.")
    if out_json:
        Path(out_json).write_text(json.dumps(
            {"patients": done, "contrasts": results}, ensure_ascii=False,
            indent=1), encoding="utf-8")
        print(f"сохранено: {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 — все пациенты")
    ap.add_argument("--ckpt", default="research/output/ios_runs/dgcnn_v3/best.pt")
    ap.add_argument("--json", default=None)
    ap.add_argument("--pipeline", choices=["raw", "case"], default="raw",
                    help="raw — точки прямо из сегментации (как ios_occlusion), "
                         "case — очищенный меш (как ios_findings по кейсу)")
    args = ap.parse_args()
    run(args.limit, args.ckpt, args.json, args.pipeline)


if __name__ == "__main__":
    main()
