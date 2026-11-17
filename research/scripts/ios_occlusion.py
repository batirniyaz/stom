"""Прикус по интраоральному скану: измерения, а не отдельная модель.

Как только зубы найдены и пронумерованы, класс по Энглю, глубина перекрытия,
сагиттальная щель и средняя линия — это геометрия. Тот же подход, что в
находках по КЛКТ: считаем измеримое, а врачебную разметку используем, чтобы
проверить пороги, а не чтобы обучать ещё одну сеть.

Обязательное условие: обе челюсти в ОДНОЙ системе координат со смыканием.
Teeth3DS для этого не годится — там каждый скан отцентрован отдельно, прикус
уничтожен. Годятся Bits2Bites и сканы клиники (Exocad хранит анатомически).

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_occlusion.py \
        --upper .../UpperJaw.obj --lower .../LowerJaw.obj
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ios_seg_train import DGCNNSeg, N_CLASSES  # noqa: E402
from ios_seg_eval import predict_scan  # noqa: E402
from ios_seg_infer import load_mesh  # noqa: E402
from ios_canonical import canonical_transform  # noqa: E402
from ios_geometry import (arch_frame_from_centroids,  # noqa: E402
                          occlusion_measures)
from teeth3ds_preprocess import SCALE_MM, pos_to_fdi, robust_center  # noqa: E402

# Пороги и сами измерения живут в ios_geometry: их использует и этот CLI, и
# слой находок по кейсу. Разводить по двум местам нельзя — ровно так уже
# разъехалась вертикаль, и это стоило нам провала измерений прикуса.

MIN_PTS = 60              # меньше точек на зубе — не считаем


def segment_jaw(path: Path, jaw: str, model, device, chunk):
    """-> (вершины в ИСХОДНЫХ координатах, метка FDI на вершину)"""
    verts, normals = load_mesh(path)
    pts = ((verts - robust_center(verts)) / SCALE_MM).astype(np.float32)
    nrm = normals.astype(np.float32)
    R = canonical_transform(model, pts, nrm, device, chunk, jaw=jaw)
    lab = predict_scan(model, (pts @ R.T).astype(np.float32),
                       (nrm @ R.T).astype(np.float32), device, chunk=chunk)
    fdi = np.array([pos_to_fdi(int(p), jaw) if p > 0 else 0
                    for p in range(N_CLASSES)], dtype=np.int32)[lab]
    return verts, fdi


def tooth_points(verts, fdi, tooth):
    m = fdi == tooth
    return verts[m] if m.sum() >= MIN_PTS else None


def build_frame(vu, fu, vl, fl):
    """Оси дуги по сегментированным челюстям -> (ArchFrame, центры верх, низ).

    Вертикаль раньше считалась как направление от нижней челюсти к верхней. В
    смыкании центры верхних и нижних зубов расходятся всего на 4 мм — плечо
    короткое, ось уезжает. Замер на 180 пациентах Bits2Bites: расхождение с
    перпендикуляром к плоскости дуги медиана 31.6°, и вот что это давало
    (AUC — разделение врачебных категорий):

        глубокий прикус против нормы   0.573 -> 0.971
        открытый прикус против нормы   0.871 -> 0.959
        II класс справа против I       0.578 -> 0.738
        III класс справа против I      0.564 -> 0.750
        II класс слева против I        0.648 -> 0.791
        III класс слева против I       0.582 -> 0.809
        средняя линия                  0.544 -> 0.692

    То есть глубокий прикус не ловился не потому, что измерение неверное, а
    потому, что ось, вдоль которой его мерили, стояла косо.
    """
    cu = {t: p.mean(0) for t in range(11, 29)
          if (p := tooth_points(vu, fu, t)) is not None}
    cl = {t: p.mean(0) for t in range(31, 49)
          if (p := tooth_points(vl, fl, t)) is not None}
    if len(cu) < 4 or len(cl) < 4:
        raise SystemExit("слишком мало зубов для построения системы координат")
    return arch_frame_from_centroids({**cu, **cl}), cu, cl


def measure(verts_u, fdi_u, verts_l, fdi_l):
    """Прикус по сегментированным челюстям. Считает ios_geometry."""
    frame, _, _ = build_frame(verts_u, fdi_u, verts_l, fdi_l)
    pts = {}
    for verts, fdi, lo, hi in ((verts_u, fdi_u, 11, 29), (verts_l, fdi_l, 31, 49)):
        for t in range(lo, hi):
            p = tooth_points(verts, fdi, t)
            if p is not None:
                pts[t] = p
    res = occlusion_measures(frame, pts)
    res["draft"] = True
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", required=True)
    ap.add_argument("--lower", required=True)
    ap.add_argument("--ckpt", default="research/output/ios_runs/dgcnn_v3/best.pt")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DGCNNSeg(6, N_CLASSES, ck["args"]["k"],
                     ck["args"].get("width", 1)).to(device).eval()
    model.load_state_dict(ck["model"])
    chunk = int(ck["args"].get("points", 8192))

    vu, fu = segment_jaw(Path(args.upper), "upper", model, device, chunk)
    vl, fl = segment_jaw(Path(args.lower), "lower", model, device, chunk)
    res = measure(vu, fu, vl, fl)

    print(f"\n{'=' * 62}\nПРИКУС — черновик измерений\n{'=' * 62}")
    for k, v in res["measurements"].items():
        print(f"   {k:<26} {v}")
    print("\nнаходки:")
    if not res["findings"]:
        print("   отклонений не выявлено")
    for name, val, who in res["findings"]:
        print(f"   • {name}: {val}   -> {who}")
    print(f"\n{'-' * 62}\nЧЕРНОВИК. Пороги не подтверждены ортодонтом.")

    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
