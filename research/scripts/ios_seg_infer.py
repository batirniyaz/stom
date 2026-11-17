"""Сегментация зубов на произвольном интраоральном скане (OBJ/STL/PLY).

Проверка на клиническом скане, у которого нет разметки. Раз сверить не с чем,
проверяем правдоподобие геометрией: зубы вдоль дуги обязаны идти по порядку.
Если модель путает стороны или номера, порядок ломается — это ловит ошибку
без всякой разметки.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_seg_infer.py path/to/UpperJaw.obj --jaw upper
    ... --png out.png --ply out.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ios_seg_train import DGCNNSeg, N_CLASSES  # noqa: E402
from ios_seg_eval import predict_scan  # noqa: E402
from ios_canonical import canonical_transform  # noqa: E402
from teeth3ds_preprocess import (N_POINTS, SCALE_MM, pos_to_fdi,
                                 robust_center)  # noqa: E402

# цвета по квадрантам, как в КЛКТ-вьюере
QUAD_COLOR = {1: (147, 197, 253), 2: (110, 231, 183),
              3: (252, 211, 77), 4: (249, 168, 212)}
GUM = (200, 160, 160)


def load_mesh(path: Path):
    import trimesh
    m = trimesh.load(str(path), process=False, force="mesh")
    return (np.asarray(m.vertices, np.float64),
            np.asarray(m.vertex_normals, np.float64))


def arch_order_check(pts_mm, labels, jaw):
    """Зубы вдоль дуги должны идти по порядку. Возвращает (последовательность,
    сколько пар нарушают порядок)."""
    cents = {}
    for c in range(1, N_CLASSES):
        m = labels == c
        if m.sum() < 30:
            continue
        cents[c] = pts_mm[m].mean(axis=0)
    if len(cents) < 3:
        return [], 0, 0
    arr = np.array(list(cents.values()))
    center = arr[:, :2].mean(axis=0)
    # угол вокруг центра дуги в плоскости окклюзии (x, y)
    ang = {c: np.arctan2(v[1] - center[1], v[0] - center[0])
           for c, v in cents.items()}
    order = sorted(ang, key=lambda c: ang[c])
    # линейный индекс вдоль дуги: 8..1 (правая сторона), затем 9..16 (левая)
    lin = [(9 - c) if c <= 8 else c for c in order]
    # Дуга замкнута, а сортировка по углу начинается в произвольном месте:
    # [15,14,13,12,11,21..28,17,16] — это правильный обход, просто разрезанный
    # не с края. Поэтому перебираем все сдвиги и оба направления.
    n = len(lin)
    best = n
    for s in range(n):
        rot = lin[s:] + lin[:s]
        best = min(best,
                   sum(1 for a, b in zip(rot, rot[1:]) if b < a),
                   sum(1 for a, b in zip(rot, rot[1:]) if b > a))
    return order, best, n - 1


def render(pts, colors, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0a0c12")
    for ax, (e, a) in zip(axes, [(90, -90), (25, -90)]):
        el, az = np.radians(e), np.radians(a)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                        np.sin(el)])
        up0 = np.array([0.0, 0.0, 1.0])
        r = np.cross(fwd, up0)
        r = r / (np.linalg.norm(r) + 1e-9)
        u = np.cross(r, fwd)
        p = pts - pts.mean(axis=0)
        x, y, z = p @ r, p @ u, p @ fwd
        o = np.argsort(-z)
        ax.scatter(x[o], y[o], c=colors[o] / 255.0, s=5.0, marker=".",
                   linewidths=0)
        ax.set_facecolor("#0a0c12")
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, facecolor="#0a0c12")
    print(f"картинка: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--jaw", choices=["upper", "lower"], default=None)
    ap.add_argument("--ckpt", default="research/output/ios_runs/dgcnn_v3/best.pt")
    ap.add_argument("--png", default=None)
    ap.add_argument("--ply", default=None)
    args = ap.parse_args()

    path = Path(args.mesh)
    jaw = args.jaw or ("lower" if "lower" in path.stem.lower() else "upper")

    verts, normals = load_mesh(path)
    print(f"{path.name}: вершин={len(verts):,}  челюсть={jaw}")
    print(f"   габарит, мм: {np.round(verts.max(0) - verts.min(0), 1)}")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(verts), min(N_POINTS, len(verts)),
                     replace=len(verts) < N_POINTS)
    center = robust_center(verts)
    pts = ((verts[idx] - center) / SCALE_MM).astype(np.float32)
    nrm = normals[idx].astype(np.float32)

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DGCNNSeg(6, N_CLASSES, ck["args"]["k"], ck["args"].get("width", 1)).to(device).eval()
    model.load_state_dict(ck["model"])
    R = canonical_transform(model, pts, nrm, device,
                            int(ck['args'].get('points', 8192)), jaw=jaw)
    if not np.allclose(R, np.eye(3)):
        print(f"   ориентация: скан развёрнут")
    labels = predict_scan(model, (pts @ R.T).astype(np.float32),
                          (nrm @ R.T).astype(np.float32), device)

    pts_mm = verts[idx]
    found = []
    for c in range(1, N_CLASSES):
        n = int((labels == c).sum())
        if n >= 30:
            found.append((c, pos_to_fdi(c, jaw), n))
    gum = int((labels == 0).sum())
    print(f"\nдесна: {gum:,} точек ({gum/len(labels)*100:.0f}%)")
    print(f"зубов найдено: {len(found)}")
    print(f"   FDI: {[f for _, f, _ in found]}")

    order, inversions, pairs = arch_order_check(pts_mm, labels, jaw)
    print(f"\nпорядок вдоль дуги: {[pos_to_fdi(c, jaw) for c in order]}")
    print(f"нарушений порядка: {inversions} из {pairs} пар"
          + ("   — порядок верный" if inversions == 0 else "   — ЕСТЬ СБОЙ"))

    colors = np.tile(np.array(GUM, np.float32), (len(labels), 1))
    for c, fdi, _ in found:
        colors[labels == c] = QUAD_COLOR.get(fdi // 10, (255, 255, 255))
    if args.png:
        render(pts_mm, colors, args.png)
    if args.ply:
        with open(args.ply, "w") as f:
            f.write("ply\nformat ascii 1.0\n"
                    f"element vertex {len(pts_mm)}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "end_header\n")
            for (x, y, z), (r, g, b) in zip(pts_mm, colors.astype(int)):
                f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")
        print(f"облако: {args.ply}")


if __name__ == "__main__":
    main()
