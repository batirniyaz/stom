"""Калибровка отсева посторонних кусков в меше зуба — по эталону Teeth3DS.

Сегментация иногда прицепляет к зубу лоскут поверхности в стороне. Отсеивать
их по РАЗМЕРУ нельзя: в CLAUDE.md записано, что жёсткий фильтр съедал до 70%
поверхности — зуб на скане нередко распадается на два больших куска. Значит
судить надо по РАССТОЯНИЮ: свой кусок лежит вплотную, чужой в стороне.

Здесь порог подбирается по данным: для каждой вторичной компоненты известно из
эталонной разметки, её ли это зуб. Считаем, что теряем и что выигрываем при
каждом пороге.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_seg_components.py --limit 40
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
from ios_canonical import canonical_transform  # noqa: E402
from teeth3ds_preprocess import SCALE_MM, pos_to_fdi, robust_center  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "research_ios" / "datasets" / "teeth3ds"
GAPS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--ckpt", default="research/output/ios_runs/dgcnn_v3/best.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DGCNNSeg(6, N_CLASSES, ck["args"]["k"],
                     ck["args"].get("width", 1)).to(device).eval()
    model.load_state_dict(ck["model"])
    chunk = int(ck["args"].get("points", 8192))

    scans = []
    for jaw in ("lower", "upper"):
        for d in sorted((ROOT / "scans" / jaw).iterdir()):
            obj = d / f"{d.name}_{jaw}.obj"
            if obj.is_file() and obj.with_suffix(".json").is_file():
                scans.append((obj, jaw))
    scans = scans[:args.limit // 2] + scans[-args.limit // 2:]
    print(f"сканов: {len(scans)}")

    rows = []          # (зазор до основного тела, доля «свой зуб», граней)
    np.random.seed(0)
    for n, (obj, jaw) in enumerate(scans, 1):
        m = trimesh.load(str(obj), process=False, force="mesh")
        V = np.asarray(m.vertices, float)
        gt = np.asarray(json.loads(obj.with_suffix(".json").read_text(
            encoding="utf-8"))["labels"], int)
        if len(gt) != len(V):
            continue
        pts = ((V - robust_center(V)) / SCALE_MM).astype(np.float32)
        nrm = np.asarray(m.vertex_normals, np.float32)
        R = canonical_transform(model, pts, nrm, device, chunk, jaw=jaw)
        lab = predict_scan(model, (pts @ R.T).astype(np.float32),
                           (nrm @ R.T).astype(np.float32), device, chunk=chunk)
        fdi = np.array([pos_to_fdi(int(p), jaw) if p > 0 else 0
                        for p in range(N_CLASSES)], dtype=np.int32)[lab]

        F = np.asarray(m.faces)
        face_lab = fdi[F]
        keep = (face_lab[:, 0] == face_lab[:, 1]) & (face_lab[:, 1] == face_lab[:, 2])
        for t in sorted(set(fdi.tolist())):
            if t == 0:
                continue
            sel = np.flatnonzero(keep & (face_lab[:, 0] == t))
            if len(sel) < 60:
                continue
            sub = m.submesh([sel], append=True, repair=False)
            if sub is None or len(sub.faces) < 60:
                continue
            comp = trimesh.graph.connected_component_labels(
                sub.face_adjacency, node_count=len(sub.faces))
            sizes = np.bincount(comp)
            if len(sizes) < 2:
                continue
            main = int(np.argmax(sizes))
            SV, SF = np.asarray(sub.vertices), np.asarray(sub.faces)
            mainpts = SV[np.unique(SF[comp == main])]
            tree = cKDTree(mainpts)
            for l in range(len(sizes)):
                if l == main or sizes[l] < 10:
                    continue
                vid = np.unique(SF[comp == l])
                p = SV[vid]
                gap = float(tree.query(p)[0].min())
                # эталон: какая доля этих вершин действительно этот зуб
                d0, i0 = cKDTree(V).query(p)
                own = float((gt[i0] == t).mean())
                rows.append((gap, own, int(sizes[l])))
        if n % 10 == 0:
            print(f"  {n}/{len(scans)}", flush=True)

    if not rows:
        print("вторичных компонент не нашлось")
        return
    gap = np.array([r[0] for r in rows])
    own = np.array([r[1] for r in rows])
    area = np.array([r[2] for r in rows])
    good = own >= 0.5                      # компонента и правда этого зуба
    print(f"\nвторичных компонент: {len(rows)}  "
          f"(свои {good.sum()}, чужие {(~good).sum()})")
    print(f"зазор до основного тела: свои медиана {np.median(gap[good]):.2f} мм, "
          f"чужие {np.median(gap[~good]):.2f} мм")
    print(f"\n{'порог':>7} {'отсеяно чужих':>15} {'потеряно своих':>16} "
          f"{'потеряно граней':>17}")
    for g in GAPS:
        drop = gap > g
        print(f"{g:7.1f} {100*(drop & ~good).sum()/max((~good).sum(),1):14.0f}% "
              f"{100*(drop & good).sum()/max(good.sum(),1):15.0f}% "
              f"{100*area[drop & good].sum()/max(area[good].sum(),1):16.0f}%")


if __name__ == "__main__":
    main()
