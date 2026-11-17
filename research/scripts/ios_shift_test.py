"""Устойчивость модели IOS к сдвигу облака точек.

Это слепое пятно, которое едва не стоило нам недели: у v2 mIoU на тесте был
0.92, а на чужих сканах модель теряла центральный резец в 14 случаях из 25.
Причина — сеть запомнила АБСОЛЮТНОЕ положение зуба, и сдвиг на 10 мм ронял
её с 0.972 до 0.721. Обычные метрики этого не показывали вообще.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_shift_test.py \
        --ckpt research/output/ios_runs/dgcnn_v3/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ios_seg_train import DGCNNSeg, N_CLASSES, NPZ  # noqa: E402
from ios_seg_eval import predict_scan  # noqa: E402
from teeth3ds_preprocess import SCALE_MM, robust_center  # noqa: E402

SHIFTS_MM = (0.0, 5.0, 10.0, 15.0, 20.0)


def teeth_ok(pred, gt):
    hit = tot = 0
    for c in range(1, N_CLASSES):
        m = gt == c
        if m.sum() < 50:
            continue
        tot += 1
        v, n = np.unique(pred[m], return_counts=True)
        if v[n.argmax()] == c:
            hit += 1
    return hit, tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scans", type=int, default=40)
    args = ap.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = DGCNNSeg(6, N_CLASSES, ck["args"]["k"],
                     ck["args"].get("width", 1)).to(dev).eval()
    model.load_state_dict(ck["model"])
    chunk = int(ck["args"].get("points", 8192))
    print(f"чекпойнт: {args.ckpt}")

    files = sorted((NPZ / "test").glob("*.npz"))[:args.scans]
    print(f"сканов: {len(files)}\n")
    print(f"{'сдвиг':>8}{'точность':>11}{'зубов верно':>16}")
    base = None
    for mm in SHIFTS_MM:
        acc, hit, tot = [], 0, 0
        for f in files:
            d = np.load(f)
            pts = d["points"] - robust_center(d["points"])
            pts = pts + np.float32(mm / SCALE_MM)
            pr = predict_scan(model, pts, d["normals"], dev, chunk=chunk)
            gt = d["labels"].astype(np.int64)
            acc.append((pr == gt).mean())
            h, t = teeth_ok(pr, gt)
            hit += h
            tot += t
        share = hit / max(tot, 1)
        if base is None:
            base = share
        print(f"{mm:>6.0f}мм{np.mean(acc):>11.3f}"
              f"{f'{hit}/{tot} ({share*100:.1f}%)':>16}"
              + ("" if mm == 0 else f"   потеря {(base - share) * 100:+.1f} п.п."))


if __name__ == "__main__":
    main()
