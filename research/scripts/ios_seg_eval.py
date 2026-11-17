"""Честная оценка модели сегментации IOS на полном тесте.

Во время обучения mIoU считался по 25 батчам со случайной подвыборкой точек
и скакал на 0.08 между соседними эпохами — доверять такому числу нельзя.
Здесь: все 600 сканов, все 24000 точек каждого, IoU накапливается по всему
набору, а не усредняется по батчам.

Отдельно меряем зеркальную путаницу (зуб 16 против 26): именно на ней
посыпалась КЛКТ-модель, и здесь это главный риск.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_seg_eval.py
    .venv/Scripts/python.exe research/scripts/ios_seg_eval.py --limit 50
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ios_seg_train import DGCNNSeg, N_CLASSES, NPZ  # noqa: E402
from teeth3ds_preprocess import pos_to_fdi, robust_center  # noqa: E402

CHUNK = 8192      # столько же точек, сколько сеть видела при обучении


def predict_scan(model, pts, nrm, device, chunk=CHUNK, rounds=1):
    """Предсказание по всем точкам: случайная перестановка -> непересекающиеся
    куски того же размера, что при обучении. Граф соседей строится внутри куска,
    как и на обучении, поэтому условия совпадают.

    rounds > 1 — усреднение по нескольким разбивкам. Каждая вершина иначе
    получает ровно одно предсказание, зависящее от того, какие соседи случайно
    попали с ней в один кусок; голосование по нескольким прогонам снимает эту
    лотерею. Стоит ровно во столько же раз дороже.
    """
    n = len(pts)
    votes = np.zeros((n, N_CLASSES), dtype=np.int16)
    for _ in range(max(rounds, 1)):
        order = np.random.permutation(n)
        for s in range(0, n, chunk):
            idx = order[s:s + chunk]
            if len(idx) < 64:
                idx = order[max(0, n - chunk):]
            feat = np.concatenate([pts[idx], nrm[idx]], axis=1)[None]
            t = torch.from_numpy(feat).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(t).argmax(dim=1)[0].cpu().numpy()
            votes[idx, pred[:len(idx)]] += 1
    return votes.argmax(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="research/output/ios_runs/dgcnn_v2/best.pt")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DGCNNSeg(6, N_CLASSES, ck["args"]["k"], ck["args"].get("width", 1)).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"чекпойнт: {args.ckpt}  (эпоха {ck.get('epoch')}, "
          f"mIoU на обучении {ck.get('miou_teeth'):.4f})")

    files = sorted((NPZ / "test").glob("*.npz"))
    if args.limit:
        files = files[:args.limit]
    chunk = int(ck["args"].get("points", CHUNK))   # как при обучении
    print(f"сканов в тесте: {len(files)}, кусок {chunk} точек")

    inter = np.zeros(N_CLASSES, np.int64)
    union = np.zeros(N_CLASSES, np.int64)
    gt_cnt = np.zeros(N_CLASSES, np.int64)
    mirror = right = wrong_nb = 0          # зеркальная путаница и соседи
    per_scan = []
    tooth_hit = tooth_tot = 0

    for i, f in enumerate(files, 1):
        d = np.load(f)
        pts = d["points"] - robust_center(d["points"])
        pred = predict_scan(model, pts, d["normals"], device, chunk)
        gt = d["labels"].astype(np.int64)

        ious = []
        for c in range(N_CLASSES):
            p, t = pred == c, gt == c
            it, un = int((p & t).sum()), int((p | t).sum())
            inter[c] += it
            union[c] += un
            gt_cnt[c] += int(t.sum())
            if c > 0 and t.any():
                ious.append(it / max(un, 1))
        if ious:
            per_scan.append(float(np.mean(ious)))

        # зеркальная ошибка: позиция p и p+-8 — один и тот же зуб на другой стороне
        m = gt > 0
        g, p = gt[m], pred[m]
        right += int((g == p).sum())
        gm = np.where(g <= 8, g + 8, g - 8)
        mirror += int((p == gm).sum())
        adj = (np.abs(p - g) == 1) & ((p <= 8) == (g <= 8)) & (p > 0)
        wrong_nb += int(adj.sum())

        # «зуб найден»: большинство точек зуба получили правильную метку
        for c in range(1, N_CLASSES):
            t = gt == c
            if t.sum() < 50:
                continue
            tooth_tot += 1
            vals, cnts = np.unique(pred[t], return_counts=True)
            if vals[cnts.argmax()] == c:
                tooth_hit += 1
        if i % 100 == 0:
            print(f"  {i}/{len(files)}")

    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    teeth = iou[1:][~np.isnan(iou[1:])]
    tot = max(right + mirror + wrong_nb, 1)
    print(f"\n{'='*60}")
    print(f"IoU десны:            {iou[0]:.4f}")
    print(f"mIoU по зубам (global): {np.nanmean(teeth):.4f}")
    print(f"mIoU по сканам:         {np.mean(per_scan):.4f}")
    print(f"зубов определено верно: {tooth_hit}/{tooth_tot} "
          f"({tooth_hit/max(tooth_tot,1)*100:.1f}%)")
    print(f"\nиз точек, лежащих на зубах:")
    print(f"   верная позиция:      {right/tot*100:5.1f}%")
    print(f"   ЗЕРКАЛЬНАЯ ошибка:   {mirror/tot*100:5.1f}%  (зуб отнесён на другую сторону)")
    print(f"   соседний зуб:        {wrong_nb/tot*100:5.1f}%")

    print(f"\n{'позиция':<10}{'FDI(верх/низ)':<16}{'IoU':>8}{'точек в GT':>14}")
    for c in range(1, N_CLASSES):
        fu, fl = pos_to_fdi(c, "upper"), pos_to_fdi(c, "lower")
        v = f"{iou[c]:.4f}" if not np.isnan(iou[c]) else "  —"
        print(f"{c:<10}{f'{fu} / {fl}':<16}{v:>8}{gt_cnt[c]:>14,}")


if __name__ == "__main__":
    main()
