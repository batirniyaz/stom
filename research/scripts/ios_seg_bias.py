"""Куда ошибается сегментация зубов: захватывает лишнее или не добирает своё.

Обычный mIoU одним числом этого не показывает. А разница принципиальная:
если сеть систематически ЗАХВАТЫВАЕТ ЛИШНЕЕ, меш зуба раздувается и все
точечные измерения (ориентиры, мезио-дистальный размер) уезжают; если НЕ
ДОБИРАЕТ — зуб выходит обрубком, как 41 на клиническом скане.

На каждый зуб считаем:
    полнота  — какая доля истинной поверхности зуба нам досталась;
    точность — какая доля размеченного нами действительно этот зуб;
    куда ушло лишнее — на соседний зуб или на десну.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_seg_bias.py --limit 100
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ios_seg_train import DGCNNSeg, N_CLASSES, NPZ  # noqa: E402
from ios_seg_eval import predict_scan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ckpt", default="research/output/ios_runs/dgcnn_v3/best.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DGCNNSeg(6, N_CLASSES, ck["args"]["k"],
                     ck["args"].get("width", 1)).to(device).eval()
    model.load_state_dict(ck["model"])
    chunk = int(ck["args"].get("points", 8192))

    files = sorted((NPZ / "test").glob("*.npz"))[:args.limit]
    print(f"сканов: {len(files)}, чекпойнт {Path(args.ckpt).parent.name}")

    tp = defaultdict(int)      # верно отнесено к позиции
    fn = defaultdict(int)      # истинное, но мы отдали другому
    fp_gum = defaultdict(int)  # мы назвали зубом то, что десна
    fp_nb = defaultdict(int)   # мы назвали этим зубом соседний зуб
    for n, f in enumerate(files, 1):
        d = np.load(f)
        pts, nrm = d["points"].astype(np.float32), d["normals"].astype(np.float32)
        gt = d["labels"].astype(int)
        pred = predict_scan(model, pts, nrm, device, chunk=chunk)
        for c in range(1, N_CLASSES):
            g, p = gt == c, pred == c
            tp[c] += int((g & p).sum())
            fn[c] += int((g & ~p).sum())
            fp_gum[c] += int((~g & p & (gt == 0)).sum())
            fp_nb[c] += int((~g & p & (gt != 0)).sum())
        if n % 25 == 0:
            print(f"  {n}/{len(files)}", flush=True)

    print(f"\n{'поз.':>5} {'полнота':>9} {'точность':>9} {'лишнее с десны':>15} "
          f"{'лишнее с соседа':>16}")
    tot = defaultdict(int)
    for c in range(1, N_CLASSES):
        t, f_, gu, nb = tp[c], fn[c], fp_gum[c], fp_nb[c]
        if t + f_ < 500:
            continue
        rec = t / max(t + f_, 1)
        prec = t / max(t + gu + nb, 1)
        print(f"{c:>5} {rec:9.3f} {prec:9.3f} {gu/max(t+gu+nb,1):15.3f} "
              f"{nb/max(t+gu+nb,1):16.3f}")
        for k, v in (("tp", t), ("fn", f_), ("gum", gu), ("nb", nb)):
            tot[k] += v

    rec = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    prec = tot["tp"] / max(tot["tp"] + tot["gum"] + tot["nb"], 1)
    print(f"\nв целом по зубам: полнота {rec:.3f}, точность {prec:.3f}")
    print(f"перекос: {'ЗАХВАТЫВАЕМ ЛИШНЕЕ' if prec < rec else 'НЕ ДОБИРАЕМ СВОЁ'} "
          f"(разница {abs(prec - rec):.3f})")
    print(f"из лишнего: с десны {tot['gum']/max(tot['gum']+tot['nb'],1):.0%}, "
          f"с соседних зубов {tot['nb']/max(tot['gum']+tot['nb'],1):.0%}")


if __name__ == "__main__":
    main()
