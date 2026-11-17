"""Сегментация и нумерация зубов на интраоральном скане (Teeth3DS+).

Архитектура — DGCNN-подобная: граф k ближайших соседей строится на лету
(cdist + topk), поэтому не нужны ни torch_geometric, ни torch_cluster, ни
sparse conv — под torch 2.11/cu128 готовых сборок для них может не быть,
а ломать рабочее окружение ради этого не стоит.

Классов 17: 0 = десна, 1-8 правая сторона (резец -> моляр), 9-16 левая.
Челюсть известна из имени файла, обратно в FDI разворачивается однозначно
(см. teeth3ds_preprocess.pos_to_fdi).

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_seg_train.py --epochs 100
    .venv/Scripts/python.exe research/scripts/ios_seg_train.py --smoke
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teeth3ds_preprocess import robust_center  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "research_ios" / "datasets" / "teeth3ds_npz"
OUT = ROOT / "research" / "output" / "ios_runs"

N_CLASSES = 17
N_POINTS = 8192           # из 24000 берём случайные — это и есть аугментация
K = 20                    # соседей в графе
SHIFT_MM = 10.0           # аугментация сдвигом; см. robust_center
SCALE_MM = 50.0           # тот же масштаб, что в препроцессинге


# ============================ данные ============================

class Teeth3DSPoints(Dataset):
    """Облака точек из .npz. Аугментации только на train."""

    def __init__(self, split: str, train: bool, n_points: int = N_POINTS):
        self.files = sorted((NPZ / split).glob("*.npz"))
        if not self.files:
            raise SystemExit(f"нет файлов в {NPZ / split}")
        self.train = train
        self.n_points = n_points

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        d = np.load(self.files[i])
        pts = d["points"].astype(np.float32)
        nrm = d["normals"].astype(np.float32)
        lab = d["labels"].astype(np.int64)

        n = len(pts)
        idx = np.random.choice(n, self.n_points, replace=n < self.n_points)
        pts, nrm, lab = pts[idx], nrm[idx], lab[idx]
        pts = pts - robust_center(pts)      # так же, как на инференсе

        if self.train:
            # Повороты умеренные (+-20°), а не произвольные: скан всегда
            # приходит примерно в одной ориентации, а полный поворот отнял бы
            # у сети возможность отличать верх от низа и левое от правого —
            # ровно на этом мы обожглись с парными структурами на КЛКТ.
            ang = np.random.uniform(-np.pi / 9, np.pi / 9, 3).astype(np.float32)
            for axis, a in enumerate(ang):
                c, s = np.cos(a), np.sin(a)
                r = np.eye(3, dtype=np.float32)
                u, v = [x for x in range(3) if x != axis]
                r[u, u], r[u, v], r[v, u], r[v, v] = c, -s, s, c
                pts, nrm = pts @ r.T, nrm @ r.T
            pts = pts * np.random.uniform(0.9, 1.1)
            # Сдвиг: без него сеть запоминает АБСОЛЮТНОЕ положение зуба и на
            # чужих сканах, где центр стоит иначе, нумерация съезжает —
            # замер показал обвал 0.97 -> 0.72 при сдвиге на 10 мм.
            pts = pts + np.random.uniform(-SHIFT_MM, SHIFT_MM, 3).astype(
                np.float32) / SCALE_MM
            pts = pts + np.random.normal(0, 0.002, pts.shape).astype(np.float32)

        feat = np.concatenate([pts, nrm], axis=1)          # (N, 6)
        return torch.from_numpy(feat), torch.from_numpy(lab)


# ============================ модель ============================

def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """x: (B, N, 3) -> индексы соседей (B, N, k). Считаем на лету."""
    with torch.no_grad():
        d = torch.cdist(x, x)                       # (B, N, N)
        return d.topk(k + 1, largest=False).indices[..., 1:]   # без самой точки


def edge_features(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """feat: (B, C, N), idx: (B, N, k) -> (B, 2C, N, k): [f_i, f_j - f_i]."""
    B, C, N = feat.shape
    k = idx.shape[-1]
    flat = idx.reshape(B, -1)                                # (B, N*k)
    nb = torch.gather(feat, 2, flat.unsqueeze(1).expand(B, C, N * k))
    nb = nb.view(B, C, N, k)
    ctr = feat.unsqueeze(-1).expand(B, C, N, k)
    return torch.cat([ctr, nb - ctr], dim=1)


class EdgeConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(c_in * 2, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_out, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_features(x, idx)).max(dim=-1).values


class DGCNNSeg(nn.Module):
    """Три блока EdgeConv + глобальный вектор -> по-точечная классификация."""

    def __init__(self, c_in: int = 6, n_classes: int = N_CLASSES, k: int = K,
                 w: int = 1):
        super().__init__()
        self.k = k
        c1, c2, c3 = 64 * w, 64 * w, 128 * w
        cat_c = c1 + c2 + c3
        self.e1 = EdgeConv(c_in, c1)
        self.e2 = EdgeConv(c1, c2)
        self.e3 = EdgeConv(c2, c3)
        self.fuse = nn.Sequential(
            nn.Conv1d(cat_c, 1024, 1, bias=False),
            nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True))
        self.head = nn.Sequential(
            nn.Conv1d(1024 + cat_c, 512, 1, bias=False),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, n_classes, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, N, 6). Граф строим по координатам, признаки — xyz + нормали.
        xyz = feat[..., :3]
        idx = knn_graph(xyz, self.k)
        x = feat.transpose(1, 2).contiguous()        # (B, 6, N)
        x1 = self.e1(x, idx)
        x2 = self.e2(x1, idx)
        x3 = self.e3(x2, idx)
        cat = torch.cat([x1, x2, x3], dim=1)         # (B, 256, N)
        g = self.fuse(cat).max(dim=-1, keepdim=True).values
        g = g.expand(-1, -1, cat.shape[-1])
        return self.head(torch.cat([g, cat], dim=1))  # (B, n_classes, N)


# ============================ обучение ============================

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Мягкий Dice по классам: без него 69% десны перетягивают кросс-энтропию."""
    prob = logits.softmax(dim=1)
    oh = F.one_hot(target, logits.shape[1]).permute(0, 2, 1).float()
    inter = (prob * oh).sum(dim=(0, 2))
    denom = prob.sum(dim=(0, 2)) + oh.sum(dim=(0, 2))
    present = oh.sum(dim=(0, 2)) > 0                 # классы, которых нет в батче
    d = (2 * inter + eps) / (denom + eps)
    return 1.0 - d[present].mean()


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 0):
    model.eval()
    inter = torch.zeros(N_CLASSES, device=device)
    union = torch.zeros(N_CLASSES, device=device)
    seen = torch.zeros(N_CLASSES, device=device)
    correct = total = 0
    for b, (feat, lab) in enumerate(loader):
        feat, lab = feat.to(device), lab.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(feat).argmax(dim=1)
        correct += (pred == lab).sum().item()
        total += lab.numel()
        for c in range(N_CLASSES):
            p, t = pred == c, lab == c
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()
            seen[c] += t.sum()
        if max_batches and b + 1 >= max_batches:
            break
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.nan)
    teeth = iou[1:][~torch.isnan(iou[1:])]
    return {"acc": correct / max(total, 1),
            "miou_teeth": float(teeth.mean()) if len(teeth) else float("nan"),
            "iou_gum": float(iou[0]),
            "iou": iou.cpu().numpy()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--points", type=int, default=N_POINTS)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--width", type=int, default=1,
                    help="множитель ширины сети")
    ap.add_argument("--name", default="dgcnn_v1")
    ap.add_argument("--smoke", action="store_true",
                    help="короткий прогон: проверить, что всё сходится")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    tr = Teeth3DSPoints("train", True, args.points)
    te = Teeth3DSPoints("test", False, args.points)
    if args.smoke:
        tr.files, te.files = tr.files[:16], te.files[:8]
        args.epochs = 2
        # иначе смоук-тест пишется в тот же каталог и затирает обученный
        # чекпойнт — уже случилось однажды с dgcnn_v1
        args.name = f"{args.name}_smoke"
    print(f"train={len(tr)} сканов, test={len(te)}")

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True,
                       persistent_workers=args.workers > 0)
    dl_te = DataLoader(te, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers,
                       persistent_workers=args.workers > 0)

    model = DGCNNSeg(6, N_CLASSES, args.k, args.width).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"параметров: {n_par/1e6:.2f} млн")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    run = OUT / args.name
    run.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        t0, tot, nb = time.perf_counter(), 0.0, 0
        for feat, lab in dl_tr:
            feat, lab = feat.to(device, non_blocking=True), lab.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(feat)
                loss = F.cross_entropy(logits, lab) + dice_loss(logits.float(), lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        m = evaluate(model, dl_te, device, max_batches=25 if not args.smoke else 0)
        dt = time.perf_counter() - t0
        print(f"эпоха {ep:>3}  loss={tot/max(nb,1):.4f}  acc={m['acc']:.4f}  "
              f"mIoU(зубы)={m['miou_teeth']:.4f}  IoU(десна)={m['iou_gum']:.4f}  "
              f"{dt:.0f} с", flush=True)
        if m["miou_teeth"] == m["miou_teeth"] and m["miou_teeth"] > best:
            best = m["miou_teeth"]
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": ep, "miou_teeth": best}, run / "best.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, run / "final.pt")
    print(f"лучший mIoU по зубам: {best:.4f}   -> {run}")


if __name__ == "__main__":
    main()
