"""Модель канала на вырезе зуба: сегментация пульпы до верхушки.

    .venv\\Scripts\\python.exe research\\scripts\\train_canal_net.py --epochs 120

ЗАЧЕМ. Пульпа основной модели (`sp`) не доходит до верхушки: 72-91% длины
зуба. Корень при этом сегментирован верно — на эталоне DentVoxel расхождение
у верхушки 0.26 мм по 90-му перцентилю. Недорисован именно канал, и это
наследство обучающих данных: у ToothFairy3 канал доведён до 85% длины лишь у
46% зубов. Поэтому вторая модель учится на ВЫРЕЗАХ ЗУБОВ и только там, где
канал размечен до конца (`cbct_canal_crops.py`, порог `MIN_REACH`).

ВХОД — два канала: выпрямленный вырез зуба (плотность) и маска ТЕЛА зуба
(зуб вместе с пульпой, замкнутый). Маска нужна, чтобы сеть не искала канал
за пределами зуба: тёмная периодонтальная щель снаружи похожа на канал, и
без подсказки сеть на неё садится (та же ловушка, что у трассировки по
плотности — см. `cbct_canals`).

ПОЧЕМУ ИМЕННО ТЕЛО, А НЕ ЗУБ. Маска зуба и в эталоне, и у нашей модели
пульпу не включает — на её месте дырка, то есть готовый ответ. С такой
подсказкой сеть на второй эпохе дала Dice 0.996, просто обводя дырку.
Тело зуба строится и на приёме (`cbct_backend.canal_body`), и там дырки
в нужном месте нет.

ВЫХОД — маска пульпы с каналами в той же сетке. Номер зуба не
предсказывается: он известен от первой модели.

НОРМИРОВКА — по САМОМУ ЗУБУ: (плотность минус медиана дентина) / разброс
дентина. Шкала КЛКТ не количественная (у наших аппаратов дентин читается
1400-3200 при настоящих 1200-1800), и абсолютные значения переносить между
аппаратами нельзя — это правило в проекте выучено дважды (эндо-статус,
порог канала в трассировке).

МЕТРИКА ОТБОРА — не Dice. Dice пульпы определяется камерой: она даёт 60-80%
объёма, и модель, полностью пропустившая каналы, всё равно получит 0.8.
Отбираем по ДОХОДУ канала: доля длины зуба, до которой дотянулась пульпа,
против эталонной.

Но и один доход не годится: на первой же эпохе он вышел 0.96 против 0.91 у
эталона при Dice 0.589 — рыхлое предсказание тянется дальше канала и потому
получает ХОРОШУЮ ошибку дохода. Поэтому отбор по сумме
`ошибка дохода + (1 - Dice) / 4`: делитель ставит вклад формы (0.04-0.10 при
Dice 0.6-0.85) в один масштаб с ошибкой дохода (0.03-0.10), и ни одно из двух
не перевешивает. Чистый Dice в отборе так и не участвует — иначе он опять
решался бы камерой.

МЕТРИКИ ОТДЕЛЬНО ПО ИСТОЧНИКАМ. Pulpy3D — другой аппарат и другая популяция,
чем ToothFairy3, и зубы там от нашей модели, а не из эталона. Среднее по двум
источникам спрятало бы провал на одном: такое в проекте уже было (модель
проигрывала на внешней выборке, а сводная метрика этого не показывала).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research_3d/datasets/canal_crops"
OUT = REPO / "research/output/canal_net"


SOURCES = ("tf3", "pulpy3d")


class Crops(Dataset):
    def __init__(self, files: list[Path], train: bool):
        self.files, self.train = files, train
        # Источник зашит в имя файла нарезкой: tf3_<том>_<зуб>.npz.
        self.src = [SOURCES.index(f.name.split("_")[0]) for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = d["img"].astype(np.float32)
        body = d["body"].astype(np.float32)
        pulp = d["pulp"].astype(np.float32)
        m = body > 0.5
        if m.sum() > 100:
            med = float(np.median(img[m]))
            mad = float(np.median(np.abs(img[m] - med))) * 1.4826
        else:
            med, mad = 0.0, 500.0
        img = (img - med) / max(mad, 50.0)
        if self.train:
            # Отражение поперёк оси зуба: канал зеркальной симметрией не
            # различается, а сторона зуба сети знать не нужно.
            for ax in (1, 2):
                if np.random.rand() < 0.5:
                    img, body, pulp = (np.flip(a, ax).copy() for a in (img, body, pulp))
            img = img * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.2, 0.2)
            img = img + np.random.normal(0, 0.05, img.shape).astype(np.float32)
        x = np.stack([img, body])
        return torch.from_numpy(x), torch.from_numpy(pulp)[None], self.src[i]


def conv(i, o):
    return nn.Sequential(
        nn.Conv3d(i, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o), nn.LeakyReLU(0.01, True),
        nn.Conv3d(o, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o), nn.LeakyReLU(0.01, True))


class UNet3D(nn.Module):
    """Небольшой 3D U-Net: вырез маленький (128x64x64), большой сети не нужно."""

    def __init__(self, ch=(24, 48, 96, 192)):
        super().__init__()
        self.enc = nn.ModuleList()
        prev = 2
        for c in ch:
            self.enc.append(conv(prev, c))
            prev = c
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], 2, 2)
                                 for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([conv(ch[i] * 2, ch[i]) for i in range(len(ch) - 1)])
        self.head = nn.Conv3d(ch[0], 1, 1)

    def forward(self, x):
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                feats.append(x)
                x = self.pool(x)
        for i in range(len(self.dec) - 1, -1, -1):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, feats[i]], 1))
        return self.head(x)


def dice_loss(logit, target, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum((1, 2, 3, 4)) + eps
    den = p.sum((1, 2, 3, 4)) + target.sum((1, 2, 3, 4)) + eps
    return 1 - (num / den).mean()


def reach(mask: np.ndarray, tooth: np.ndarray) -> float:
    """Доля длины зуба (по уровням выреза), до которой доходит маска."""
    lt = np.where(tooth.any(axis=(1, 2)))[0]
    lp = np.where(mask.any(axis=(1, 2)))[0]
    if not len(lt) or not len(lp) or lt.max() == lt.min():
        return 0.0
    return float((lp.max() - lt.min()) / (lt.max() - lt.min()))


def _summary(d: np.ndarray, rp: np.ndarray, rg: np.ndarray) -> dict:
    if not len(d):
        return {"n": 0}
    return {"dice": float(np.nanmedian(d)),
            "reach_pred": float(np.median(rp)), "reach_gt": float(np.median(rg)),
            "reach_err": float(np.median(np.abs(rp - rg))),
            "short": float((rg - rp > 0.1).mean()),
            "over": float((rp - rg > 0.1).mean()), "n": int(len(d))}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    dices, r_pred, r_gt, srcs = [], [], [], []
    for x, y, s in loader:
        out = torch.sigmoid(model(x.to(device))).cpu().numpy() > 0.5
        y = y.numpy() > 0.5
        t = x[:, 1].numpy() > 0.5
        for k in range(len(y)):
            tot = out[k, 0].sum() + y[k, 0].sum()
            dices.append(2 * (out[k, 0] & y[k, 0]).sum() / tot if tot else np.nan)
            r_pred.append(reach(out[k, 0], t[k]))
            r_gt.append(reach(y[k, 0], t[k]))
            srcs.append(int(s[k]))
    d = np.array(dices, float)
    rp, rg, sr = np.array(r_pred), np.array(r_gt), np.array(srcs)
    m = _summary(d, rp, rg)
    m["by_source"] = {name: _summary(d[sr == i], rp[sr == i], rg[sr == i])
                      for i, name in enumerate(SOURCES)}
    # Отбор: доход канала плюс форма в том же масштабе — см. докстринг.
    m["score"] = m["reach_err"] + (1 - m["dice"]) / 4
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0, help="сколько вырезов взять (отладка)")
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    tr = sorted((data / "train").glob("*.npz"))
    va = sorted((data / "val").glob("*.npz"))
    if a.limit:
        tr, va = tr[:a.limit], va[:max(a.limit // 5, 2)]
    src = {}
    for f in tr:
        src[f.name.split("_")[0]] = src.get(f.name.split("_")[0], 0) + 1
    print(f"обучение {len(tr)} вырезов ({src}), проверка {len(va)}")
    if not tr or not va:
        raise SystemExit("нет данных: сначала cbct_canal_crops.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    # Канал — доли процента объёма выреза, и одна лишь BCE выучила бы «пусто».
    # Dice-слагаемое смотрит на перекрытие, BCE с весом — на редкий класс.
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))
    dl_tr = DataLoader(Crops(tr, True), batch_size=a.batch, shuffle=True,
                       num_workers=4, drop_last=True, pin_memory=True)
    dl_va = DataLoader(Crops(va, False), batch_size=a.batch, num_workers=4)

    out.mkdir(parents=True, exist_ok=True)
    best, best_ep, patience = 1e9, 0, 25
    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for x, y, _ in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logit = model(x)
            loss = bce(logit, y) + dice_loss(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += float(loss.detach())
        sched.step()
        m = evaluate(model, dl_va, device)
        torch.save({"model": model.state_dict(), "metrics": m, "epoch": ep}, out / "last.pt")
        mark = ""
        if m["score"] < best:
            best, best_ep = m["score"], ep
            torch.save({"model": model.state_dict(), "metrics": m, "epoch": ep},
                       out / "best.pt")
            mark = "  <- лучшая"
        bs = m["by_source"]
        print(f"эпоха {ep:3d}  потеря {run / max(len(dl_tr), 1):.3f}  "
              f"Dice {m['dice']:.3f}  доход модели {m['reach_pred']:.2f} против "
              f"{m['reach_gt']:.2f} у эталона  ошибка {m['reach_err']:.3f}  "
              f"недотянуто >0.1 у {m['short']:.0%}, перетянуто у {m['over']:.0%}  "
              f"отбор {m['score']:.3f}"
              f"  [tf3 Dice {bs['tf3'].get('dice', float('nan')):.3f} ошибка "
              f"{bs['tf3'].get('reach_err', float('nan')):.3f} · "
              f"pulpy3d Dice {bs['pulpy3d'].get('dice', float('nan')):.3f} ошибка "
              f"{bs['pulpy3d'].get('reach_err', float('nan')):.3f}]{mark}", flush=True)
        if ep - best_ep >= patience:
            print(f"плато {patience} эпох — останавливаюсь")
            break
    ck = torch.load(out / "best.pt", weights_only=False)
    print(f"\nлучшая эпоха {ck['epoch']}: {json.dumps(ck['metrics'], ensure_ascii=False)}")
    print(f"веса: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
