"""Классификатор кропа зуба — второй этап двухэтапной схемы.

    .venv\\Scripts\\python.exe research\\scripts\\train_tooth_cls.py --epochs 30
    .venv\\Scripts\\python.exe research\\scripts\\train_tooth_cls.py \\
        --data tooth_crops2 clinic_tooth_crops --out research/output/tooth_cls_clinic

На вход идёт вырезанный зуб (первый этап нашёл его и назвал номером), на
выходе две независимые головы:

    caries   0 нет · 1 кариес · 2 глубокий кариес
    apical   0 нет · 1 периапикальный очаг

ПОЧЕМУ ДВЕ ГОЛОВЫ, А НЕ ОДИН НАБОР КЛАССОВ. Кариес и очаг бывают на одном
зубе одновременно; слитые в один классификатор, они заставили бы сеть
выбирать между ними.

ПОЧЕМУ ВЗВЕШИВАНИЕ ОБЯЗАТЕЛЬНО. Классы редкие: из 15 372 обучающих кропов
кариес у 1 414, глубокий у 342, очаг у 110. Без весов сеть выучит «здоров» —
и получит 89% точности, не найдя ни одного больного зуба. Вес класса берём
обратным его частоте.

ЧЕГО ЖДАТЬ ОТ ГОЛОВЫ ОЧАГА: немного. 110 примеров на обучение и 23 на
проверку — это разведка, а не рабочая модель. Числа по ней надо читать как
«есть ли сигнал вообще», и на такой выборке одна ошибка двигает метрику на
четыре процента.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "research/datasets"
DATA = DATASETS / "tooth_crops2"
# Эти веса читает демо (`opg_teeth_findings`). Пробные прогоны — только с
# `--out` в другую папку, иначе врач молча получит недоученную модель.
OUT = REPO / "research/output/tooth_cls"

# Вход сети ВЫТЯНУТЫЙ, а не квадратный.
#
# Зуб на панораме — вертикальный объект: замер по кропам дал медиану
# 158 x 466 пикселей, то есть высота втрое больше ширины. Приведение к
# квадрату 224x224 сплющивало зуб по вертикали втрое, и кариес — мелкая
# деталь на контактной поверхности — размазывался. На первых эпохах это
# давало F1 около 0.27 и не росло.
#
# Размер кратен 32 (требование свёрточной сети) и близок к медиане кропа,
# поэтому большинство зубов почти не масштабируется.
IN_W, IN_H = 160, 448
CARIES_N, APICAL_N = 3, 2


class Crops(Dataset):
    def __init__(self, rows, train: bool):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(r["_root"] / r["file"]).convert("L").resize(
            (IN_W, IN_H), Image.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
        if self.train:
            # Отражение по горизонтали допустимо: номер зуба здесь не
            # предсказывается, а кариес зеркальной симметрией не различается.
            if np.random.rand() < 0.5:
                a = a[:, ::-1].copy()
            # Яркость и контраст: панорамы с разных аппаратов заметно
            # различаются экспозицией, и без этого сеть цепляется за неё.
            a = np.clip(a * np.random.uniform(0.80, 1.20)
                        + np.random.uniform(-0.10, 0.10), 0, 1)
            # Сдвиг кропа. Рамка от детектора не совпадает с рамкой разметки,
            # и без этого сеть привыкает к идеально отцентрованному зубу —
            # а в работе кроп ей даёт детектор со своей погрешностью.
            sh = int(IN_W * 0.06)
            dx, dy = np.random.randint(-sh, sh + 1, 2)
            a = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
            # Стирание случайного участка: заставляет смотреть на весь зуб,
            # а не запоминать одну область. Дешёвая замена сложным приёмам.
            if np.random.rand() < 0.35:
                hh = np.random.randint(IN_H // 12, IN_H // 5)
                ww = np.random.randint(IN_W // 8, IN_W // 3)
                y0 = np.random.randint(0, IN_H - hh)
                x0 = np.random.randint(0, IN_W - ww)
                a[y0:y0 + hh, x0:x0 + ww] = float(a.mean())
        x = torch.from_numpy(a)[None].repeat(3, 1, 1)     # серый -> 3 канала
        return x, r["caries"], r["apical"]


class Net(nn.Module):
    """ResNet18 с двумя головами.

    Восемнадцатый слой, а не глубже: обучающих кропов 15 тысяч, и более
    крупная сеть на таком объёме переучивается быстрее, чем успевает выучить
    редкий класс.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.body = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat = self.body.fc.in_features
        self.body.fc = nn.Identity()
        # Прореживание перед головами. Без него прошлый прогон дошёл до
        # потери 0.002 при F1 0.39 — сеть выучила обучающую выборку наизусть.
        self.drop = nn.Dropout(0.4)
        self.caries = nn.Linear(feat, CARIES_N)
        self.apical = nn.Linear(feat, APICAL_N)

    def forward(self, x):
        f = self.drop(self.body(x))
        return self.caries(f), self.apical(f)


def class_weights(values, n: int) -> torch.Tensor:
    cnt = np.bincount(np.asarray(values), minlength=n).astype(np.float64)
    cnt[cnt == 0] = 1.0
    w = cnt.sum() / (n * cnt)
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model, loader, device) -> dict:
    model.eval()
    pc, tc, pa, ta = [], [], [], []
    with torch.no_grad():
        for x, c, a in loader:
            oc, oa = model(x.to(device))
            pc += oc.argmax(1).cpu().tolist(); tc += c.tolist()
            pa += oa.argmax(1).cpu().tolist(); ta += a.tolist()
    pc, tc = np.array(pc), np.array(tc)
    pa, ta = np.array(pa), np.array(ta)

    def prf(pred, true, cls):
        tp = int(((pred == cls) & (true == cls)).sum())
        fp = int(((pred == cls) & (true != cls)).sum())
        fn = int(((pred != cls) & (true == cls)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp + fn

    out = {}
    for cls, name in ((1, "кариес"), (2, "глубокий")):
        out[name] = prf(pc, tc, cls)
    out["очаг"] = prf(pa, ta, 1)
    # Полнота по «есть кариес любой глубины» — то, что важно врачу: пропуск
    # опаснее, чем спутанная градация.
    any_p, any_t = (pc > 0), (tc > 0)
    tp = int((any_p & any_t).sum())
    p_any = tp / max(int(any_p.sum()), 1)
    r_any = tp / max(int(any_t.sum()), 1)
    out["любой кариес"] = (p_any, r_any,
                           2 * p_any * r_any / (p_any + r_any) if p_any + r_any else 0.0,
                           int(any_t.sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", nargs="+", default=[DATA.name],
                    help="наборы вырезов (имя в research/datasets или путь)")
    ap.add_argument("--out", default=str(OUT),
                    help="куда писать веса; по умолчанию — те, что читает демо")
    args = ap.parse_args()
    out_dir = Path(args.out)

    # Несколько наборов учатся вместе, а проверяются ПОРОЗНЬ: вопрос не «как
    # в среднем», а переносится ли модель на снимки клиники (другой аппарат).
    # Выгрузка разметки врачей (`export_annotations`) пишет в тот же формат.
    rows = []
    for name in args.data:
        root = Path(name) if Path(name).is_dir() else DATASETS / name
        part = json.loads((root / "labels.json").read_text(encoding="utf-8"))
        # Метка null — класс на этом снимке врач не проверял (см.
        # export_annotations): такой вырез этой голове не пример.
        keep = [r for r in part
                if r.get("caries") is not None and r.get("apical") is not None]
        for r in keep:
            r["_root"], r["_src"] = root, root.name
        print(f"{root.name}: вырезов {len(keep):,}"
              + (f" (без меток кариеса и очага {len(part) - len(keep)})"
                 if len(keep) < len(part) else ""))
        rows += keep
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    print(f"обучение {len(tr):,} кропов, проверка {len(va):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    wc = class_weights([r["caries"] for r in tr], CARIES_N).to(device)
    wa = class_weights([r["apical"] for r in tr], APICAL_N).to(device)
    print("веса классов кариеса:", wc.cpu().numpy().round(2),
          "| очага:", wa.cpu().numpy().round(2))
    loss_c = nn.CrossEntropyLoss(weight=wc)
    loss_a = nn.CrossEntropyLoss(weight=wa)

    dl_tr = DataLoader(Crops(tr, True), batch_size=args.batch, shuffle=True,
                       num_workers=4, drop_last=True)
    dl_va = DataLoader(Crops(va, False), batch_size=args.batch, num_workers=4)
    sources = sorted({r["_src"] for r in va})
    dl_src = {s: DataLoader(Crops([r for r in va if r["_src"] == s], False),
                            batch_size=args.batch, num_workers=4)
              for s in sources} if len(sources) > 1 else {}

    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_ep, PATIENCE = -1.0, 0, 12
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for x, c, a in dl_tr:
            x, c, a = x.to(device), c.to(device), a.to(device)
            oc, oa = model(x)
            loss = loss_c(oc, c) + loss_a(oa, a)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss)
        sched.step()
        m = evaluate(model, dl_va, device)
        # Отбираем по F1 кариеса: это основная задача головы, и она не
        # вырождается при дисбалансе, в отличие от общей точности.
        score = m["кариес"][2]
        mark = ""
        if score > best:
            best, best_ep = score, ep
            per_src = {s: evaluate(model, dl, device) for s, dl in dl_src.items()}
            torch.save({"model": model.state_dict(), "metrics": m,
                        "metrics_by_source": per_src, "data": args.data,
                        "epoch": ep}, out_dir / "best.pt")
            mark = "  <- лучшая"
        print(f"эпоха {ep:2d}  потеря {run / len(dl_tr):.3f}  "
              f"кариес F1 {m['кариес'][2]:.3f}  "
              f"глубокий F1 {m['глубокий'][2]:.3f}  "
              f"очаг F1 {m['очаг'][2]:.3f}{mark}", flush=True)
        # Ранняя остановка: держать сеть после плато значит только глубже
        # заучивать обучающую выборку.
        if ep - best_ep >= PATIENCE:
            print(f"плато {PATIENCE} эпох — останавливаюсь")
            break

    ck = torch.load(out_dir / "best.pt", weights_only=False)
    print(f"\nлучшая эпоха: {ck['epoch']}")
    blocks = [("все наборы", ck["metrics"])]
    blocks += list(ck.get("metrics_by_source", {}).items())
    for title, met in blocks:
        print(f"  {title}:")
        for name, (p, r, f1, n) in met.items():
            print(f"    {name:14s} точность {p:.3f}  полнота {r:.3f}  "
                  f"F1 {f1:.3f}   (в проверке {n})")
    print(f"веса: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
