"""Кропы зубов со ВСЕЙ подвыборки патологий — расширенный набор для 2-го этапа.

    .venv\\Scripts\\python.exe research\\scripts\\make_tooth_crops2.py

ЗАЧЕМ ВТОРОЙ КОНВЕРТЕР. Первый (`make_tooth_crops.py`) берёт только те 260
снимков, что есть в обеих подвыборках DENTEX — иначе метки патологий легли бы
на чужие снимки (имена файлов в подвыборках независимы, см. CLAUDE.md).
Достоверно, но мало: 7 471 кроп, и классификатор переобучался до потери 0.002
при F1 0.39.

Здесь данные берутся из подвыборки патологий ЦЕЛИКОМ — 705 снимков. Номер
зуба для этого не нужен, а значит не нужна и вторая разметка:

    больные зубы   размечены прямо здесь: рамка зуба + патология
    здоровые зубы  находит наш детектор (`teeth_detect`), и здоровым
                   считается зуб, чья рамка НЕ перекрылась с больной

РАМКА В ЭТОЙ ПОДВЫБОРКЕ — ЭТО ЗУБ, А НЕ ПАТОЛОГИЯ. Проверено: у зуба 36 с
кариесом и периапикальным очагом обе рамки различаются на два пикселя
(1739,778,228,302 против 1737,776,228,302), а медианная площадь «рамки
патологии» на треть БОЛЬШЕ рамки зуба из соседней подвыборки. То есть внутри
зуба локализации нет, и детекция патологии внутри кропа на этих данных
невозможна — только классификация кропа целиком.

ЧЕМ ПЛАТИМ ЗА ОБЪЁМ. «Здоровый» тут определяется нашим же детектором: если он
пропустил зуб, кроп не появится вовсе, а если ошибся рамкой — здоровый зуб
может перекрыться с больным и получить чужую метку. Порог перекрытия поэтому
высокий (0.5), а не «любое касание».
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = REPO / "research/datasets/DENTEX/DENTEX/extracted/training_data"
DIS = BASE / "quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
XRAYS = DIS.parent / "xrays"
DST = REPO / "research/datasets/tooth_crops2"

PAD = 0.25
VAL_FRACTION = 0.15
SEED = 0
CARIES = {"Caries": 1, "Deep Caries": 2}

# Перекрытие, при котором найденный зуб считается тем же, что размеченный.
# Высокое намеренно: при низком пороге сосед больного зуба получал бы его
# метку, и в классе «здоров» оказывались бы больные.
IOU_SAME = 0.5


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / (aw * ah + bw * bh - i)


def build(conf: float = 0.30) -> None:
    import teeth_detect as td

    dis = json.loads(DIS.read_text(encoding="utf-8"))
    names = {im["id"]: im["file_name"] for im in dis["images"]}
    cats = {c["id"]: c["name"] for c in dis["categories_3"]}

    sick: dict[str, list] = {}
    for a in dis["annotations"]:
        sick.setdefault(names[a["image_id"]], []).append(
            (a["bbox"], cats[a["category_id_3"]]))

    files = sorted(sick)
    random.Random(SEED).shuffle(files)
    n_val = max(1, int(len(files) * VAL_FRACTION))
    val_files = set(files[:n_val])

    model = td._load()
    rows, counts = [], Counter()
    for k, fname in enumerate(files, 1):
        path = XRAYS / fname
        if not path.is_file():
            continue
        dets, _ = td.resolve(td.detect(str(path), model=model, conf=conf))
        if not dets:
            continue
        split = "val" if fname in val_files else "train"
        with Image.open(path) as im:
            im = im.convert("L")
            for d in dets:
                cx, cy, w, h = d["xywh"]
                box = (cx - w / 2, cy - h / 2, w, h)
                tags = {t for bb, t in sick[fname] if iou(box, bb) >= IOU_SAME}
                caries = max((CARIES.get(t, 0) for t in tags), default=0)
                apical = int("Periapical Lesion" in tags)

                dx, dy = w * PAD, h * PAD
                crop = im.crop((int(max(cx - w / 2 - dx, 0)),
                                int(max(cy - h / 2 - dy, 0)),
                                int(min(cx + w / 2 + dx, im.width)),
                                int(min(cy + h / 2 + dy, im.height))))
                name = f"{path.stem}_{d['fdi']}.png"
                out = DST / split / name
                out.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out)
                rows.append({"file": f"{split}/{name}", "split": split,
                             "fdi": d["fdi"], "caries": caries,
                             "apical": apical})
                counts[(split, caries, apical)] += 1
        if k % 100 == 0:
            print(f"  обработано снимков: {k}/{len(files)}", flush=True)

    (DST / "labels.json").write_text(json.dumps(rows, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nкропов: {len(rows):,} из {len(files)} снимков")
    for split in ("train", "val"):
        car, ap = Counter(), 0
        for (s, c, a), v in counts.items():
            if s != split:
                continue
            car[c] += v
            ap += v if a else 0
        n = sum(car.values())
        print(f"  {split}: {n:,} | кариес нет/есть/глубокий "
              f"{car[0]:,}/{car[1]:,}/{car[2]:,} | очаг {ap:,}")
    print(f"метки: {DST / 'labels.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()
    build(conf=args.conf)


if __name__ == "__main__":
    main()
