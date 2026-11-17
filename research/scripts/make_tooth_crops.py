"""Нарезка кропов зубов с метками патологий — данные для второго этапа.

    .venv\\Scripts\\python.exe research\\scripts\\make_tooth_crops.py

ОТКУДА БЕРУТСЯ МЕТКИ. В DENTEX две подвыборки:

    quadrant_enumeration           634 снимка, ВСЕ зубы с номерами
    quadrant-enumeration-disease   705 снимков, БОЛЬНЫЕ зубы с номерами
                                   и патологией

Первую даёт «где зуб и какой», вторую — «что с ним не так». Нужны обе.

ГЛАВНАЯ ЛОВУШКА, СТОИВШАЯ ОДНОГО ОБУЧЕНИЯ: ИМЕНА ФАЙЛОВ В ПОДВЫБОРКАХ
НЕЗАВИСИМЫ. `train_283.png` в одной и в другой — РАЗНЫЕ снимки: из 60
проверенных пар не совпал ни один байт, а размеры расходятся у 622 файлов из
634 (2710x1316 против 2872x1504). Первая версия сопоставляла по имени, метки
патологий легли на чужие снимки, и классификатор учился на шуме — F1 кариеса
держался около 0.27 и не рос.

Меряется это сразу: перекрытие рамок ОДНОГО И ТОГО ЖЕ зуба в двух
подвыборках.

    сопоставление по имени файла        медиана IoU 0.12, IoU>0.7 у 2%
    сопоставление по содержимому        медиана IoU 0.94, IoU>0.7 у 88%

Поэтому соответствие строится по MD5 файлов (`enum_to_disease.json`): одни и
те же снимки лежат в подвыборках под разными именами. Их 260 из 634 — только
на них метки достоверны, и только они идут в набор.

ГЛАВНАЯ ОГОВОРКА, и её надо помнить при чтении метрик. «Здоровый» здесь
означает «не отмечен как больной», а не «проверен и признан здоровым».
Разметка патологий может быть неполной, и часть кропов в классе `norm`
на деле больна. Это занижает измеренную точность, но не завышает — то есть
ошибается в безопасную сторону.

ДВЕ ГОЛОВЫ, А НЕ ОДИН КЛАСС. Кариес и периапикальный очаг живут на одном зубе
одновременно, и слить их в один набор классов значило бы заставить сеть
выбирать между ними. Поэтому метки независимы:

    caries    0 нет · 1 кариес · 2 глубокий кариес
    apical    0 нет · 1 периапикальный очаг

`Impacted` в кропы НЕ идёт: ретенция — это про положение зуба в челюсти, её
видно по всему снимку, а не по вырезанному зубу. Её и так находит первый этап.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "research/datasets/DENTEX/DENTEX/extracted/training_data"
ENUM = BASE / "quadrant_enumeration/train_quadrant_enumeration.json"
DIS = BASE / "quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
DST = REPO / "research/datasets/tooth_crops"
MAP = REPO / "research/datasets/DENTEX/enum_to_disease.json"

# Запас по краям кропа. Тот же, что в `teeth_detect.crop`: кариес на
# контактной поверхности и убыль кости у шейки лежат ровно на границе зуба,
# и без запаса кроп отрезал бы то, ради чего берётся.
PAD = 0.25

# Разбиение по СНИМКАМ и ТЕМ ЖЕ сидом, что у детектора: иначе снимок из
# обучения детектора попал бы в проверку классификатора, и оценка каскада
# завысилась бы на утечке.
VAL_FRACTION = 0.15
SEED = 0

CARIES = {"Caries": 1, "Deep Caries": 2}


def build(min_side: int = 24) -> None:
    enum = json.loads(ENUM.read_text(encoding="utf-8"))
    dis = json.loads(DIS.read_text(encoding="utf-8"))

    # Соответствие снимков по СОДЕРЖИМОМУ, а не по имени (см. докстринг).
    mapping = json.loads(MAP.read_text(encoding="utf-8"))

    dis_names = {im["id"]: im["file_name"] for im in dis["images"]}
    dis_cats = {c["id"]: c["name"] for c in dis["categories_3"]}
    # (файл, номер зуба) -> набор патологий
    sick: dict[tuple[str, int], set[str]] = {}
    for a in dis["annotations"]:
        fdi = (a["category_id_1"] + 1) * 10 + a["category_id_2"] + 1
        key = (dis_names[a["image_id"]], fdi)
        sick.setdefault(key, set()).add(dis_cats[a["category_id_3"]])

    images = {im["id"]: im for im in enum["images"]}
    by_img: dict[int, list] = {}
    for a in enum["annotations"]:
        fdi = (a["category_id_1"] + 1) * 10 + a["category_id_2"] + 1
        by_img.setdefault(a["image_id"], []).append((fdi, a["bbox"]))

    # Оставляем только снимки, для которых есть парная разметка патологий.
    ids = sorted(i for i in by_img
                 if images[i]["file_name"] in mapping)
    random.Random(SEED).shuffle(ids)
    n_val = max(1, int(len(ids) * VAL_FRACTION))
    val_ids = set(ids[:n_val])

    rows, counts = [], Counter()
    src_dir = ENUM.parent / "xrays"
    for img_id in ids:
        im_info = images[img_id]
        path = src_dir / im_info["file_name"]
        if not path.is_file():
            continue
        split = "val" if img_id in val_ids else "train"
        with Image.open(path) as im:
            im = im.convert("L")
            for fdi, (x, y, w, h) in by_img[img_id]:
                if w < min_side or h < min_side:
                    continue
                dx, dy = w * PAD, h * PAD
                box = (max(x - dx, 0), max(y - dy, 0),
                       min(x + w + dx, im.width), min(y + h + dy, im.height))
                crop = im.crop(tuple(int(v) for v in box))

                tags = sick.get((mapping[im_info["file_name"]], fdi), set())
                caries = max((CARIES.get(t, 0) for t in tags), default=0)
                apical = int("Periapical Lesion" in tags)

                name = f"{Path(im_info['file_name']).stem}_{fdi}.png"
                out = DST / split / name
                out.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out)
                rows.append({"file": f"{split}/{name}", "split": split,
                             "fdi": fdi, "caries": caries, "apical": apical})
                counts[(split, caries, apical)] += 1

    (DST / "labels.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    print(f"кропов: {len(rows):,} из {len(ids)} снимков")
    for split in ("train", "val"):
        n = sum(v for (s, _, _), v in counts.items() if s == split)
        car = Counter()
        ap = 0
        for (s, c, a), v in counts.items():
            if s != split:
                continue
            car[c] += v
            ap += v if a else 0
        print(f"  {split}: {n:,} | кариес нет/есть/глубокий "
              f"{car[0]:,}/{car[1]:,}/{car[2]:,} | очаг {ap:,}")
    print(f"метки: {DST / 'labels.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-side", type=int, default=24,
                    help="пропускать рамки мельче, px")
    args = ap.parse_args()
    build(min_side=args.min_side)


if __name__ == "__main__":
    main()
