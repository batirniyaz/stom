"""DENTEX -> YOLO: детекция и НУМЕРАЦИЯ каждого зуба (32 класса FDI).

    .venv\\Scripts\\python.exe research\\scripts\\convert_dentex_teeth.py

ЗАЧЕМ ОТДЕЛЬНЫЙ КОНВЕРТЕР. `convert_dentex_yolo.py` берёт подвыборку
`quadrant-enumeration-disease` и оставляет из неё ТОЛЬКО патологию
(`category_id_3`), выбрасывая номер зуба. Этого хватало, пока 2D работал одной
сетью на весь снимок.

Для двухэтапной схемы нужен первый этап — найти каждый зуб и назвать его
номером. Под это в DENTEX есть своя подвыборка `quadrant_enumeration`, где
размечены ВСЕ зубы, а не только больные:

    quadrant-enumeration-disease   705 снимков, 3529 меток (5.0 на снимок)
    quadrant_enumeration           634 снимка, 18095 меток (28.5 на снимок)

Номер собирается из двух полей: `category_id_1` даёт квадрант (0-3 -> 1-4),
`category_id_2` — позицию в квадранте (0-7 -> 1-8). FDI = квадрант*10 + позиция.

ПОЧЕМУ 32 КЛАССА, А НЕ ОДИН «ЗУБ» ПЛЮС РЕГРЕССИЯ НОМЕРА. Номер зуба это не
величина, а метка: между 18 и 21 нет «среднего». Плюс сеть, знающая номер,
опирается на форму — моляр не спутать с резцом, и это дополнительный сигнал
против пропусков.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = (REPO / "research/datasets/DENTEX/DENTEX/extracted/training_data"
       / "quadrant_enumeration")
DST = REPO / "research/datasets/dentex_teeth"

# Доля снимков в проверочной части. Разбиение по СНИМКАМ, а не по зубам:
# зубы одного пациента похожи, и разнеси их по обе стороны — метрика
# завысится. Та же ошибка, что в официальном разбиении Teeth3DS, где 382
# пациента попали и в обучение, и в тест.
VAL_FRACTION = 0.15
SEED = 0

FDI = [q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9)]   # 11..48
FDI_TO_CLS = {f: i for i, f in enumerate(FDI)}


def convert(link: bool = True) -> None:
    ann_path = SRC / "train_quadrant_enumeration.json"
    if not ann_path.is_file():
        raise SystemExit(
            f"нет разметки: {ann_path}\n"
            "распакуйте подвыборку quadrant_enumeration из training_data.zip")
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    images = {im["id"]: im for im in data["images"]}
    by_image: dict[int, list] = {}
    skipped = 0
    for a in data["annotations"]:
        q = a["category_id_1"] + 1          # индекс -> номер квадранта
        pos = a["category_id_2"] + 1        # индекс -> позиция в квадранте
        fdi = q * 10 + pos
        if fdi not in FDI_TO_CLS:
            skipped += 1
            continue
        by_image.setdefault(a["image_id"], []).append((fdi, a["bbox"]))

    ids = sorted(by_image)
    random.Random(SEED).shuffle(ids)
    n_val = max(1, int(len(ids) * VAL_FRACTION))
    val_ids = set(ids[:n_val])

    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    per_image = []
    for img_id in ids:
        im = images[img_id]
        split = "val" if img_id in val_ids else "train"
        src_img = SRC / "xrays" / im["file_name"]
        if not src_img.is_file():
            continue
        dst_img = DST / "images" / split / im["file_name"]
        if not dst_img.exists():
            # Жёсткая ссылка вместо копии: снимки весят гигабайты, а лежат на
            # том же диске. Так же сделано в соседних конвертерах.
            try:
                if link:
                    dst_img.hardlink_to(src_img)
                else:
                    shutil.copy2(src_img, dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

        w, h = im["width"], im["height"]
        lines = []
        for fdi, (x, y, bw, bh) in by_image[img_id]:
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                continue
            lines.append(f"{FDI_TO_CLS[fdi]} {cx:.6f} {cy:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
            counts[fdi] += 1
        (DST / "labels" / split / f"{Path(im['file_name']).stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        per_image.append(len(lines))

    names = "\n".join(f"  {i}: {f}" for i, f in enumerate(FDI))
    (DST / "dentex_teeth.yaml").write_text(
        f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\n\n"
        f"names:\n{names}\n", encoding="utf-8")

    print(f"снимков: {len(per_image)} (обучение {len(ids) - n_val}, "
          f"проверка {n_val})")
    print(f"зубов размечено: {sum(counts.values()):,}, "
          f"в среднем на снимок {sum(per_image) / max(len(per_image), 1):.1f}")
    if skipped:
        print(f"пропущено меток с неверным номером: {skipped}")
    rare = sorted(counts.items(), key=lambda kv: kv[1])[:6]
    print("самые редкие зубы:", ", ".join(f"{f}:{n}" for f, n in rare))
    miss = [f for f in FDI if counts[f] == 0]
    if miss:
        print("НЕТ ПРИМЕРОВ вовсе:", miss)
    print(f"конфиг: {DST / 'dentex_teeth.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", action="store_true",
                    help="копировать снимки вместо жёстких ссылок")
    args = ap.parse_args()
    convert(link=not args.copy)


if __name__ == "__main__":
    main()
