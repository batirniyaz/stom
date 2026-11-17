"""Конвертация DENTEX disease-разметки в стандартный COCO для RF-DETR.

RF-DETR ждёт структуру:
    dataset_dir/
        train/_annotations.coco.json + *.png
        valid/_annotations.coco.json + *.png

Исходные JSON иерархические (category_id_1/2/3) — берём только category_id_3
(4 патологии) и приводим к обычному COCO (category_id / categories).

Запуск:
    py research/scripts/convert_dentex_coco.py
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENTEX = ROOT / "datasets" / "DENTEX" / "DENTEX"
OUT = ROOT / "datasets" / "dentex_coco"

SPLITS = {
    "train": (
        DENTEX / "extracted/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
        DENTEX / "extracted/training_data/quadrant-enumeration-disease/xrays",
    ),
    "valid": (
        DENTEX / "validation_triple.json",
        DENTEX / "extracted/validation_data/quadrant_enumeration_disease/xrays",
    ),
}

CATEGORIES = [
    {"id": 0, "name": "Impacted", "supercategory": "disease"},
    {"id": 1, "name": "Caries", "supercategory": "disease"},
    {"id": 2, "name": "Periapical Lesion", "supercategory": "disease"},
    {"id": 3, "name": "Deep Caries", "supercategory": "disease"},
]


def convert_split(json_path: Path, xrays_dir: Path, split: str) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    out_dir = OUT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    for img in data["images"]:
        src = xrays_dir / img["file_name"]
        if not src.exists():
            print(f"  ! нет файла: {src}")
            continue
        dst = out_dir / img["file_name"]
        if not dst.exists():
            os.link(src, dst)
        images.append(
            {"id": img["id"], "file_name": img["file_name"],
             "width": img["width"], "height": img["height"]}
        )

    for ann in data["annotations"]:
        cat3 = ann.get("category_id_3")
        if cat3 is None:
            continue
        annotations.append({
            "id": ann_id,
            "image_id": ann["image_id"],
            "category_id": cat3,
            "bbox": ann["bbox"],
            "area": ann["area"],
            "iscrowd": 0,
            "segmentation": ann.get("segmentation", []),
        })
        ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    with open(out_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"{split}: {len(images)} изображений, {len(annotations)} боксов")


def main() -> None:
    for split, (json_path, xrays_dir) in SPLITS.items():
        convert_split(json_path, xrays_dir, split)
    print(f"dataset_dir: {OUT}")


if __name__ == "__main__":
    main()
