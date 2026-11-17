"""Конвертация DENTEX (quadrant-enumeration-disease) из COCO JSON в YOLO-формат.

Используется только category_id_3 — 4 класса патологий:
    0: Impacted, 1: Caries, 2: Periapical Lesion, 3: Deep Caries

Выход:
    research/datasets/dentex_yolo/
        images/{train,val}/*.png   (hardlink, без копирования данных)
        labels/{train,val}/*.txt
        dentex.yaml
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENTEX = ROOT / "datasets" / "DENTEX" / "DENTEX"
OUT = ROOT / "datasets" / "dentex_yolo"

SPLITS = {
    "train": (
        DENTEX / "extracted/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
        DENTEX / "extracted/training_data/quadrant-enumeration-disease/xrays",
    ),
    "val": (
        DENTEX / "validation_triple.json",
        DENTEX / "extracted/validation_data/quadrant_enumeration_disease/xrays",
    ),
}

CLASSES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]


def convert_split(json_path: Path, xrays_dir: Path, split: str) -> tuple[int, int]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list] = {img_id: [] for img_id in images}
    for ann in data["annotations"]:
        cat3 = ann.get("category_id_3")
        if cat3 is None:
            continue  # зуб без патологии — пропускаем
        anns_by_image[ann["image_id"]].append(ann)

    img_out = OUT / "images" / split
    lbl_out = OUT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_boxes = 0
    for img_id, img in images.items():
        w, h = img["width"], img["height"]
        src = xrays_dir / img["file_name"]
        if not src.exists():
            print(f"  ! нет файла: {src}")
            continue

        dst = img_out / img["file_name"]
        if not dst.exists():
            os.link(src, dst)  # hardlink — не дублируем гигабайты

        lines = []
        for ann in anns_by_image[img_id]:
            x, y, bw, bh = ann["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            # клип в [0, 1] на случай выхода за границы
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{ann['category_id_3']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_boxes += 1

        (lbl_out / (Path(img["file_name"]).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    return len(images), n_boxes


def write_yaml() -> Path:
    yaml_path = OUT / "dentex.yaml"
    names = "\n".join(f"  {i}: {name!r}" for i, name in enumerate(CLASSES))
    yaml_path.write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n\nnc: {len(CLASSES)}\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    for split, (json_path, xrays_dir) in SPLITS.items():
        n_img, n_boxes = convert_split(json_path, xrays_dir, split)
        print(f"{split}: {n_img} изображений, {n_boxes} боксов")

    yaml_path = write_yaml()
    print(f"yaml: {yaml_path}")


if __name__ == "__main__":
    main()
