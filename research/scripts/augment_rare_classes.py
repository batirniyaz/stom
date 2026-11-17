"""Position-aware copy-paste аугментация редких классов DENTEX.

Идея: панорамные снимки стандартизированы по композиции, поэтому объект
(зуб с патологией), вырезанный по полигональной маске, можно вставить в ту же
нормализованную позицию другого снимка — это анатомически правдоподобно.

Для каждого объекта редкого класса создаём N копий на случайных донорах.
Донор копируется целиком (со всеми своими боксами) + вставленный объект.
Если на месте вставки уже есть бокс с IoU > 0.5 — берём другого донора.

Выход: research/datasets/dentex_yolo_aug/ — оригинальный train + аугментированные
изображения (hardlink оригиналов), val не трогаем.

Запуск:
    py research/scripts/augment_rare_classes.py
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DENTEX = ROOT / "datasets" / "DENTEX" / "DENTEX"
SRC_JSON = DENTEX / "extracted/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
SRC_XRAYS = DENTEX / "extracted/training_data/quadrant-enumeration-disease/xrays"
OUT = ROOT / "datasets" / "dentex_yolo_aug"

CLASSES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
# сколько копий каждого объекта класса создать
COPIES = {0: 2, 2: 8, 3: 3}  # Impacted x2, Periapical x8, Deep Caries x3
JITTER = 0.03      # джиттер позиции вставки (доля размера изображения)
MAX_IOU = 0.5      # порог конфликта с существующими боксами донора
PER_IMAGE = 3      # макс. вставок в один аугментированный снимок


def bbox_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def extract_patch(img, ann):
    """Вырезает объект по полигону: возвращает (patch_bgr, alpha_mask, bbox)."""
    x, y, w, h = [int(round(v)) for v in ann["bbox"]]
    H, W = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg in ann.get("segmentation", []):
        pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
        pts[:, 0] -= x
        pts[:, 1] -= y
        cv2.fillPoly(mask, [pts], 255)
    if mask.sum() == 0:
        mask[:] = 255  # нет полигона — берём весь бокс
    patch = img[y:y + h, x:x + w]
    return patch, mask, (x, y, w, h)


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    with open(SRC_JSON, encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        if ann.get("category_id_3") is not None:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out = OUT / "images" / "train"
    lbl_out = OUT / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # 1. копируем оригинальный train (hardlink) — берём из dentex_yolo
    src_yolo = ROOT / "datasets" / "dentex_yolo"
    n_orig = 0
    for f in (src_yolo / "images" / "train").glob("*.png"):
        dst = img_out / f.name
        if not dst.exists():
            os.link(f, dst)
        n_orig += 1
    for f in (src_yolo / "labels" / "train").glob("*.txt"):
        dst = lbl_out / f.name
        if not dst.exists():
            os.link(f, dst)
    print(f"оригиналов: {n_orig}")

    # val — симлинк на оригинальный (конфиг укажет на dentex_yolo/val)
    # 2. собираем донорский пул и объекты редких классов
    donor_ids = list(images.keys())
    img_cache: dict[int, np.ndarray] = {}

    def load_img(img_id):
        if img_id not in img_cache:
            if len(img_cache) > 30:  # простой лимит кэша
                img_cache.pop(next(iter(img_cache)))
            img_cache[img_id] = cv2.imread(str(SRC_XRAYS / images[img_id]["file_name"]))
        return img_cache[img_id]

    rare = [a for img_id, anns in anns_by_image.items() for a in anns
            if a["category_id_3"] in COPIES for _ in range(COPIES[a["category_id_3"]])]
    random.shuffle(rare)
    print(f"объектов для вставки: {len(rare)}")

    # 3. вставляем пачками по PER_IMAGE в один аугментированный снимок
    n_aug, n_pasted, i = 0, 0, 0
    while i < len(rare):
        donor_id = random.choice(donor_ids)
        donor_info = images[donor_id]
        donor_img = load_img(donor_id).copy()
        H, W = donor_img.shape[:2]
        donor_boxes = [a["bbox"] for a in anns_by_image.get(donor_id, [])]
        new_labels = list(anns_by_image.get(donor_id, []))
        pasted_here = 0

        while pasted_here < PER_IMAGE and i < len(rare):
            ann = rare[i]
            src_img = load_img(ann["image_id"])
            patch, mask, (px, py, pw, ph) = extract_patch(src_img, ann)
            src_info = images[ann["image_id"]]
            # нормализованный центр на исходном снимке -> позиция на доноре
            cx = (px + pw / 2) / src_info["width"] + random.uniform(-JITTER, JITTER)
            cy = (py + ph / 2) / src_info["height"] + random.uniform(-JITTER, JITTER)
            dx = int(cx * W - pw / 2)
            dy = int(cy * H - ph / 2)
            if dx < 0 or dy < 0 or dx + pw > W or dy + ph > H:
                i += 1
                continue
            new_bbox = [dx, dy, pw, ph]
            if any(bbox_iou(new_bbox, b) > MAX_IOU for b in donor_boxes):
                i += 1
                continue
            # вставка по маске
            roi = donor_img[dy:dy + ph, dx:dx + pw]
            m3 = cv2.merge([mask, mask, mask])
            np.copyto(roi, patch, where=m3 > 0)
            donor_boxes.append(new_bbox)
            new_labels.append({"bbox": new_bbox, "category_id_3": ann["category_id_3"]})
            pasted_here += 1
            n_pasted += 1
            i += 1

        if pasted_here == 0:
            continue
        name = f"aug_{n_aug:05d}"
        cv2.imwrite(str(img_out / f"{name}.png"), donor_img)
        lines = []
        for a in new_labels:
            x, y, bw, bh = a["bbox"]
            lines.append(f"{a['category_id_3']} {(x + bw / 2) / W:.6f} {(y + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}")
        (lbl_out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_aug += 1

    # 4. yaml
    names = "\n".join(f"  {i_}: {n!r}" for i_, n in enumerate(CLASSES))
    (OUT / "dentex_aug.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\n"
        f"val: {(src_yolo / 'images' / 'val').as_posix()}\n\nnc: 4\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print(f"аугментированных снимков: {n_aug}, вставлено объектов: {n_pasted}")
    print(f"yaml: {OUT / 'dentex_aug.yaml'}")


if __name__ == "__main__":
    main()
