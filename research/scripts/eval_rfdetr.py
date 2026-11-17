"""Финальная оценка RF-DETR на валидации DENTEX: COCO mAP50 / mAP50-95 по классам.

Запуск:
    py research/scripts/eval_rfdetr.py [--checkpoint research/output/runs/rfdetr_large/checkpoint_best_total.pth]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRLarge

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "datasets" / "dentex_coco" / "valid"
CLASSES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "output/runs/rfdetr_large/checkpoint_best_total.pth"))
    args = parser.parse_args()

    coco_gt = COCO(str(VALID / "_annotations.coco.json"))
    model = RFDETRLarge(pretrain_weights=args.checkpoint)

    preds = []
    for img_info in coco_gt.loadImgs(coco_gt.getImgIds()):
        img = Image.open(VALID / img_info["file_name"]).convert("RGB")
        det = model.predict(img, threshold=0.001)
        for xyxy, conf, cls in zip(det.xyxy, det.confidence, det.class_id):
            x1, y1, x2, y2 = map(float, xyxy)
            preds.append({
                "image_id": img_info["id"],
                "category_id": int(cls),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(conf),
            })

    if not preds:
        print("Нет предсказаний!")
        return

    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # по классам
    print(f"{'Class':<20} {'AP50':>8} {'AP50:95':>8}")
    for i, name in enumerate(CLASSES):
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [i]
        ev.evaluate()
        ev.accumulate()
        p50 = ev.eval["precision"][0, :, 0, 0, -1]   # IoU=0.5
        p5095 = ev.eval["precision"][:, :, 0, 0, -1]  # все IoU
        ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else float("nan")
        ap = float(np.mean(p5095[p5095 > -1])) if (p5095 > -1).any() else float("nan")
        print(f"{name:<20} {ap50:8.3f} {ap:8.3f}")


if __name__ == "__main__":
    main()
