"""Оценка ансамбля YOLO12x + YOLO26x на валидации DENTEX (Weighted Boxes Fusion).

Сравнивает: каждая модель отдельно vs ансамбль (WBF по классам).
Метрики: COCO mAP50 / mAP50-95, в целом и по классам.

Запуск:
    py research/scripts/eval_ensemble.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VALID_IMG = ROOT / "datasets/dentex_yolo/images/val"
VALID_JSON = ROOT / "datasets/dentex_coco/valid/_annotations.coco.json"
RUNS = ROOT / "output/runs"

MODELS = {
    "yolo12x": RUNS / "yolo12x_imgsz960/weights/best.pt",
    "yolo26x": RUNS / "yolo26x_tuned/weights/best.pt",
}
CLASSES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
CONF = 0.01          # низкий порог — полная PR-кривая на этапе COCOeval
WBF_IOU = 0.55


def predict_all(model: YOLO, path: Path) -> list:
    """[x1, y1, x2, y2, score, cls] в абсолютных координатах."""
    res = model.predict(str(path), imgsz=960, conf=CONF, max_det=100, verbose=False)[0]
    out = []
    for xyxy, cf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy(),
                             res.boxes.cls.cpu().numpy().astype(int)):
        out.append([*map(float, xyxy), float(cf), int(cls)])
    return out


def wbf(dets: list, iou_thr: float = WBF_IOU) -> list:
    """Weighted Boxes Fusion отдельно по каждому классу."""
    fused = []
    for cls in range(len(CLASSES)):
        d = sorted([x for x in dets if x[5] == cls], key=lambda x: -x[4])
        clusters = []  # каждый: [boxes...]
        for det in d:
            placed = False
            for cl in clusters:
                rep = cl[0]
                xx1, yy1 = max(det[0], rep[0]), max(det[1], rep[1])
                xx2, yy2 = min(det[2], rep[2]), min(det[3], rep[3])
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                a = (det[2] - det[0]) * (det[3] - det[1])
                b = (rep[2] - rep[0]) * (rep[3] - rep[1])
                if inter / (a + b - inter + 1e-9) >= iou_thr:
                    cl.append(det)
                    placed = True
                    break
            if not placed:
                clusters.append([det])
        for cl in clusters:
            wsum = sum(x[4] for x in cl)
            x1 = sum(x[0] * x[4] for x in cl) / wsum
            y1 = sum(x[1] * x[4] for x in cl) / wsum
            x2 = sum(x[2] * x[4] for x in cl) / wsum
            y2 = sum(x[3] * x[4] for x in cl) / wsum
            # стандартный WBF score: средний вес с бонусом за согласие моделей
            score = min(1.0, (wsum / len(cl)) * (len(cl) / len(MODELS) + 0.5))
            fused.append([x1, y1, x2, y2, score, cls])
    return fused


def coco_map(gt: COCO, preds: list) -> tuple:
    """Возвращает (mAP50, mAP50-95, per-class список)."""
    if not preds:
        return 0.0, 0.0, [(float("nan"),) * 2] * len(CLASSES)
    dt = gt.loadRes(preds)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    m50, m5095 = ev.stats[1], ev.stats[0]
    per_class = []
    for i in range(len(CLASSES)):
        ev2 = COCOeval(gt, dt, "bbox")
        ev2.params.catIds = [i]
        ev2.evaluate(); ev2.accumulate()
        p = ev2.eval["precision"]
        p50 = p[0, :, 0, 0, -1]
        pall = p[:, :, 0, 0, -1]
        ap50 = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else float("nan")
        ap = float(np.mean(pall[pall > -1])) if (pall > -1).any() else float("nan")
        per_class.append((ap50, ap))
    return m50, m5095, per_class


def main() -> None:
    gt = COCO(str(VALID_JSON))
    imgs = {i["id"]: i["file_name"] for i in gt.loadImgs(gt.getImgIds())}

    models = {name: YOLO(str(w)) for name, w in MODELS.items()}
    preds_single = {name: [] for name in models}
    preds_ens = []

    for img_id, fname in imgs.items():
        path = VALID_IMG / fname
        all_dets = []
        for name, m in models.items():
            dets = predict_all(m, path)
            all_dets += dets
            for x1, y1, x2, y2, s, c in dets:
                preds_single[name].append({"image_id": img_id, "category_id": c,
                                           "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
        for x1, y1, x2, y2, s, c in wbf(all_dets):
            preds_ens.append({"image_id": img_id, "category_id": c,
                              "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})

    print(f"\n{'Модель':<14} {'mAP50':>7} {'mAP50-95':>9}")
    results = {}
    for name in models:
        m50, m5095, _ = coco_map(gt, preds_single[name])
        results[name] = (m50, m5095)
        print(f"{name:<14} {m50:7.3f} {m5095:9.3f}")
    m50, m5095, per_class = coco_map(gt, preds_ens)
    results["ensemble"] = (m50, m5095)
    print(f"{'ENSEMBLE WBF':<14} {m50:7.3f} {m5095:9.3f}")

    print(f"\n{'Класс':<20} {'AP50':>7} {'AP50-95':>8}  (ансамбль)")
    for (ap50, ap), name in zip(per_class, CLASSES):
        print(f"{name:<20} {ap50:7.3f} {ap:8.3f}")


if __name__ == "__main__":
    main()
