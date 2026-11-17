"""Преп-разметка снимков клиники (model-assisted labeling).

Вход: research/example/2d_images/*.jpg — отчётные распечатки Dentium
(панорама в шаблоне с шапкой/подвалом). Рамку снимка находит
`report_crop` — то же правило, что у сервера разметки.

Что делает:
1. Кропает рентген-область из шаблона -> research/datasets/clinic_preannotated/images/
2. Прогоняет ансамбль (YOLO12x + YOLO26x-tuned, WBF) -> YOLO labels (.txt)
   в research/datasets/clinic_preannotated/labels/ (для правки врачом в CVAT)
3. Аннотированные превью -> research/output/clinic_preview/

Запуск:
    py research/scripts/preannotate_clinic.py [--conf 0.25]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from eval_ensemble import wbf  # noqa: E402
from report_crop import crop_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "example/2d_images"
OUT_IMG = ROOT / "datasets/clinic_preannotated/images"
OUT_LBL = ROOT / "datasets/clinic_preannotated/labels"
OUT_PREV = ROOT / "output/clinic_preview"

MODELS = [
    ROOT / "output/runs/yolo12x_imgsz960/weights/best.pt",
    ROOT / "output/runs/yolo26x_tuned/weights/best.pt",
]
CLASSES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
COLORS = [(255, 59, 48), (255, 204, 0), (255, 69, 58), (255, 159, 10)]  # BGR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)
    OUT_PREV.mkdir(parents=True, exist_ok=True)

    models = [YOLO(str(w)) for w in MODELS]
    files = sorted(SRC.glob("*.jpg"))
    print(f"снимков: {len(files)}")

    for f in files:
        img = cv2.imread(str(f))
        crop = crop_report(img).copy()
        H, W = crop.shape[:2]
        cv2.imwrite(str(OUT_IMG / f.name), crop)

        dets = []
        for m in models:
            res = m.predict(crop, imgsz=960, conf=0.05, max_det=100, verbose=False)[0]
            for xyxy, cf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                     res.boxes.conf.cpu().numpy(),
                                     res.boxes.cls.cpu().numpy().astype(int)):
                dets.append([*map(float, xyxy), float(cf), int(cls)])
        fused = [d for d in wbf(dets) if d[4] >= args.conf]

        lines = []
        for bx1, by1, bx2, by2, score, cls in fused:
            cx, cy = (bx1 + bx2) / 2 / W, (by1 + by2) / 2 / H
            bw, bh = (bx2 - bx1) / W, (by2 - by1) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.3f}")
            cv2.rectangle(crop, (int(bx1), int(by1)), (int(bx2), int(by2)), COLORS[cls], 3)
            cv2.putText(crop, f"{CLASSES[cls]} {score:.0%}", (int(bx1), int(by1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[cls], 2)
        (OUT_LBL / (f.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                 encoding="utf-8")
        cv2.imwrite(str(OUT_PREV / (f.stem + "_pred.png")), crop)
        counts = {}
        for d in fused:
            counts[CLASSES[d[5]]] = counts.get(CLASSES[d[5]], 0) + 1
        print(f"{f.name}: кроп {W}x{H}, находок {len(fused)} {counts if counts else ''}")


if __name__ == "__main__":
    main()
