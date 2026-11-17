"""Тренировка RF-DETR (large) на DENTEX для сравнения с YOLO12.

Запуск:
    py research/scripts/train_rfdetr.py [--epochs 100] [--batch 4]

Датасет: research/datasets/dentex_coco (COCO-формат, см. convert_dentex_coco.py)
Результаты: research/output/runs/rfdetr_large/
"""

import argparse
from pathlib import Path

from rfdetr import RFDETRLarge

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "dentex_coco"
OUT_DIR = ROOT / "output" / "runs" / "rfdetr_large"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4, help="эффективный batch = batch * grad_accum")
    parser.add_argument("--resolution", type=int, default=784, help="делится на 56; выше = лучше для мелких объектов")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    model = RFDETRLarge(resolution=args.resolution)
    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        output_dir=str(OUT_DIR),
        early_stopping=True,
        early_stopping_patience=20,
        tensorboard=True,
    )


if __name__ == "__main__":
    main()
