"""Baseline-тренировка YOLO12 на DENTEX (4 класса патологий).

Запуск:
    py research/scripts/train_yolo.py [--model yolo12x.pt] [--epochs 100] [--imgsz 1536]

Результаты: research/output/runs/<name>/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "dentex_yolo" / "dentex.yaml"
RUNS_DIR = ROOT / "output" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12x.pt", help="веса/модель ultralytics")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1536, help="высокое разрешение — кариес мелкий")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default=None)
    # Отдельный набор для ДЕТЕКЦИИ ЗУБОВ (32 класса FDI) — это первый этап
    # двухэтапной схемы, у него свои данные и свой конфиг.
    parser.add_argument("--data", default=None,
                        help="путь к yaml; по умолчанию набор с патологиями")
    args = parser.parse_args()

    name = args.name or f"{Path(args.model).stem}_imgsz{args.imgsz}"

    model = YOLO(args.model)
    model.train(
        data=str(args.data or DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(RUNS_DIR),
        name=name,
        patience=30,          # early stopping
        cos_lr=True,
        close_mosaic=15,      # мозаика мешает на мелких объектах в конце
        workers=8,
        seed=42,
    )

    # финальная валидация на лучших весах
    best = RUNS_DIR / name / "weights" / "best.pt"
    if best.exists():
        YOLO(str(best)).val(data=str(args.data or DATA_YAML), imgsz=args.imgsz, project=str(RUNS_DIR), name=f"{name}_val")


if __name__ == "__main__":
    main()
