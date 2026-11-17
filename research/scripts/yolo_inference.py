"""
Inference pre-trained YOLOv8 на стоматологических снимках.
Запуск: py research/scripts/yolo_inference.py

Требует: pip install ultralytics Pillow
Пока использует pre-trained COCO модель — позже заменим на fine-tuned DENTEX.
"""

from pathlib import Path
from PIL import Image

from ultralytics import YOLO


def run_inference(image_path, model_name="yolov8n.pt", save_dir=None):
    """Запустить inference на изображении."""
    print(f"📷 Загрузка изображения: {image_path}")
    print(f"🤖 Модель: {model_name}")

    # Загрузить модель
    model = YOLO(model_name)

    # Запустить inference
    results = model(image_path, conf=0.25)

    # Показать результат
    result_img = results[0].plot()

    # Сохранить
    if save_dir:
        save_path = Path(save_dir) / f"result_{Path(image_path).name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_img).save(save_path)
        print(f"✅ Результат сохранён: {save_path}")

    # Вывести детекции
    boxes = results[0].boxes
    if len(boxes) > 0:
        print(f"\n🔍 Найдено {len(boxes)} объектов:")
        for box in boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"   {cls_name:15s} {conf:.2%}")
    else:
        print("\n⚠️  Объекты не найдены (pre-trained на COCO, не на dental)")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 inference на снимках")
    parser.add_argument("--image", type=str, default=None,
                        help="Путь к изображению")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO модель (yolov8n.pt, yolov8s.pt, etc)")
    parser.add_argument("--save-dir", type=str,
                        default="./research/output/predictions",
                        help="Директория для сохранения")
    args = parser.parse_args()

    if args.image:
        run_inference(args.image, args.model, args.save_dir)
    else:
        print("❌ Укажите изображение: --image path/to/image.png")
        print()
        print("Примеры:")
        print("  py yolo_inference.py --image research/datasets/DENTEX/panorama_001.png")
        print("  py yolo_inference.py --image my_xray.jpg")


if __name__ == "__main__":
    main()
