"""
Визуализация датасета DENTEX — показывает снимки с bounding box аннотациями.
Запуск: py research/scripts/visualize_dentex.py

Требует: pip install huggingface_hub matplotlib Pillow
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# Цвета для классов
CLASS_COLORS = {
    "Caries": "#FF0000",           # Красный — кариес
    "Deep Caries": "#CC0000",      # Тёмно-красный — глубокий кариес
    "Periapical Lesion": "#FF8C00",# Оранжевый — периапикальное поражение
    "Impacted Tooth": "#8B4513",   # Коричневый — ретенированный зуб
    "Tooth": "#0000FF",            # Синий — зуб (FDI нумерация)
    "Quadrant": "#00FF00",         # Зелёный — квадрант
}


def load_coco_annotations(annotations_path):
    """Загрузить COCO аннотации."""
    with open(annotations_path, "r") as f:
        return json.load(f)


def get_category_map(coco_data):
    """Создать маппинг category_id -> class_name."""
    return {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}


def draw_annotations(image_path, annotations, category_map, max_boxes=None):
    """Нарисовать bounding boxes на изображении."""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(16, 10))
    ax.imshow(img, cmap="gray")

    # Фильтровать аннотации для этого изображения
    img_id = None
    for ann_img in [a for a in annotations if "image_id" in a]:
        # Найти image_id по filename
        pass

    # Проще: показать все boxes если файл совпадает
    boxes_drawn = 0
    for ann in annotations:
        if max_boxes and boxes_drawn >= max_boxes:
            break

        bbox = ann.get("bbox", [])
        if not bbox:
            continue

        x, y, w, h = bbox
        cat_id = ann.get("category_id", 0)
        class_name = category_map.get(cat_id, f"Unknown ({cat_id})")

        color = CLASS_COLORS.get(class_name, "#FFFFFF")

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        # Подпись
        ax.text(
            x, y - 5,
            class_name,
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )
        boxes_drawn += 1

    ax.set_title(f"DENTEX: {Path(image_path).name}\n{boxes_drawn} annotations",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    return fig


def show_dataset_stats(coco_data):
    """Показать статистику датасета."""
    category_map = get_category_map(coco_data)
    annotations = coco_data.get("annotations", [])
    images = coco_data.get("images", [])

    # Подсчёт по классам
    class_counts = {}
    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        class_name = category_map.get(cat_id, f"Unknown ({cat_id})")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("=" * 60)
    print("DENTEX DATASET STATS")
    print("=" * 60)
    print(f"Всего изображений: {len(images)}")
    print(f"Всего аннотаций:   {len(annotations)}")
    print()
    print("Классы:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        color = CLASS_COLORS.get(cls, "⬜")
        print(f"  {color} {cls:25s} {count:5d}")
    print("=" * 60)


def visualize_random_samples(coco_data, images_dir, n_samples=3):
    """Показать случайные примеры с аннотациями."""
    annotations = coco_data.get("annotations", [])
    images = coco_data.get("images", [])
    category_map = get_category_map(coco_data)

    if not images:
        print("❌ Нет изображений в аннотациях")
        return

    # Найти файлы изображений
    img_files = list(Path(images_dir).glob("*.png")) + \
                list(Path(images_dir).glob("*.jpg"))

    if not img_files:
        print(f"❌ Нет изображений в {images_dir}")
        return

    # Выбрать случайные
    samples = random.sample(img_files, min(n_samples, len(img_files)))

    fig, axes = plt.subplots(1, len(samples), figsize=(8 * len(samples), 10))
    if len(samples) == 1:
        axes = [axes]

    for idx, img_path in enumerate(samples):
        img = Image.open(img_path)
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(img_path.name, fontsize=12, fontweight="bold")
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = Path(__file__).parent.parent / "output" / "dentex_samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"✅ Примеры сохранены: {output_path}")
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Визуализация DENTEX датасета")
    parser.add_argument("--stats", action="store_true", help="Показать статистику")
    parser.add_argument("--samples", type=int, default=0, help="Показать N примеров")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Путь к директории DENTEX")
    args = parser.parse_args()

    # Найти датасет
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = Path(__file__).parent.parent / "datasets" / "DENTEX" / "DENTEX"

    if not dataset_dir.exists():
        print(f"❌ Датасет не найден: {dataset_dir}")
        print("Сначала скачайте: py research/scripts/download_dentex.py --download")
        return

    # Найти аннотации
    annot_files = list(dataset_dir.glob("**/*.json"))
    if not annot_files:
        print("❌ COCO аннотации не найдены")
        return

    annot_path = annot_files[0]
    print(f"📂 Аннотации: {annot_path}")

    coco_data = load_coco_annotations(annot_path)

    if args.stats:
        show_dataset_stats(coco_data)

    if args.samples > 0:
        # Найти директорию с изображениями
        images_dir = dataset_dir.parent / "DENTEX"  # или где лежат PNG
        if not images_dir.exists():
            # Попробовать найти
            for d in dataset_dir.parent.iterdir():
                if d.is_dir() and any(d.glob("*.png")):
                    images_dir = d
                    break

        visualize_random_samples(coco_data, images_dir, args.samples)


if __name__ == "__main__":
    main()
