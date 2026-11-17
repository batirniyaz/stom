"""
Скачивание датасета DENTEX для dental X-ray анализа.
Запуск: py research/scripts/download_dentex.py

Требует: pip install huggingface_hub
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

DATASET_NAME = "ibrahimhamamci/DENTEX"
LOCAL_DIR = Path(__file__).parent.parent / "datasets" / "DENTEX"


def check_huggingface():
    """Проверить установлен ли huggingface_hub."""
    try:
        import huggingface_hub
        print(f"✅ huggingface_hub установлен: v{huggingface_hub.__version__}")
        return True
    except ImportError:
        print("❌ huggingface_hub не установлен")
        print("Установите: pip install huggingface_hub")
        return False


def download_dentex():
    """Скачать датасет DENTEX."""
    from huggingface_hub import snapshot_download

    print(f"\n📥 Скачивание датасета DENTEX...")
    print(f"   Источник: {DATASET_NAME}")
    print(f"   Локальная директория: {LOCAL_DIR}")
    print(f"   Размер: ~11.8 GB")
    print()

    if LOCAL_DIR.exists():
        print(f"⚠️  Директория {LOCAL_DIR} уже существует")
        response = input("Перезаписать? (y/n): ").strip().lower()
        if response != "y":
            print("Отмена скачивания.")
            return

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Начинаю загрузку... (это может занять время)")
    print("-" * 50)

    try:
        path = snapshot_download(
            DATASET_NAME,
            repo_type="dataset",
            local_dir=str(LOCAL_DIR),
            resume_download=True,
        )
        print("-" * 50)
        print(f"✅ Датасет скачан в: {path}")
        print(f"\nСтруктура:")
        for item in LOCAL_DIR.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  📄 {item.name} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        print("\nВозможные причины:")
        print("  - Нет доступа к интернету")
        print("  - Требуется авторизация (huggingface-cli login)")
        print("  - Недостаточно места на диске")
        sys.exit(1)


def show_info():
    """Показать информацию о датасете."""
    print("=" * 60)
    print("DENTEX — Dental Panoramic X-ray Dataset")
    print("=" * 60)
    print()
    print("Тип:           Панорамные рентген снимки (OPG)")
    print("Формат:        PNG + COCO JSON аннотации")
    print("Размечено:     2332 изображения")
    print("Неразмечено:   1571 изображение")
    print("Общий размер:  11.8 GB")
    print("Лицензия:      CC BY-NC-SA 4.0")
    print()
    print("Классы патологий:")
    print("  1. Caries (кариес)")
    print("  2. Deep Caries (глубокий кариес)")
    print("  3. Periapical Lesion (периапикальное поражение)")
    print("  4. Impacted Tooth (ретенированный зуб)")
    print()
    print("Train: 705 | Val: 50 | Test: 250")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Скачать датасет DENTEX")
    parser.add_argument("--info", action="store_true", help="Показать информацию")
    parser.add_argument("--download", action="store_true", help="Скачать датасет")
    args = parser.parse_args()

    if args.info:
        show_info()
    elif args.download:
        if check_huggingface():
            download_dentex()
    else:
        show_info()
        print()
        print("Использование:")
        print("  py download_dentex.py --info       # Информация")
        print("  py download_dentex.py --download   # Скачать")
