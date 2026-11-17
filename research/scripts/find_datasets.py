"""
Поиск доступных датасетов для стоматологического AI.
Запуск: py research/scripts/find_datasets.py
"""

import json

# Открытые датасеты для стоматологического AI
DATASETS = {
    "2D_intraoral": [
        {
            "name": "Dental Panoramic X-ray Dataset",
            "url": "https://www.kaggle.com/datasets/sumaiyaafroz/dental-panoramic-x-ray-dataset",
            "format": "JPG/PNG",
            "description": "Панорамные рентген снимки с аннотациями патологий",
            "size": "~2000 изображений",
            "license": "CC BY-NC 4.0",
        },
        {
            "name": "TUHH Dental MRI Dataset",
            "url": "https://www.ipi.uni-hannover.de/en/research/projects/tuhh-dental-data-set",
            "format": "DICOM/MRI",
            "description": "МРТ зубов с segmentation masks",
            "size": "~50 субъектов",
            "license": "Research",
        },
        {
            "name": "Caries Detection Dataset (Kaggle)",
            "url": "https://www.kaggle.com/datasets/walterhamayel/caries-detection",
            "format": "JPG",
            "description": "Внутриротовые снимки: кариес/здоровые зубы",
            "size": "~3000 изображений",
            "license": "CC",
        },
    ],
    "2D_bitewing_periapical": [
        {
            "name": "Boğaziçi Periapical X-ray Dataset",
            "url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVY/CXQZQG",
            "format": "JPG/PNG",
            "description": "Периапикальные снимки с разметкой патологий",
            "size": "~1000+ снимков",
            "license": "CC0",
        },
        {
            "name": "DFL Periapical Teeth Dataset",
            "url": "https://www.kaggle.com/datasets/kredy20/periapical-teeth-detection",
            "format": "JPG + YOLO labels",
            "description": "Периапикальные снимки, YOLO-разметка зубов",
            "size": "~6000 снимков",
            "license": "CC",
        },
    ],
    "3D_CBCT": [
        {
            "name": "MICCAI Dental Implants Challenge",
            "url": "https://www.synapse.org/Synapse:syn51158470",
            "format": "DICOM",
            "description": "CBCT с имплантами, segmentation challenge",
            "size": "Challenge dataset",
            "license": "Research",
        },
        {
            "name": "MIDAS Dental CBCT",
            "url": "https://www.midassociety.org/",
            "format": "DICOM",
            "description": "Различные CBCT датасеты",
            "size": "Varies",
            "license": "Varies",
        },
        {
            "name": "3D CBCT Tooth Segmentation (ISBI)",
            "url": "https://dbconnect.grand-challenge.org/",
            "format": "DICOM",
            "description": "CBCT для сегментации зубов",
            "size": "Challenge dataset",
            "license": "Research",
        },
    ],
    "cephalometric": [
        {
            "name": "ISBI Cephalometric Landmark Detection",
            "url": "https://www.synapse.org/#!Synapse:syn22416692",
            "format": "PNG",
            "description": "Цефалометрические снимки с landmarks",
            "size": "~150 снимков",
            "license": "Challenge",
        },
    ],
}


def print_datasets():
    """Вывести все найденные датасеты."""
    print("=" * 70)
    print("ДОСТУПНЫЕ ДАТАСЕТЫ ДЛЯ СТОМАТОЛОГИЧЕСКОГО AI")
    print("=" * 70)
    
    for category, datasets in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"КАТЕГОРИЯ: {category}")
        print(f"{'='*70}")
        
        for i, ds in enumerate(datasets, 1):
            print(f"\n  [{i}] {ds['name']}")
            print(f"      URL: {ds['url']}")
            print(f"      Формат: {ds['format']}")
            print(f"      Размер: {ds['size']}")
            print(f"      Лицензия: {ds['license']}")
            print(f"      Описание: {ds['description']}")
    
    print(f"\n{'='*70}")
    print(f"ИТОГО: {sum(len(v) for v in DATASETS.values())} датасетов")
    print(f"{'='*70}")


if __name__ == "__main__":
    print_datasets()
