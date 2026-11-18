# Stomadent — Архитектура AI-ядра

## Фаза 1: Research (сейчас)
Цель: найти датасеты, обучить baseline-модели, сравнить подходы.

### Датасеты для поиска
| Тип | Формат | Что ищем | Потенциальные источники |
|-----|--------|----------|------------------------|
| Внутриротовые (periradical) | 2D JPG/PNG | Кариес, периапикальные поражения | Kaggle, DentalPan, TUHH |
| ОПТГ (панорамные) | 2D JPG/PNG/DICOM | Все зубы, патологии, кариес | Kaggle "Dental Panoramic", UofL |
| КЛКТ (CBCT) | 3D DICOM | Сегментация зубов, каналов, костей | CBCT datasets, MIDAS, MICCAI |
| Цефалометрия | 2D X-ray | Landmarks для ортодонтии | ISBI challenges |

### Модели для экспериментов
| Задача | Кандидаты |
|--------|-----------|
| 2D Detection (кариес) | YOLOv8, Faster R-CNN, RetinaNet |
| 2D Classification (периапикальные) | ResNet50, EfficientNet, ConvNeXt |
| 3D Segmentation | nnU-Net, MONAI, 3D U-Net |
| Panoram from CBCT | VTK + pydicom (не ML) |

## Фаза 2: Production Pipeline
Цель: рабочий пайплайн DICOM → AI → PDF отчёт.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  DICOM      │────▶│  Preprocess  │────▶│  AI Model    │────▶│  Report  │
│  (КТ/ОПТГ)  │     │  (resize,    │     │  (detection/ │     │  (PDF)   │
│  2D/3D      │     │   normalize) │     │   segment)   │     │          │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────┘
                           │                     │
                      ┌────▼─────┐          ┌────▼─────┐
                      │  MinIO   │          │  MinIO   │
                      │  storage │          │  results │
                      └──────────┘          └──────────┘
```

## Фаза 3: Web Platform (потом)
- FastAPI backend
- DICOM viewer (Cornerstone.js)
- 3D viewer (Three.js/VTK.js)
- User management
- Billing

## Стек

### ML/AI
- Python 3.11
- PyTorch + MONAI (3D medical imaging)
- Ultralytics YOLOv8 (2D detection)
- pydicom + SimpleITK (DICOM processing)
- ReportLab / WeasyPrint (PDF generation)

### Инфраструктура
- MinIO (S3 storage)
- MLRun / Prefect (pipeline orchestration)
- FastAPI (API, позже)

## Структура проекта
```
stomadent/
├── research/
│   ├── datasets/           # Скачанные датасеты
│   ├── notebooks/          # Jupyter эксперименты
│   ├── scripts/            # Скрипты: download, train, eval
│   └── output/             # Результаты: модели, метрики, предикты
├── src/
│   ├── models/             # Обёртки моделей для inference
│   ├── pipeline/           # Пайплайн обработки
│   ├── dicom/              # DICOM процессор
│   ├── report/             # PDF генератор
│   └── config/             # Конфиги
├── tests/
├── requirements.txt
└── README.md
```
