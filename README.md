# Stomadent — AI-платформа для стоматологической диагностики

Аналог Diagnocat для рынка Узбекистана. AI-анализ стоматологических снимков (2D/3D), автоматическая генерация отчётов, 3D-моделирование.

## Текущий статус

**Фаза 1: Research** — поиск датасетов, exploration, baseline

### Что готово
- ✅ Архитектура проекта
- ✅ Анализ Diagnocat (полный функционал)
- ✅ Найден основной датасет: **DENTEX** (11.8 GB, 2332 annotated X-rays)
- ✅ Структура проекта создана
- ✅ Скрипты для скачивания и визуализации

### Следующие шаги
- [ ] Скачать DENTEX датасет
- [ ] Визуализировать данные + аннотации
- [ ] Inference pre-trained YOLOv8
- [ ] Training на DENTEX (нужен GPU)
- [ ] PDF-генератор отчётов

## Структура проекта

```
stomadent/
├── research/              # Эксперименты и исследования
│   ├── datasets/          # Скачанные датасеты
│   │   └── DENTEX/        # DENTEX (панорамные рентген)
│   ├── notebooks/         # Jupyter эксперименты
│   ├── scripts/           # Скрипты
│   │   ├── find_datasets.py       # Поиск датасетов
│   │   ├── download_dentex.py     # Скачать DENTEX
│   │   ├── visualize_dentex.py    # Визуализация аннотаций
│   │   └── yolo_inference.py      # YOLO inference
│   ├── output/            # Результаты экспериментов
│   └── requirements.txt   # Зависимости для research
├── src/                   # Production код (позже)
│   ├── models/            # ML модели
│   ├── pipeline/          # Пайплайн обработки
│   ├── dicom/             # DICOM процессор
│   ├── report/            # PDF генератор
│   └── config/            # Конфигурация
├── tests/                 # Тесты
├── requirements.txt       # Основные зависимости
├── ARCHITECTURE.md        # Архитектура системы
├── DIAGNOCAT_ANALYSIS.md  # Анализ Diagnocat
└── README.md
```

## Датасеты

### Основной: DENTEX
- **Панорамные рентген снимки (OPG)**
- 2332 размеченных + 1571 неразмеченных
- 4 класса: кариес, глубокий кариес, периапикальные, ретенированные зубы
- COCO JSON формат аннотаций
- [HuggingFace](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)

### Альтернативный: DFL Periapical
- ~6000 периапикальных снимков
- YOLO format bounding boxes
- [Kaggle](https://www.kaggle.com/datasets/kredy20/periapical-teeth-detection)

## Быстрый старт

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Скачать DENTEX
```bash
pip install huggingface_hub
py research/scripts/download_dentex.py --download
```

### Визуализировать датасет
```bash
py research/scripts/visualize_dentex.py --stats --samples 3
```

### Inference (pre-trained)
```bash
py research/scripts/yolo_inference.py --image path/to/xray.png
```

## Модель

**Текущая:** YOLOv8n (pre-trained на COCO) — baseline inference
**Целевая:** YOLOv8 fine-tuned на DENTEX — detection 4 патологий + зубы

| Задача | Архитектура | Статус |
|--------|-------------|--------|
| 2D Detection (кариес) | YOLOv8 | 🔄 Fine-tune pending |
| 3D Segmentation (КЛКТ) | nnU-Net / MONAI | ⏳ Later |
| PDF Report | ReportLab | ⏳ Pending |

## Оборудование

- **CPU:** Intel Core Ultra 9
- **RAM:** 128 GB
- **GPU:** NVIDIA RTX 5090
- **Training:** Локальное обучение с высокой эффективностью
