# 2D Датасеты для стоматологического AI

## 🏆 DENTEX — Основной выбор для baseline

**URL:** https://huggingface.co/datasets/ibrahimhamamci/DENTEX
**GitHub:** https://github.com/ibrahimethemhamamci/DentEx

| Параметр | Значение |
|----------|----------|
| Тип снимков | Панорамные рентген (OPG) |
| Формат | PNG/JSON (COCO-формат) |
| Общий размер | 11.8 GB |
| Размеченные изображения | 2332 |
| Неразмеченные | 1571 |
| Лицензия | CC BY-NC-SA 4.0 (данные), MIT (код) |
| Источник | 3 мед. учреждения, пациенты >12 лет |

### Структура разметки

| Поднабор | Кол-во | Тип разметки |
|----------|--------|--------------|
| Quadrant-only | 693 | Только квадранты |
| Quadrant + Enumeration | 634 | Квадранты + нумерация зубов (FDI) |
| **Fully Labeled** | **1005** | Квадранты + нумерация + **диагноз** |

### Классы патологий (fully labeled)
1. **Caries** (кариес)
2. **Deep Caries** (глубокий кариес)
3. **Periapical Lesion** (периапикальное поражение)
4. **Impacted Tooth** (ретенированный зуб)

### Train/Val/Test
| Split | Кол-во | Ground truth |
|-------|--------|--------------|
| Train | 705 | ✅ Да |
| Val | 50 | ❌ Без GT (для challenge) |
| Test | 250 | ❌ Скрыт |

### Как скачать
```bash
pip install huggingface_hub
huggingface-cli download ibrahimhamamci/DENTEX --repo-type dataset --local-dir ./research/datasets/DENTEX
```

Или через Python:
```python
from huggingface_hub import snapshot_download
snapshot_download("ibrahimhamamci/DENTEX", repo_type="dataset", local_dir="./research/datasets/DENTEX")
```

---

## Альтернативные датасеты

### 2. sudhakark4227/dental-xray-dataset (HuggingFace)
**URL:** https://huggingface.co/datasets/sudhakark4227/dental-xray-dataset
| Параметр | Значение |
|----------|----------|
| Размер | 2.29 GB |
| Формат | imagefolder (JPG/PNG) |
| Лицензия | MIT |
| Аннотации | Неизвестно (README пуст) |

### 3. sudhakark4227/dental-xray-segmentation (HuggingFace)
**URL:** https://huggingface.co/datasets/sudhakark4227/dental-xray-segmentation
| Параметр | Значение |
|----------|----------|
| Тип | Сегментация зубов |
| Формат | imagefolder |

### 4. reza362/dental-xray-caries (HuggingFace)
**URL:** https://huggingface.co/datasets/reza362/dental-xray-caries
| Параметр | Значение |
|----------|----------|
| Размер | 4.64 MB |
| Изображений | 20 (очень маленький) |
| Тип | Кариес |

### 5. Датасет из dentex-yolo11-xray (GitHub)
**URL:** https://github.com/SafalNarshing/dentex-yolo11-xray
- Использует DENTEX
- Готовый YOLO11n-seg код для сегментации
- Классы: Caries, Deep Caries, Periapical Lesion, Impacted Tooth
- COCO JSON формат аннотаций

### 6. Dental-xray-YOLOv11-detection (GitHub)
**URL:** https://github.com/KhaledEldfry/Dental-xray-YOLOv11-detection
- YOLOv11 для обнаружения зубных компонентов
- Custom annotated dataset
- Keras + Ultralytics

### 7. DFL Periapical Teeth Dataset (Kaggle)
**URL:** https://www.kaggle.com/datasets/kredy20/periapical-teeth-detection
| Параметр | Значение |
|----------|----------|
| Тип | Периапикальные снимки |
| Формат | JPG + YOLO labels |
| Размер | ~6000 снимков |
| Аннотации | YOLO format (bounding boxes) |

---

## Рекомендация по порядку работы

1. **DENTEX** → основной датасет (панорама + патологии + нумерация)
2. **DFL Periapical** → для 2D periradical моделей (отдельно от панорамы)
3. **DENTEX unlabeled (1571)** → для self-supervised pretraining
