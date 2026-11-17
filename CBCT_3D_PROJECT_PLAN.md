# STOMADENT 3D CBCT — ДЕТАЛЬНЫЙ ПЛАН ПРОЕКТА

**Дата:** 15 августа 2026  
**Цель:** Создание AI-системы для анализа 3D CBCT снимков с демонстрацией врачам клиники в Нукусе

---

## 📋 EXECUTIVE SUMMARY

### Что мы делаем:
Создаем систему автоматической сегментации 3D CBCT снимков зубов с определением:
- **32 зуба** (FDI нумерация: 11-18, 21-28, 31-38, 41-48)
- **Нижнечелюстной нерв** (Inferior Alveolar Nerve/Canal)
- **Верхнечелюстные пазухи** (Maxillary Sinuses)
- **Корневые каналы** (Root Canals)
- **Патологии**: периапикальные поражения, ретенированные зубы, близость к нерву

### Почему это важно:
- **Для хирурга**: планирование имплантации (расстояние до нерва, плотность кости)
- **Для ортопеда**: планирование протезирования
- **Для ортодонта**: оценка положения ретенированных зубов
- **Для терапевта**: 3D визуализация корневых каналов

### Технологический стек:
- **Архитектура**: nnU-Net (state-of-the-art для медицинской сегментации)
- **Фреймворк**: MONAI (PyTorch-based, специально для медицинских изображений)
- **Датасеты**: ToothFairy2 (443 CBCT), CTooth+ (22+146), Teeth3DS (1800 IOS)
- **Hardware**: RTX 5090 (24GB VRAM), 128GB RAM, Intel Ultra 9

---

## 🎯 ЭТАПЫ ПРОЕКТА

### **ЭТАП 1: ПОДГОТОВКА ИНФРАСТРУКТУРЫ** [1-2 дня]

#### 1.1 Установка библиотек
```bash
# Базовые библиотеки
pip install monai[all] nnunetv2 SimpleITK nibabel pydicom

# Визуализация
pip install itkwidgets vtk pyvista napari

# Дополнительно
pip install pytorch-lightning wandb tensorboard
```

#### 1.2 Структура проекта
```
stomadent/
├── research_3d/
│   ├── datasets/
│   │   ├── toothfairy2/          # 443 CBCT (нерв)
│   │   ├── ctooth/               # 22 CBCT (зубы)
│   │   ├── teeth3ds/             # 1800 IOS (интраоральные)
│   │   └── clinic_nukus/         # Реальные данные клиники
│   ├── preprocessing/
│   │   ├── dicom_loader.py
│   │   ├── resampling.py
│   │   └── intensity_normalization.py
│   ├── training/
│   │   ├── nnunet_train.py
│   │   ├── monai_auto3dseg.py
│   │   └── configs/
│   ├── inference/
│   │   ├── predict.py
│   │   └── postprocessing.py
│   ├── visualization/
│   │   ├── 3d_viewer.py
│   │   └── slice_viewer.py
│   └── evaluation/
│       ├── metrics.py
│       └── clinical_validation.py
```

---

### **ЭТАП 2: ЗАГРУЗКА И АНАЛИЗ ДАТАСЕТОВ** [2-3 дня]

#### 2.1 ToothFairy2 Challenge Dataset
**Источник**: https://toothfairy.grand-challenge.org/  
**Описание**: Largest public CBCT dataset for inferior alveolar canal segmentation  
**Размер**: 443 CBCT scans (153 с полными аннотациями, 290 sparse labels)  
**Классы**: 
- Inferior Alveolar Canal (IAN/нижнечелюстной нерв) - основной класс
- Background

**Формат**: NIfTI (.nii.gz)  
**Spacing**: различный (требуется resampling к 1×1×1 мм³)  
**Размер сканов**: ~512×512×(200-400) slices

**Как скачать**:
```bash
# Регистрация на https://toothfairy.grand-challenge.org/
# Скачивание через браузер или CLI
wget --user=<username> --password=<password> <dataset_url>
```

**Статистика** (из статьи):
- Mean Dice: 0.9253 (winner nnU-Net)
- HD95: 18.472 мм
- Сложность: тонкая извилистая структура нерва

**Применение**: 
- Планирование имплантации (критично избежать повреждения нерва)
- Хирургия нижней челюсти

---

#### 2.2 CTooth+ Dataset
**Источник**: https://github.com/liangjiubujiu/CTooth  
**Описание**: Large-scale CBCT dataset for tooth volume segmentation  
**Размер**: 22 fully annotated + 146 unlabeled volumes (5803 slices total)  
**Классы**: 
- 32 зуба по FDI системе (11-48)
- Background

**Формат**: DICOM series или NIfTI  
**Особенности**: 
- Missing teeth (отсутствующие зубы)
- Dental restorations (пломбы, коронки)
- Implants (импланты)
- High variance (разные виды патологий)

**Статистика**:
- 4243 slices с аннотациями зубов
- Средний размер: ~400×400×300

**Как получить**:
```bash
# Заполнить форму на GitHub: https://github.com/liangjiubujiu/CTooth
# Ждать одобрения (обычно 1-3 дня)
# Скачать по ссылке из email
```

**Применение**:
- Instance segmentation каждого зуба
- FDI numbering (автоматическая нумерация)
- Планирование ортодонтии

---

#### 2.3 Teeth3DS+ Dataset
**Источник**: https://osf.io/xctdy/  
**Описание**: Intraoral 3D scans (mesh format, не CBCT!)  
**Размер**: 1800 scans (23,999 annotated teeth + 30,000 landmarks)  
**Формат**: STL/OBJ mesh files  
**Классы**: 32 зуба + landmarks (Mesial/Distal/Cusps)

**Применение**:
- Интраоральное сканирование (следующий этап проекта)
- Alignment с CBCT (superimposition)
- Digital treatment planning

**Примечание**: Этот датасет для mesh segmentation, не для volumetric CBCT

---

#### 2.4 3DTeethSeg22 Challenge
**Источник**: https://3dteethseg.grand-challenge.org/  
**То же что Teeth3DS** — challenge на основе этого датасета

---

#### 2.5 STS 2024 Challenge Dataset
**Источник**: https://arxiv.org/abs/2407.13246  
**Описание**: Semi-supervised tooth segmentation (2D panoramic + 3D CBCT)  
**Размер**: 
- 2,380 panoramic X-rays (OPG)
- 330 CBCT scans
- 90,000+ 2D images/slices

**Формат**: PNG (2D) + NIfTI (3D)  
**Особенность**: Semi-supervised learning (только часть данных размечена)

**Применение**:
- Комбинированный подход 2D+3D
- Наш текущий 2D модуль может интегрироваться

---

### **ЭТАП 3: PREPROCESSING PIPELINE** [3-4 дня]

#### 3.1 Загрузка DICOM серий
```python
import SimpleITK as sitk
import numpy as np
from pathlib import Path

def load_cbct_scan(dicom_folder: Path) -> tuple:
    """Загрузка DICOM серии в 3D volume."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Получаем метаданные
    volume = sitk.GetArrayFromImage(image)  # (D, H, W)
    spacing = image.GetSpacing()  # (x, y, z) в мм
    origin = image.GetOrigin()
    direction = image.GetDirection()
    
    return volume, spacing, origin, direction, image
```

**Проблемы и решения**:
- **Различные spacing**: ToothFairy (0.4×0.4×0.4 мм), CTooth (0.3×0.3×0.5 мм), клиника (?)
- **Решение**: Resample все к единому spacing (например 1×1×1 мм или 0.5×0.5×0.5 мм)

---

#### 3.2 Resampling (изменение разрешения)
```python
def resample_volume(image: sitk.Image, new_spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    """Resampling к целевому spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Вычисляем новый размер
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkBSpline)  # Для изображений
    # Для масок: sitk.sitkNearestNeighbor
    
    return resampler.Execute(image)
```

**Важно**: 
- Для изображений: BSpline/Linear interpolation
- Для масок/labels: NearestNeighbor (чтобы не создавать промежуточные классы)

---

#### 3.3 Intensity Normalization (нормализация HU)
```python
def normalize_hu(volume: np.ndarray, 
                 window_center=400, 
                 window_width=1800) -> np.ndarray:
    """Нормализация Hounsfield Units для зубов."""
    # CBCT HU ranges (приблизительно):
    # Air: -1000
    # Soft tissue: -300 to 100
    # Bone: 300 to 2000
    # Enamel (эмаль): 1500 to 3000
    # Metal (импланты): 3000+
    
    # Windowing
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    
    volume_clipped = np.clip(volume, lower, upper)
    volume_normalized = (volume_clipped - lower) / (upper - lower)
    
    return volume_normalized.astype(np.float32)
```

**Параметры windowing**:
- **Bone window**: center=400, width=1800 (для зубов и костей)
- **Soft tissue**: center=40, width=350 (для нерва)
- nnU-Net автоматически определит оптимальные параметры

---

#### 3.4 Cropping ROI (Region of Interest)
```python
def crop_to_roi(volume: np.ndarray, 
                mask: np.ndarray = None,
                margin: int = 20) -> tuple:
    """Кроп до области интереса (челюсти)."""
    if mask is None:
        # Автоматический поиск ROI по интенсивности
        threshold = volume > -500  # Убираем воздух
    else:
        threshold = mask > 0
    
    # Находим bounding box
    coords = np.argwhere(threshold)
    if len(coords) == 0:
        return volume, (0, 0, 0), volume.shape
    
    z0, y0, x0 = coords.min(axis=0) - margin
    z1, y1, x1 = coords.max(axis=0) + margin
    
    # Clamp к границам
    z0, y0, x0 = max(0, z0), max(0, y0), max(0, x0)
    z1 = min(volume.shape[0], z1)
    y1 = min(volume.shape[1], y1)
    x1 = min(volume.shape[2], x1)
    
    cropped = volume[z0:z1, y0:y1, x0:x1]
    
    return cropped, (z0, y0, x0), (z1, y1, x1)
```

**Зачем**: 
- Уменьшает размер данных (меньше VRAM)
- Убирает пустое пространство вокруг челюсти

---

#### 3.5 Конвертация в nnU-Net формат
nnU-Net требует специфическую структуру:
```
nnUNet_raw/
└── Dataset001_ToothFairy/
    ├── dataset.json
    ├── imagesTr/
    │   ├── case_0000_0000.nii.gz
    │   ├── case_0001_0000.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── case_0000.nii.gz
    │   ├── case_0001.nii.gz
    │   └── ...
    └── imagesTs/  (test set, без labels)
```

**dataset.json**:
```json
{
  "channel_names": {
    "0": "CT"
  },
  "labels": {
    "background": 0,
    "inferior_alveolar_canal": 1
  },
  "numTraining": 153,
  "file_ending": ".nii.gz"
}
```

**Скрипт конвертации**:
```python
def convert_to_nnunet_format(
    source_dir: Path,
    output_dir: Path,
    dataset_id: int = 1,
    dataset_name: str = "ToothFairy"
):
    """Конвертация датасета в формат nnU-Net."""
    # Создаем структуру
    dataset_dir = output_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    
    # Копируем/конвертируем файлы с правильными именами
    # case_{id:04d}_0000.nii.gz для изображений
    # case_{id:04d}.nii.gz для масок
```

---

### **ЭТАП 4: ОБУЧЕНИЕ nnU-Net** [5-7 дней]

#### 4.1 Установка nnU-Net v2
```bash
pip install nnunetv2

# Настройка переменных окружения
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

**Структура директорий**:
```
D:/code/dbc/stomadent/research_3d/
├── nnUNet_raw/          # Исходные данные
├── nnUNet_preprocessed/ # После preprocessing
└── nnUNet_results/      # Обученные модели
```

---

#### 4.2 Планирование и препроцессинг
```bash
# nnU-Net автоматически анализирует датасет и создает план
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

**Что делает**:
1. Анализирует spacing, размеры, интенсивности
2. Определяет оптимальные параметры:
   - Patch size (размер окна для обучения)
   - Batch size (в зависимости от VRAM)
   - Normalization strategy
   - Augmentation pipeline
3. Создает 5-fold cross-validation splits
4. Препроцессит данные (resample, crop, normalize)

**Ожидаемое время**: 
- ToothFairy (443 scans): ~2-3 часа на RTX 5090
- CTooth (22 scans): ~30 минут

---

#### 4.3 Обучение моделей
nnU-Net обучает 3 конфигурации:

##### **4.3.1 3D Full Resolution**
```bash
nnUNetv2_train 1 3d_fullres 0 --npz
```
- Полное разрешение 3D
- Лучше для мелких структур (нерв)
- Требует много VRAM (~16-20 GB)

##### **4.3.2 3D Low Resolution**
```bash
nnUNetv2_train 1 3d_lowres 0 --npz
```
- Пониженное разрешение
- Быстрее обучается
- Хорошо для крупных структур (зубы)

##### **4.3.3 3D Cascade**
```bash
# Сначала lowres
nnUNetv2_train 1 3d_lowres all --npz

# Потом fullres с предсказаниями lowres
nnUNetv2_train 1 3d_cascade_fullres all --npz
```
- Двухэтапный подход
- Best performance обычно
- Сначала грубая сегментация → потом уточнение

**Параметры**:
- `1` — dataset ID
- `0` — fold номер (0-4 для cross-validation)
- `all` — обучить на всех 5 folds
- `--npz` — сохранять softmax predictions

**Время обучения** (на RTX 5090):
- 3d_fullres: ~8-12 часов (1000 epochs)
- 3d_lowres: ~4-6 часов
- 3d_cascade: ~12-18 часов (lowres + fullres)

---

#### 4.4 Мониторинг обучения
```bash
# TensorBoard (автоматически создается)
tensorboard --logdir=nnUNet_results/Dataset001_ToothFairy/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0
```

**Метрики для отслеживания**:
- **Dice Score** (основная): >0.85 для нерва, >0.90 для зубов
- **HD95** (Hausdorff Distance 95%): <20 мм хорошо
- **Validation loss**: должен снижаться

**Ранняя остановка**: nnU-Net автоматически сохраняет лучший checkpoint

---

#### 4.5 Ансамбль моделей
```bash
# После обучения всех конфигураций, создаем ансамбль
nnUNetv2_find_best_configuration 1 -c 3d_fullres 3d_lowres 3d_cascade_fullres
```

**Что делает**:
- Комбинирует предсказания всех моделей
- Выбирает оптимальные веса для каждой
- Обычно дает прирост ~1-2% Dice

---

### **ЭТАП 5: INFERENCE НА НОВЫХ ДАННЫХ** [2-3 дня]

#### 5.1 Предсказание на тестовых данных
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 1 -c 3d_fullres -f 0 1 2 3 4
```

**Параметры**:
- `-i` — папка с DICOM или NIfTI файлами
- `-o` — папка для результатов (NIfTI masks)
- `-f 0 1 2 3 4` — использовать все 5 folds (ансамбль)

**Формат входных данных**:
```
INPUT_FOLDER/
├── case001_0000.nii.gz
├── case002_0000.nii.gz
└── ...
```

**Формат выходных данных**:
```
OUTPUT_FOLDER/
├── case001.nii.gz  (маска сегментации)
├── case002.nii.gz
└── ...
```

---

#### 5.2 Inference на DICOM из клиники
```python
def predict_on_clinic_dicom(dicom_folder: Path) -> np.ndarray:
    """Pipeline для реальных DICOM из клиники."""
    # 1. Загрузка
    volume, spacing, origin, direction, sitk_image = load_cbct_scan(dicom_folder)
    
    # 2. Resample к spacing модели
    resampled = resample_volume(sitk_image, new_spacing=(1.0, 1.0, 1.0))
    
    # 3. Сохраняем временный NIfTI
    temp_dir = Path("temp_inference")
    temp_dir.mkdir(exist_ok=True)
    sitk.WriteImage(resampled, str(temp_dir / "patient_0000.nii.gz"))
    
    # 4. Запускаем nnU-Net
    os.system(f"nnUNetv2_predict -i {temp_dir} -o {temp_dir}/output -d 1 -c 3d_fullres -f all")
    
    # 5. Загружаем результат
    mask = sitk.ReadImage(str(temp_dir / "output/patient.nii.gz"))
    mask_array = sitk.GetArrayFromImage(mask)
    
    # 6. Resample обратно к оригинальному spacing
    mask_original = resample_volume(mask, original_spacing)
    
    return sitk.GetArrayFromImage(mask_original)
```

---

#### 5.3 Post-processing
```python
def postprocess_segmentation(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Удаление мелких артефактов."""
    from scipy.ndimage import label, find_objects
    
    # Для каждого класса
    output = np.zeros_like(mask)
    for class_id in range(1, mask.max() + 1):
        binary = (mask == class_id)
        labeled, num_features = label(binary)
        
        # Удаляем компоненты меньше min_size вокселей
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if component.sum() >= min_size:
                output[component] = class_id
    
    return output
```

---

### **ЭТАП 6: ВИЗУАЛИЗАЦИЯ ДЛЯ ВРАЧЕЙ** [3-4 дня]

#### 6.1 Веб-интерфейс (FastAPI + Three.js)

##### Backend: FastAPI
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import zipfile

app = FastAPI(title="Stomadent 3D CBCT Viewer")

@app.post("/api/upload_dicom")
async def upload_dicom(files: list[UploadFile] = File(...)):
    """Загрузка DICOM серии."""
    # Сохраняем во временную папку
    temp_dir = Path(tempfile.mkdtemp())
    for file in files:
        content = await file.read()
        (temp_dir / file.filename).write_bytes(content)
    
    # Запускаем inference
    mask = predict_on_clinic_dicom(temp_dir)
    
    # Конвертируем в JSON для фронтенда
    # ... (см. ниже)
    
    return JSONResponse({"status": "success", "session_id": str(temp_dir.name)})

@app.get("/api/segmentation/{session_id}")
async def get_segmentation(session_id: str):
    """Получить результаты сегментации."""
    # Возвращаем маску в формате для 3D рендеринга
    pass
```

##### Frontend: Three.js для 3D визуализации
```html
<!DOCTYPE html>
<html>
<head>
    <title>Stomadent 3D Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="viewer3d"></div>
    <div id="controls">
        <button onclick="toggleStructure('nerve')">Нерв</button>
        <button onclick="toggleStructure('teeth')">Зубы</button>
        <button onclick="toggleStructure('sinus')">Пазухи</button>
        <label>Прозрачность: <input type="range" id="opacity" min="0" max="100" value="80"></label>
    </div>
    
    <script>
        // Three.js сцена
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        
        // Загрузка mesh из маски сегментации
        function loadSegmentationMesh(maskData) {
            // Marching Cubes алгоритм для создания mesh
            // ... (используем библиотеку типа vtk.js)
        }
        
        // Цвета для каждой структуры
        const colors = {
            nerve: 0xff4444,      // красный
            teeth: 0xffffff,      // белый
            sinus: 0x4444ff,      // синий
        };
    </script>
</body>
</html>
```

---

#### 6.2 Альтернатива: 3D Slicer (для врачей-экспертов)

**3D Slicer** — бесплатная профессиональная программа для медицинской визуализации.

##### Установка:
```bash
# Скачать с https://download.slicer.org/
# Установить MONAI Auto3DSeg extension
```

##### Workflow:
1. Открыть DICOM серию: `File → Add DICOM Data`
2. Загрузить маску сегментации (NIfTI)
3. `Segment Editor` → визуализация 3D
4. Измерения расстояний, объемов
5. Экспорт STL для 3D печати

**Преимущества**:
- Профессиональный инструмент
- Много функций (measurements, annotations)
- MPR views (Axial, Coronal, Sagittal)

**Недостатки**:
- Требует обучения
- Не веб-интерфейс

---

#### 6.3 Простая визуализация: Matplotlib + Napari

Для быстрого прототипа:

```python
import napari
import numpy as np

def visualize_segmentation(volume: np.ndarray, mask: np.ndarray):
    """Интерактивный 3D viewer."""
    viewer = napari.Viewer()
    
    # Добавляем оригинальный снимок
    viewer.add_image(volume, name='CBCT', colormap='gray', opacity=0.5)
    
    # Добавляем маску по слоям
    colors = {
        1: 'red',      # Нерв
        2: 'blue',     # Зуб 11
        3: 'blue',     # Зуб 12
        # ... до 32
    }
    
    viewer.add_labels(mask, name='Segmentation')
    
    napari.run()

# Использование
volume, _, _, _, _ = load_cbct_scan("path/to/dicom")
mask = predict_on_clinic_dicom("path/to/dicom")
visualize_segmentation(volume, mask)
```

---

#### 6.4 Генерация отчета

```python
def generate_report(volume: np.ndarray, 
                   mask: np.ndarray, 
                   patient_info: dict) -> dict:
    """Генерация клинического отчета."""
    report = {
        "patient_id": patient_info["id"],
        "scan_date": patient_info["date"],
        "findings": []
    }
    
    # 1. Подсчет зубов
    tooth_ids = range(11, 49)  # FDI система
    present_teeth = []
    missing_teeth = []
    
    for tooth_id in tooth_ids:
        if (mask == tooth_id).sum() > 0:
            present_teeth.append(tooth_id)
        else:
            missing_teeth.append(tooth_id)
    
    report["findings"].append({
        "type": "tooth_count",
        "present": len(present_teeth),
        "missing": missing_teeth
    })
    
    # 2. Анализ нерва
    nerve_mask = (mask == 1)  # класс 1 = нижнечелюстной нерв
    if nerve_mask.sum() > 0:
        # Проверка близости к зубам
        for tooth_id in present_teeth:
            if tooth_id >= 30:  # Нижняя челюсть
                tooth_mask = (mask == tooth_id)
                distance = compute_min_distance(tooth_mask, nerve_mask)
                
                if distance < 2.0:  # меньше 2 мм
                    report["findings"].append({
                        "type": "nerve_proximity",
                        "tooth": tooth_id,
                        "distance_mm": distance,
                        "warning": "КРИТИЧНО: близость к нерву"
                    })
    
    # 3. Ретенированные зубы
    # (определяем по положению относительно соседних)
    
    # 4. Объем каждого зуба
    for tooth_id in present_teeth:
        tooth_mask = (mask == tooth_id)
        volume_mm3 = tooth_mask.sum() * np.prod(spacing)
        
        report["findings"].append({
            "type": "tooth_volume",
            "tooth": tooth_id,
            "volume_mm3": volume_mm3
        })
    
    return report

def compute_min_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Минимальное расстояние между двумя структурами."""
    from scipy.ndimage import distance_transform_edt
    
    # Distance transform на маске2
    dist = distance_transform_edt(~mask2)
    
    # Минимальное расстояние от mask1
    return dist[mask1].min()
```

---

### **ЭТАП 7: ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ** [2-3 дня]

#### 7.1 Подготовка тестового набора
```
clinic_nukus/
├── patient_001/  (DICOM серия из Rainbow viewer)
├── patient_002/
├── patient_003/
└── ...
```

#### 7.2 Запуск inference
```bash
# Конвертируем DICOM → NIfTI
python research_3d/preprocessing/dicom_to_nifti.py --input clinic_nukus --output clinic_nukus_nifti

# Предсказание
nnUNetv2_predict -i clinic_nukus_nifti -o clinic_nukus_results -d 1 -c 3d_fullres -f all

# Визуализация
python research_3d/visualization/create_3d_renders.py --input clinic_nukus_results
```

#### 7.3 Валидация врачом
**Чек-лист для врача**:
- [ ] Все зубы корректно сегментированы?
- [ ] FDI нумерация правильная?
- [ ] Нерв найден и выделен полностью?
- [ ] Нет ложных срабатываний (false positives)?
- [ ] Границы структур точные?

**Метрики**:
- **Sensitivity (чувствительность)**: найдены ли все зубы?
- **Specificity (специфичность)**: нет ли ложных зубов?
- **Clinical accuracy**: можно ли использовать для планирования имплантации?

---

### **ЭТАП 8: ОПТИМИЗАЦИЯ И ДЕПЛОЙ** [2-3 дня]

#### 8.1 Оптимизация inference скорости
```python
# TensorRT оптимизация для RTX 5090
# Ускоряет inference в 2-3 раза

from onnx import load
import tensorrt as trt

def convert_onnx_to_tensorrt(onnx_path: str, engine_path: str):
    """Конвертация nnU-Net в TensorRT."""
    # Экспорт PyTorch → ONNX
    # ONNX → TensorRT
    pass
```

**Результат**: 
- Без оптимизации: ~30-60 секунд на один CBCT
- С TensorRT: ~10-20 секунд

---

#### 8.2 Упаковка в Docker
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Установка зависимостей
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install torch torchvision nnunetv2 monai fastapi uvicorn

# Копируем веса моделей
COPY nnUNet_results /app/models
COPY research_3d /app/research_3d

WORKDIR /app
EXPOSE 8000

CMD ["uvicorn", "research_3d.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t stomadent-3d:latest .

# Run
docker run --gpus all -p 8000:8000 stomadent-3d:latest
```

---

#### 8.3 API Endpoints
```python
# research_3d/api/server.py

@app.post("/api/v1/segment_cbct")
async def segment_cbct(files: list[UploadFile]):
    """
    Загрузка DICOM серии и автоматическая сегментация.
    
    Returns:
    - segmentation_mask: NIfTI file
    - report: JSON с находками
    - 3d_render: PNG preview
    """
    pass

@app.get("/api/v1/download_stl/{session_id}/{structure}")
async def download_stl(session_id: str, structure: str):
    """
    Скачать STL файл для 3D печати.
    
    structure: "nerve" | "tooth_11" | "tooth_12" | ...
    """
    pass

@app.get("/api/v1/measurements/{session_id}")
async def get_measurements(session_id: str):
    """
    Измерения: объемы зубов, расстояния до нерва, и т.д.
    """
    pass
```

---

#### 8.4 Интеграция с Rainbow Viewer?
**Вопрос**: Можем ли мы интегрироваться с Rainbow 3D Viewer?

**Варианты**:
1. **Plugin для Rainbow** (если API доступен)
2. **Экспорт результатов** в формате, который Rainbow может импортировать
3. **Standalone viewer** (наш веб-интерфейс)

**Исследовать**: 
- Rainbow SDK/API документация
- Форматы экспорта (DICOM-SEG, DICOM-RT-STRUCT)

---

### **ЭТАП 9: СЛЕДУЮЩИЕ ШАГИ** [после демо]

#### 9.1 Расширение на другие структуры

**Приоритет 1: Корневые каналы**
- Датасет: искать в литературе (пока нет public dataset)
- Архитектура: та же nnU-Net
- Применение: эндодонтия (планирование лечения каналов)

**Приоритет 2: Верхнечелюстные пазухи**
- Датасет: часть ToothFairy или отдельный sinus dataset
- Применение: синус-лифтинг при имплантации

**Приоритет 3: Плотность кости (Hounsfield Units анализ)**
- Не ML задача — просто анализ HU в области имплантации
- Классификация: D1 (>1250 HU), D2 (850-1250), D3 (350-850), D4 (<350)

---

#### 9.2 Детекция патологий на CBCT

**Задачи**:
- Кариес (сложно на CBCT из-за разрешения)
- Периапикальные поражения (вокруг корня зуба)
- Кисты
- Резорбция корня

**Датасеты**: 
- Публичных мало
- Нужна разметка от радиологов клиники
- Можно использовать weakly-supervised learning

---

#### 9.3 Интраоральное сканирование (IOS)

**Датасет**: Teeth3DS+ (1800 mesh scans)

**Задачи**:
- Mesh segmentation (PointNet++, MeshCNN)
- Tooth labeling (FDI)
- Landmark detection (cusps, contact points)
- Occlusion analysis (прикус)

**Применение**:
- Ортодонтия (планирование брекетов)
- Протезирование (digital impression)
- Alignment с CBCT (суперимпозиция)

**Архитектура**: 
- PointNet++ для point clouds
- Graph Neural Networks для mesh

---

### **ЭТАП 10: ДОПОЛНИТЕЛЬНЫЕ ФИЧИ** [опционально]

#### 10.1 FDI Auto-numbering (автоматическая нумерация зубов)

Алгоритм:
1. Определяем midline (средняя линия) по симметрии
2. Определяем верхняя/нижняя челюсть по оси Z
3. Сортируем зубы слева направо
4. Присваиваем FDI номера: 11-18 (верх право), 21-28 (верх лево), ...

```python
def assign_fdi_numbers(teeth_masks: dict) -> dict:
    """Автоматическая FDI нумерация."""
    # Находим центроиды всех зубов
    centroids = {}
    for tooth_id, mask in teeth_masks.items():
        z, y, x = np.where(mask)
        centroids[tooth_id] = (z.mean(), y.mean(), x.mean())
    
    # Определяем midline (x coordinate)
    all_x = [c[2] for c in centroids.values()]
    midline_x = np.median(all_x)
    
    # Разделяем на квадранты
    quadrants = {1: [], 2: [], 3: [], 4: []}
    for tooth_id, (z, y, x) in centroids.items():
        if z < z_threshold:  # Верхняя челюсть
            if x > midline_x:
                quadrants[1].append((tooth_id, x))  # Право пациента
            else:
                quadrants[2].append((tooth_id, x))  # Лево пациента
        else:  # Нижняя челюсть
            if x > midline_x:
                quadrants[4].append((tooth_id, x))
            else:
                quadrants[3].append((tooth_id, x))
    
    # Сортируем и присваиваем FDI
    fdi_map = {}
    for quad, teeth in quadrants.items():
        if quad in [1, 4]:  # Право → лево
            sorted_teeth = sorted(teeth, key=lambda t: -t[1])
        else:  # Лево → право
            sorted_teeth = sorted(teeth, key=lambda t: t[1])
        
        for i, (tooth_id, _) in enumerate(sorted_teeth, start=1):
            fdi_map[tooth_id] = quad * 10 + i
    
    return fdi_map
```

---

#### 10.2 Implant Planning (планирование имплантации)

**Входные данные**:
- Сегментация зубов
- Сегментация нерва
- HU карта (плотность кости)

**Выходные данные**:
- Предлагаемые позиции имплантов
- Размеры имплантов (диаметр, длина)
- Угол установки
- Расстояние до нерва (safety margin)

```python
def plan_implant(missing_tooth_position: tuple,
                 nerve_mask: np.ndarray,
                 bone_density: np.ndarray,
                 spacing: tuple) -> dict:
    """Планирование одного импланта."""
    # 1. Проверка плотности кости
    roi = bone_density[...]  # ROI вокруг позиции
    mean_hu = roi.mean()
    bone_quality = classify_bone_density(mean_hu)
    
    # 2. Безопасное расстояние до нерва
    distance_to_nerve = compute_distance(missing_tooth_position, nerve_mask)
    
    # 3. Доступная высота кости
    available_height = compute_bone_height(...)
    
    # 4. Рекомендации
    recommendations = {
        "implant_length": min(available_height - 2, 13),  # -2мм safety
        "implant_diameter": 4.0,  # стандартный
        "bone_quality": bone_quality,
        "distance_to_nerve_mm": distance_to_nerve,
        "warnings": []
    }
    
    if distance_to_nerve < 2.0:
        recommendations["warnings"].append("CRITICAL: Too close to nerve")
    
    if bone_quality == "D4":
        recommendations["warnings"].append("Low bone density - consider augmentation")
    
    return recommendations
```

---

#### 10.3 Export to CAD/CAM системам

**Форматы**:
- **STL** — для 3D печати
- **PLY** — для mesh editing
- **OBJ** — универсальный формат
- **DICOM-SEG** — стандарт для медицинских сегментаций

```python
def export_to_stl(mask: np.ndarray, 
                  spacing: tuple, 
                  output_path: str):
    """Экспорт маски в STL для 3D печати."""
    from skimage import measure
    import trimesh
    
    # Marching cubes для создания mesh
    verts, faces, normals, values = measure.marching_cubes(
        mask, 
        level=0.5, 
        spacing=spacing
    )
    
    # Создаем mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Сглаживание
    mesh = mesh.smoothed()
    
    # Экспорт
    mesh.export(output_path)
```

---

## 📊 TIMELINE И ОЦЕНКА РЕСУРСОВ

### Временные затраты (с RTX 5090):

| Этап | Действие | Время |
|------|----------|-------|
| 1 | Установка библиотек | 1-2 часа |
| 2 | Загрузка датасетов | 4-6 часов (зависит от интернета) |
| 3 | Preprocessing | 3-4 часа |
| 4 | Обучение nnU-Net | 12-18 часов (automatic) |
| 5 | Inference на тестах | 1-2 часа |
| 6 | Разработка viewer | 2-3 дня |
| 7 | Тестирование с врачом | 1-2 дня |
| 8 | Оптимизация + Docker | 1-2 дня |
| **ИТОГО** | **~7-10 дней** | **для MVP** |

### Дисковое пространство:

- ToothFairy2: ~80 GB (443 CBCT)
- CTooth+: ~5 GB (22 CBCT)
- Teeth3DS: ~10 GB (1800 IOS mesh)
- nnU-Net preprocessed: ~100 GB
- Обученные модели: ~5 GB
- **ИТОГО**: ~200 GB

### VRAM требования:

- Training: 16-20 GB (3d_fullres)
- Inference: 8-12 GB
- **RTX 5090 (24GB)**: отлично подходит ✅

---

## 🎓 ОБУЧЕНИЕ И ДОКУМЕНТАЦИЯ

### Для разработчика:

1. **nnU-Net документация**: https://github.com/MIC-DKFZ/nnUNet
2. **MONAI Tutorial**: https://github.com/Project-MONAI/tutorials
3. **Medical Image Processing**: https://simpleitk.readthedocs.io/

### Для врача:

1. **Интерпретация результатов** (создадим инструкцию)
2. **Как проверить качество сегментации**
3. **Измерения в 3D Slicer**

---

## ⚠️ РИСКИ И ОГРАНИЧЕНИЯ

### Технические риски:

1. **Различие в качестве CBCT сканеров**
   - Разные производители (Planmeca, Carestream, i-CAT, ...)
   - Различные настройки (FOV, voxel size, экспозиция)
   - **Решение**: Fine-tuning на данных из конкретной клиники

2. **Артефакты от металлических имплантов**
   - Beam hardening artifacts
   - Затрудняют сегментацию соседних зубов
   - **Решение**: Post-processing, metal artifact reduction algorithms

3. **Низкое качество CBCT для мягких тканей**
   - Нерв иногда плохо виден
   - **Решение**: Ансамбль моделей, использование контекста

4. **Overfitting на публичных датасетах**
   - Модель хорошо работает на ToothFairy, но плохо на реальных данных клиники
   - **Решение**: Domain adaptation, transfer learning

### Клинические ограничения:

1. **Не заменяет радиолога**
   - Результаты требуют проверки врачом
   - Это ассистент, а не диагност

2. **Юридическая ответственность**
   - В случае ошибки кто несет ответственность?
   - **Решение**: Disclaimer, врач принимает финальное решение

3. **Необходима валидация**
   - Нужны сравнительные исследования (AI vs врач)
   - Минимум 100-200 случаев

---

## 📈 МЕТРИКИ УСПЕХА

### Технические метрики:

| Метрика | Target | Benchmark (ToothFairy winner) |
|---------|--------|-------------------------------|
| Dice Score (нерв) | >0.85 | 0.925 |
| HD95 (нерв) | <25 мм | 18.5 мм |
| Dice Score (зубы) | >0.90 | - |
| Inference time | <30 сек | - |
| False positives | <5% | - |

### Клинические метрики:

- **Sensitivity**: найдены ли все критичные структуры? (>95%)
- **Specificity**: нет ли ложных тревог? (>90%)
- **Clinical utility**: может ли врач использовать для планирования? (оценка врачом)
- **Time saving**: на сколько ускоряет работу врача? (target: 50% времени)

---

## 🔄 ИТЕРАЦИОННАЯ РАЗРАБОТКА

### MVP (Minimum Viable Product) — 10 дней:
- ✅ Сегментация нижнечелюстного нерва
- ✅ Сегментация 32 зубов
- ✅ Базовая визуализация (Napari)
- ✅ Тест на 3-5 реальных случаях

### Version 1.0 — +2 недели:
- ✅ FDI auto-numbering
- ✅ Веб-интерфейс (FastAPI + Three.js)
- ✅ Генерация отчетов
- ✅ Export STL
- ✅ Интеграция с клиникой

### Version 2.0 — +1 месяц:
- ✅ Корневые каналы
- ✅ Верхнечелюстные пазухи
- ✅ Плотность кости (HU анализ)
- ✅ Implant planning
- ✅ Детекция патологий

### Version 3.0 — +2 месяца:
- ✅ Интраоральное сканирование (IOS)
- ✅ Superimposition (CBCT + IOS)
- ✅ Orthodontic planning
- ✅ Digital treatment plan

---

## 🛠️ ПРАКТИЧЕСКИЕ СОВЕТЫ

### Для обучения:

1. **Не обучайте с нуля** — используйте pretrained weights от nnU-Net
2. **Начните с малого** — сначала только нерв (ToothFairy), потом зубы (CTooth)
3. **Используйте cross-validation** — 5-fold для надежности
4. **Мониторьте overfitting** — следите за validation loss
5. **Augmentation критична** — rotation, scaling, elastic deformation

### Для inference:

1. **Тестируйте на разных CBCT** — разные производители, настройки
2. **Post-processing обязателен** — удаление мелких артефактов
3. **Uncertainty estimation** — если модель не уверена, покажите врачу
4. **Batch inference** — обрабатывайте несколько сканов параллельно

### Для врачей:

1. **Обучающая сессия** — 1-2 часа демонстрации
2. **Gradual adoption** — начинайте с простых случаев
3. **Feedback loop** — собирайте ошибки, переобучайте модель
4. **Integration** — встройте в существующий workflow

---

## 📚 ИСТОЧНИКИ И ДАТАСЕТЫ

### Papers (из твоего документа):

1. **ToothFairy Challenge** (2024)  
   https://arxiv.org/abs/2308.16375  
   443 CBCT, inferior alveolar canal

2. **CTooth+** (2024, Nature Scientific Data)  
   https://arxiv.org/abs/2208.01643  
   22+146 CBCT, tooth segmentation

3. **Scaling nnU-Net for CBCT** (2024)  
   https://arxiv.org/abs/2411.17213  
   Winner ToothFairy2, Dice 0.9253

4. **STS 2024 Challenge** (MICCAI)  
   https://arxiv.org/abs/2407.13246  
   Semi-supervised tooth segmentation

5. **Teeth3DS+** (2022)  
   https://arxiv.org/abs/2210.06094  
   1800 intraoral scans, 23,999 teeth

### Датасеты для скачивания:

| Название | Ссылка | Размер | Регистрация |
|----------|--------|--------|-------------|
| ToothFairy2 | https://toothfairy.grand-challenge.org/ | 80 GB | ✅ Нужна |
| CTooth+ | https://github.com/liangjiubujiu/CTooth | 5 GB | ✅ Нужна форма |
| Teeth3DS+ | https://osf.io/xctdy/ | 10 GB | ✅ Открытый |
| STS 2024 | https://tianchi.aliyun.com/... | 90 GB | ✅ Нужна |

### Frameworks:

- **nnU-Net**: https://github.com/MIC-DKFZ/nnUNet
- **MONAI**: https://github.com/Project-MONAI/MONAI
- **PyTorch**: https://pytorch.org/

---

## 🚀 NEXT STEPS (IMMEDIATE)

### День 1-2: Установка и подготовка
- [ ] Установить nnU-Net, MONAI, SimpleITK
- [ ] Настроить окружение и пути
- [ ] Зарегистрироваться на ToothFairy, CTooth
- [ ] Скачать датасеты (параллельно)

### День 3-4: Preprocessing
- [ ] Изучить структуру ToothFairy
- [ ] Конвертировать в nnU-Net формат
- [ ] Запустить `nnUNetv2_plan_and_preprocess`
- [ ] Проверить preprocessing результаты

### День 5-7: Training
- [ ] Запустить обучение 3d_fullres
- [ ] Мониторить TensorBoard
- [ ] После 1000 epochs — оценить результаты

### День 8: Inference на реальных данных
- [ ] Загрузить DICOM из клиники
- [ ] Запустить inference
- [ ] Визуализировать в Napari
- [ ] Сравнить с Rainbow viewer

### День 9-10: Демо для врача
- [ ] Подготовить презентацию
- [ ] Показать 3-5 случаев
- [ ] Собрать feedback
- [ ] Записать что нужно улучшить

---

## 💡 ИДЕИ ДЛЯ БУДУЩЕГО

### Аксиография (отдельный проект):
- **Проблема**: Нет публичных датасетов
- **Решение**: 
  1. Начать с rule-based анализа (чтение параметров из P-Art)
  2. Собрать свой датасет (100+ пациентов с МРТ ground truth)
  3. ML модель (1D-CNN/LSTM для временных рядов)

### Multi-modal fusion:
- Комбинация CBCT + Panoramic (OPG) + Intraoral scans
- Лучше accuracy, cross-validation между модальностями

### Longitudinal analysis:
- Сравнение CBCT снимков до/после лечения
- Отслеживание прогресса (рост кисты, заживление)

### Federated learning:
- Обучение на данных нескольких клиник без передачи данных
- Privacy-preserving ML

---

## 📞 КОНТАКТЫ И ПОДДЕРЖКА

### Если застрянешь:

1. **nnU-Net GitHub Issues**: https://github.com/MIC-DKFZ/nnUNet/issues
2. **MONAI Discussions**: https://github.com/Project-MONAI/MONAI/discussions
3. **Medical Imaging ML Community**: https://grand-challenge.org/forums/
4. **3D Slicer Forum**: https://discourse.slicer.org/

---

## ✅ ФИНАЛЬНЫЙ ЧЕКЛИСТ

### Перед демо врачу:
- [ ] Модель обучена и валидирована (Dice >0.85)
- [ ] Inference работает на реальных DICOM
- [ ] Визуализация понятна (цветовая кодировка, подписи)
- [ ] Измерения корректны (расстояния, объемы)
- [ ] Отчет генерируется автоматически
- [ ] Подготовлены 5-10 демо случаев
- [ ] Есть сравнение: без AI vs с AI (time savings)

### Вопросы для врача:
1. Насколько точна сегментация? (визуально)
2. Можете ли использовать для планирования имплантации?
3. Что не хватает? Какие еще структуры нужны?
4. Удобен ли интерфейс?
5. Сколько времени экономит?

---

**ДОКУМЕНТ СОЗДАН:** 15 августа 2026  
**АВТОР:** Claude (Opus 5)  
**ПРОЕКТ:** Stomadent 3D CBCT Segmentation  
**ВЕРСИЯ:** 1.0

**Готов начинать? Let's build this! 🚀**
