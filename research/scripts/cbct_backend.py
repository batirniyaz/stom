"""КЛКТ (CBCT) бэкенд для демо Stomadent: загрузка DICOM, nnU-Net сегментация,
срезы с оверлеем, 3D-меши (GLB) для three.js-фронта.

Хранилище кейсов: research_3d/web_cases/<case_id>/
    volume.mha        — исходный объём (из DICOM-серии)
    mask_<model>.mha  — маска сегментации (ours | pretrained)
    mesh_<model>.glb  — сцена с именованными мешами групп структур
    meta.json         — метаданные кейса (переживают перезапуск сервера)

Готовые примеры client2-5 регистрируются из research_3d/inference_input|output
без копирования (в meta.json — абсолютные пути).

Инференс — однопроцессный nnUNetPredictor (как inference/predict_single.py):
multiprocessing nnU-Net на этом Windows ломается, CLI не использовать.
Модели загружаются один раз (фоновый warmup при старте) и переиспользуются.
"""

import json
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

R3D = Path(__file__).resolve().parents[2] / "research_3d"
CASES_DIR = R3D / "web_cases"

# env для nnunetv2 (проставляем до любого импорта nnunetv2)
os.environ.setdefault("nnUNet_raw", str(R3D / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(R3D / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(R3D / "nnUNet_results"))

MODELS = {
    "ours": R3D / "nnUNet_results/Dataset112_ToothFairy2"
            / "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "pretrained": R3D / "nnUNet_results/Dataset112_ToothFairy2"
                  / "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
    "tf3": R3D / "nnUNet_results/Dataset113_ToothFairy3"
           / "nnUNetTrainer__nnUNetPlans__3d_fullres",
    # Дообучение бенчмарочного ResEncL на ToothFairy3 (28.08): тот же датасет
    # с пульпой, но тяжёлая сеть и окно 112x224x256 вместо 64x160x160 — ради
    # зубов, которые лёгкая tf3 теряла целыми квадрантами.
    "tf3l": R3D / "nnUNet_results/Dataset113_ToothFairy3"
            / "nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans__3d_fullres",
    # Обучена 30.08 с тремя правками: пульпа ОДНИМ классом (47 вместо 78),
    # зеркалирование лево-право выключено, 829 эпох. Классы у неё свои —
    # см. SINGLE_PULP_MODELS и remap_prediction.
    "sp": R3D / "nnUNet_results/Dataset114_ToothFairy3_SinglePulp"
          / "nnUNetTrainer_onlyMirror01_1500ep__nnUNetResEncUNetLPlans__3d_fullres",
}
MODEL_LABELS = {"ours": "Наша (быстрая)", "pretrained": "Pretrained (точнее)",
                "tf3": "ToothFairy3 (с пульпой)",
                "tf3l": "ToothFairy3 L (дообученная)",
                "sp": "Единая (пульпа одним классом)",
                "combo": "Сводная (зубы + пульпа)"}

# Составная «модель»: своей сети у неё нет, она склеивает результаты двух.
# Зачем. Замер 28.08 на трёх кейсах показал, что лучшей модели не существует:
# `pretrained` уверенно ведёт по зубам (73% зубов одним куском против 27% у
# моделей с пульпой), но пульпы не знает вовсе — а без пульпы не считается
# эндодонтический статус. Дообучение ToothFairy3 эту разницу не закрыло:
# 100 эпох с правильной скоростью дали тренд -0.001 при разбросе 0.067, то
# есть плато, а куски «зубов» у неё лежат в 30 мм от самого зуба — это
# ложные метки, а не разорванный зуб.
COMBO_SOURCES = {"anatomy": "pretrained", "pulp": "tf3"}
# ПОЧЕМУ пазухи тоже приходится брать у pretrained, а не у ToothFairy3.
# Замер разметки (60 томов Dataset114): левая пазуха размечена в 42 томах с
# медианой 49 304 вокселя, правая в 40 с медианой 54 957 — при анатомической
# норме около 500 000 вокселей (15 см3) на пазуху. То есть в эталоне остаётся
# примерно ДЕСЯТАЯ ЧАСТЬ пазухи. Для сравнения, глотка размечена во всех 60
# томах целиком (780 224 вокселя) и выучивается на 0.97.
# Следствие: на обучении с нуля пазухи не оживают вовсе — 0.000 на протяжении
# 200 эпох, тогда как зубы дошли до 0.84, а пульпа до 0.87. Это ограничение
# РАЗМЕТКИ, и обучением оно не лечится.
# Допуск «пульпа внутри своего зуба» для СВОДНОЙ маски. Внутри одной модели
# порог 1.0 мм (PULP_MAX_DIST_MM) откалиброван замером: у здоровых пульп
# медиана 0.3-0.4 мм. Но здесь зуб и пульпа приходят от РАЗНЫХ моделей, и
# граница зуба у них смещена — замер на трёх кейсах показал пульпы с медианой
# 1.3-2.2 мм, которые по всем признакам свои, и отдельно мусор в 6-60 мм.
# 2.5 мм проходит между этими группами с запасом.
COMBO_PULP_DIST_MM = 2.5

# Модель Dataset113 обучена на перенумерованных подряд метках (0..77), потому
# что nnU-Net не принимает разреженную FDI-нумерацию. Возвращаем предсказание
# к оригинальным id, чтобы весь остальной код работал с номерами зубов.
LABEL_MAP_FILE = (R3D / "nnUNet_raw/Dataset113_ToothFairy3/label_map.json")
MODEL_NEEDS_REMAP = {"tf3", "tf3l"}
# Модели, обученные на Dataset114: 47 классов, вся пульпа под одним id.
# Номер зуба ей возвращается геометрией уже после инференса.
SINGLE_PULP_MODELS = {"sp"}

TEETH_CLASSES = list(range(11, 19)) + list(range(21, 29)) + \
                list(range(31, 39)) + list(range(41, 49))
# Пульпа кодируется как номер зуба + 100 (только модель tf3)
PULP_CLASSES = [t + 100 for t in TEETH_CLASSES]
MAX_CLASS = 149          # верхняя граница id для таблиц цвета

# группы структур: ключ -> классы, цвет, прозрачность по умолчанию
GROUPS = [
    # Челюсти РАЗДЕЛЬНО: врачу нужно гасить верхнюю, чтобы смотреть нижнюю, —
    # одной галочкой на обе это не делается. Модель их и так различает
    # (классы 1 и 2), объединяла их только наша группировка.
    {"key": "mandible",    "label": "Нижняя челюсть",
     "classes": [1], "color": "#e8e2d5", "opacity": 0.35},
    {"key": "maxilla",     "label": "Верхняя челюсть",
     "classes": [2], "color": "#d9d2c2", "opacity": 0.35},
    {"key": "teeth",       "label": "Зубы",
     "classes": TEETH_CLASSES, "color": "#f5f0e6", "opacity": 1.0},
    {"key": "nerve",       "label": "Нижний альвеолярный нерв",
     "classes": [3, 4], "color": "#ef4444", "opacity": 1.0},
    {"key": "sinus",       "label": "Гайморовы пазухи",
     "classes": [5, 6], "color": "#7dd3fc", "opacity": 0.5},
    {"key": "prosthetics", "label": "Коронки / мосты / импланты",
     "classes": [8, 9, 10], "color": "#cbd5e1", "opacity": 1.0},
    {"key": "pharynx",     "label": "Дыхательные пути",
     "classes": [7], "color": "#4ade80", "opacity": 0.25},
    # Резцовый и язычный каналы (ToothFairy3, классы 103-105). Модель их
    # предсказывает, а мы их ВЫБРАСЫВАЛИ: filter_mask собирает результат по
    # GROUPS, и класс, не попавший ни в одну группу, исчезает молча. Замер по
    # маскам: 103/104 — 573 и 768 вокселей у client4, 619 и 727 у client2.
    # Клинически это не мелочь: имплантат во фронтальном отделе нижней
    # челюсти ставится ровно там, где идёт резцовый канал.
    {"key": "incisive",    "label": "Резцовый канал",
     "classes": [103, 104], "color": "#fb923c", "opacity": 1.0},
    {"key": "lingual",     "label": "Язычный канал",
     "classes": [105], "color": "#fbbf24", "opacity": 1.0},
    # Пульпа: то, что врач называет «нерв внутри зуба». Есть только у модели
    # tf3; на моделях Dataset112 группа просто пустая и в UI не появляется.
    {"key": "pulp",        "label": "Пульпа и каналы зуба",
     "classes": PULP_CLASSES, "color": "#f43f5e", "opacity": 1.0},
]
GROUP_BY_KEY = {g["key"]: g for g in GROUPS}

# квартальные цвета зубов (FDI-квадрант = class // 10)
QUADRANT_COLORS = {1: "#93c5fd", 2: "#6ee7b7", 3: "#fcd34d", 4: "#f9a8d4"}

# лимиты треугольников на меш (decimate fast_simplification)
# Лимиты подняты ради вида: на 2500 треугольниках зуб выглядел гранёным
# камнем, а челюсть на 20000 рассыпалась в острые осколки по краю поля зрения.
# Размер GLB растёт примерно с 2.4 до 7 МБ — для локальной раздачи и туннеля
# это приемлемо, а врач первым делом смотрит именно на картинку.
FACE_CAP = {"mandible": 45000, "maxilla": 45000, "nerve": 14000,
            "sinus": 12000, "pharynx": 8000, "prosthetics": 8000,
            "tooth": 7000, "pulp": 40000, "incisive": 6000, "lingual": 4000}

# Сглаживание поверхности: размываем МАСКУ до marching cubes (см. _class_mesh).
# sigma в вокселях; pad — запас кропа под размытие (>= 4*sigma).
MESH_BLUR_SIGMA = 0.9
MESH_PAD = 3

HU_MIN, HU_MAX = -1000.0, 3000.0  # окно для срезов
OVERLAY_ALPHA = 0.45

# Версия постпроцессинга маски И построения меша. Меняем при изменении правил
# фильтрации или _class_mesh — при старте сервера устаревшие меши/инфо
# инвалидируются (см. _load_persisted).
POSTPROC_VERSION = 21   # 21: канал рисует модель (train_canal_net), трассировка — запасной путь

# Версия СЛОЯ НАХОДОК (cbct_findings и всё, что он тянет: cbct_perio, cbct_endo).
# Находки кэшируются в meta.json, и без версии врач после правки правил молча
# видел бы прошлый результат. Инкрементировать при изменении находок или их
# порогов; POSTPROC_VERSION для этого не годится — он про маску и меши.
FINDINGS_VERSION = 15   # 15: канал от модели — меняются длина канала и зазор до апекса

# HU-гейтинг классов (физический диапазон ткани) — ПЕРВЫЙ шаг filter_mask,
# до CC-фильтра. Ловит streak-воксели, СРАЩЕННЫЕ с настоящей анатомией
# (CC и форм-правила против них бессильны): модель метит яркие полосы
# металлоартефактов как челюсть/зубы/пазухи, но вне ткани их HU не тот.
# Отсев измерен на client2/client4/кейсе клиники: jaw 12-16%, teeth 0.2-1.9%,
# prosthetics <1%, pharynx <1% — настоящая анатомия не страдает.
HU_BONE_MIN = 300     # челюсти 1-2, зубы 11-48: только костная ткань
HU_PROSTH_MIN = 800   # коронки/мосты/импланты 8-10: плотные реставрации/металл
HU_AIR_MAX = -300     # пазухи 5-6, глотка 7: только воздушные полости
HU_GATE = {"mandible": ("min", HU_BONE_MIN), "maxilla": ("min", HU_BONE_MIN),
           "teeth": ("min", HU_BONE_MIN),
           "prosthetics": ("min", HU_PROSTH_MIN),
           "sinus": ("max", HU_AIR_MAX), "pharynx": ("max", HU_AIR_MAX)}
# нерв (3,4): гейт НЕ применяем — канал внутри кости, HU переменный

# Фильтрация связных компонент против металлоартефактов КЛКТ:
# streak-артефакты модель местами помечает как структуры — разрозненные
# воксели в полосах разлёта, marching cubes рисует их «шипами».
# Правила (минимум вокселей в компоненте, макс. число компонент на класс):
CC_RULES = {
    "large": (2000, 2),      # челюсти 1-2, пазухи 5-6, глотка 7
    "nerve": (500, 1),       # нерв 3-4: непрерывный, хвост-артефакт — мелкая CC
    "tooth": (200, 2),       # зубы: топ-2 — артефакт может «разрезать» зуб
    "prosthesis": (100, None),  # коронки/мосты/импланты: мелкие и их много
    # Пульпа тонкая: камера плюс каналы, объём на порядок меньше зуба.
    # Морфологическое opening, как у зубов, её бы просто съел.
    "pulp": (60, 3),
    # Резцовый и язычный каналы тоньше нижнечелюстного: у client5 левый
    # резцовый — 23 вокселя, правый 424. Порог 500, как у нерва, снёс бы
    # оба. 120 вокселей при вокселе 0.31 мм — это около 3.6 мм3, то есть
    # трубка диаметром ~1.2 мм длиной ~3 мм: короче настоящего канала.
    "canal_small": (120, 1),
}
GROUP_CC_RULE = {"mandible": "large", "maxilla": "large", "sinus": "large",
                 "pharynx": "large", "incisive": "canal_small",
                 "lingual": "canal_small",
                 "nerve": "nerve", "teeth": "tooth",
                 "prosthetics": "prosthesis", "pulp": "pulp"}

# Форм-правило против streak-компонент (крупные структуры + prosthetics):
# компонента отбраковывается, если fill_ratio < fill_min И aspect > aspect_max
# (тонкая линия в длинном bbox). На client2/client4 проверено: настоящие
# компоненты классов 1,2,5-10 имеют fill >= 0.04 и aspect <= 3.2 — под правило
# не попадает ни одна. Нерв и зубы форм-правилом НЕ трогаем.
SHAPE_RULES = {"fill_min": 0.05, "aspect_max": 6.0}
# Резцовый и язычный каналы под форм-правило НЕ попадают намеренно: они и
# есть тонкие вытянутые трубки, правило снесло бы их целиком — как и нерв.
SHAPE_RULE_GROUPS = {"mandible", "maxilla", "sinus", "pharynx", "prosthetics"}

# Зубы: streak-хвосты сращены с компонентой зуба, CC их не отделить.
# 1) морфологическое opening(2) срезает тонкие (1-4 вокс.) streak-перемычки;
# 2) дополнительная компонента класса принимается только если её центроид
#    близко к основной (разрезанный артефактом зуб, FDI-двойник через
#    среднюю линию — dist 26-63 вокс.), а дальний артефактный двойник
#    (dist 82-182 вокс.) отбрасывается. Порог подобран по client2.
TOOTH_OPENING_ITERS = 2
# Зазор между поверхностями, при котором второй кусок считается тем же зубом.
# Калибровка по эталону ToothFairy3: 790 зубов, из двух кусков состоят 2
# (0.3%), максимальный зазор у них 1.5 мм. Порог 2.0 покрывает эталон с
# запасом и совпадает с `COMPONENT_GAP_MM` трека сканов. Прежнее правило
# считало расстояние между ЦЕНТРОИДАМИ (70 вокселей = 22 мм) и пропускало
# куски, разнесённые с телом зуба на сантиметр.
TOOTH_GAP_MM = 2.0
# Зуб сидит в челюстной кости, и далеко от неё уехать не может. Порог
# откалиброван на эталоне ToothFairy3 (790 зубов, 30 томов): медиана
# расстояния до кости 2.0 мм, 99-й перцентиль 5.4, максимум 8.4 — дальше
# 12 мм НИ ОДНОГО. У наших кейсов настоящие зубы укладываются туда же
# (client2: самый дальний 48-й, 7.8 мм), а мусор стоит особняком: на client5
# кусок в 228 вокселей, помеченный зубом 28, висел в 21.9 мм от кости, за
# пределами челюсти. Его пульпа отпадает следом сама — правило «пульпа
# внутри своего зуба» не находит зуба.
#
# Реставрации меряются тем же порогом: на эталоне 31 компонента в 14 томах,
# максимум 7.3 мм, дальше 12 мм ни одной. На client5 настоящая пломба на
# режущем крае 11 стоит в 7.5 мм (коронка законно выступает из кости), а
# мусорный кусок в 126 вокселей — в 26.8 мм. Оговорка: эталонных компонент
# всего 31, поэтому запас здесь определён слабее, чем у зубов.
TOOTH_BONE_MAX_MM = 12.0
BONE_ANCHORED_GROUPS = ("teeth", "prosthetics")
TOOTH_MIN_VOX = 150        # минимум вокселей компонента после opening

# Крупные структуры (челюсти, пазухи, глотка): дополнительные компоненты
# принимаются только вблизи топ-1. Дальние — это плотные чужие объекты,
# ошибочно размеченные как челюсть (подголовник/стол КЛКТ — HU-гейт их не
# ловит, т.к. они реально плотные; кейс клиники: блобы 7-104k вокс. на
# dist 234-396 вокс. ≈ 73-123 мм от челюсти). Настоящие фрагменты
# анатомии (части верхней челюсти, полости носа) — в пределах ~30-60 вокс.
LARGE_MERGE_DIST = 100.0   # вокселей (~31 мм) от центроида топ-1

# Пульпа обязана лежать ВНУТРИ своего зуба (её номер = номер зуба + 100).
# Модель это правило иногда нарушает: на client4 пульпа 116 села в кость в
# 2.5 мм от зуба, 128 — в 1.2 мм, а у 117, 123 и 137 оторвались куски на
# 6-11 мм. Такое в отчёт врачу пускать нельзя: «нерв внутри зуба» на чужом
# месте выглядит как настоящая находка.
# Замер по client4 и client5 (медиана расстояния до своего зуба по вокселям
# компоненты): у всех здоровых пульп 0.3-0.4 мм, у сбойных 1.2 и 2.5 мм.
# Порог 1.0 мм лежит ровно в разрыве.
PULP_MAX_DIST_MM = 1.0

CASES_DIR.mkdir(parents=True, exist_ok=True)

# case_id -> запись кейса (см. _new_case)
cases: dict[str, dict] = {}
cases_lock = threading.Lock()

# прогретые предикторы nnU-Net: model -> nnUNetPredictor
_predictors: dict = {}
_pred_lock = threading.Lock()

# кэш массивов для срезов: один кейс (vol int16 ~300МБ + mask uint8 ~150МБ)
_arr_cache = {"key": None, "vol": None, "mask": None}


# ================= постпроцессинг маски (connected components) =================

def _cc_table(binmask: np.ndarray):
    """cc3d по bbox-кропу: (lab, sizes, кроп-срез). Внутренняя."""
    import cc3d
    idx = np.where(binmask)
    lo = [max(int(i.min()) - 1, 0) for i in idx]
    hi = [int(i.max()) + 2 for i in idx]
    crop = binmask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    lab = cc3d.connected_components(crop, connectivity=26)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab, sizes, (lo, hi)


def _comp_shape(comp: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(fill_ratio, aspect, centroid) компоненты внутри её bbox."""
    idx = np.where(comp)
    dims = np.array([int(i.max() - i.min()) + 1 for i in idx], dtype=np.float64)
    fill = comp.sum() / dims.prod()
    aspect = dims.max() / max(float(np.median(dims)), 1.0)
    centroid = np.array([i.mean() for i in idx])
    return fill, aspect, centroid


def _is_streak(comp: np.ndarray) -> bool:
    """Форм-правило: тонкая вытянутая компонента = streak-артефакт."""
    fill, aspect, _ = _comp_shape(comp)
    return fill < SHAPE_RULES["fill_min"] and aspect > SHAPE_RULES["aspect_max"]


def _filter_components(binmask: np.ndarray, min_voxels: int,
                       max_components: int | None,
                       shape_rule: bool = False,
                       merge_dist: float | None = None) -> np.ndarray:
    """Оставляет крупнейшие связные компоненты (26-связность) по bbox-кропу.
    shape_rule — отбраковка streak по форме; merge_dist — компоненты дальше
    этого расстояния от центроида топ-1 отбрасываются (чужие объекты)."""
    out = np.zeros_like(binmask)
    if binmask.sum() < min_voxels:
        return out
    lab, sizes, (lo, hi) = _cc_table(binmask)
    order = np.argsort(sizes)[::-1]
    keep = []
    main_cen = None
    for l in order:
        if sizes[l] < min_voxels:
            continue
        comp = lab == l
        if shape_rule and _is_streak(comp):
            continue  # streak-артефакт
        if merge_dist is not None:
            cen = _comp_shape(comp)[2]
            if main_cen is None:
                main_cen = cen
            elif np.linalg.norm(cen - main_cen) > merge_dist:
                continue  # далеко от основной массы — чужой объект
        keep.append(int(l))
        if max_components is not None and len(keep) >= max_components:
            break
    if not keep:
        return out
    keepmask = np.isin(lab, keep)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = keepmask
    return out


def _filter_tooth(binmask: np.ndarray, spacing=None) -> np.ndarray:
    """Зуб: opening(2) против сращенных streak-хвостов -> CC -> топ-1 +
    фрагмент ВПЛОТНУЮ к основному (разрезанный артефактом зуб), остальное прочь.

    Второй кусок принимается по ЗАЗОРУ между поверхностями, а не по расстоянию
    между центроидами. Прежнее правило (70 вокселей между центроидами, это
    22 мм) оставляло во вьюере зуб, разнесённый на два тела: замер по трём
    клиническим кейсам — принимались куски с зазорами 2.3, 4.7, 7.3, 10.6,
    12.8, 17.0 мм. Врач с приёма так и сказал: «как зуб может состоять из двух
    частей».

    Порог откалиброван по ЭТАЛОНУ, а не подобран: на 790 зубах ToothFairy3
    из двух кусков состоят всего 2 зуба (0.3%), и зазор у них не больше
    1.5 мм. То есть анатомически разорванный зуб — редкость, а разрыв на
    сантиметр анатомией не бывает. Порог 2.0 мм совпал с `COMPONENT_GAP_MM`
    трека сканов, где та же задача калибровалась по своему эталону.
    """
    from scipy import ndimage as ndi
    out = np.zeros_like(binmask)
    if binmask.sum() < CC_RULES["tooth"][0]:
        return out
    idx = np.where(binmask)
    lo = [max(int(i.min()) - 2, 0) for i in idx]
    hi = [int(i.max()) + 3 for i in idx]
    crop = binmask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    opened = ndi.binary_opening(crop, iterations=TOOTH_OPENING_ITERS)

    import cc3d
    lab = cc3d.connected_components(opened, connectivity=26)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = [int(l) for l in np.argsort(sizes)[::-1] if sizes[l] >= TOOTH_MIN_VOX]
    if not order:
        # opening уничтожил мелкий зуб — фолбэк на обычный CC-фильтр сырого
        return _filter_components(binmask, CC_RULES["tooth"][0], 1)
    keep = [order[0]]
    if len(order) > 1:
        main = lab == order[0]
        sp = tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0)
        gap = ndi.distance_transform_edt(~main, sampling=sp)
        for l in order[1:]:
            if len(keep) >= CC_RULES["tooth"][1]:
                break
            comp = lab == l
            if float(gap[comp].min()) <= TOOTH_GAP_MM:
                keep.append(l)  # тот же зуб, разрезанный артефактом
    keepmask = np.isin(lab, keep)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = keepmask
    return out


def _filter_pulp(binmask: np.ndarray, tooth: np.ndarray,
                 spacing) -> np.ndarray:
    """Пульпа: обычный CC-фильтр, затем отсев компонент вне своего зуба.

    Компонента остаётся, если медиана расстояния её вокселей до зуба не больше
    PULP_MAX_DIST_MM. Медиана, а не минимум: сбойная пульпа может краем
    касаться зуба, и минимум её пропустит. И не максимум: у нормальной пульпы
    канал выходит за апекс, там расстояние законно растёт.
    """
    from scipy import ndimage as ndi
    import cc3d
    base = _filter_components(binmask, *CC_RULES["pulp"])
    if not base.any() or tooth.sum() < 30:
        return np.zeros_like(binmask)

    pi, ti = np.where(base), np.where(tooth)
    lo = [max(min(int(a.min()), int(b.min())) - 2, 0) for a, b in zip(pi, ti)]
    hi = [min(max(int(a.max()), int(b.max())) + 3, s)
          for a, b, s in zip(pi, ti, binmask.shape)]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub_p, sub_t = base[box], tooth[box]
    dist = ndi.distance_transform_edt(~sub_t, sampling=tuple(spacing))

    lab = cc3d.connected_components(sub_p, connectivity=26)
    keep = np.zeros_like(sub_p)
    for l in range(1, int(lab.max()) + 1):
        comp = lab == l
        if comp.sum() and float(np.median(dist[comp])) <= PULP_MAX_DIST_MM:
            keep |= comp
    out = np.zeros_like(binmask)
    out[box] = keep
    return out


# «Реставрация», зажатая между зубами ВЕРХНЕЙ и НИЖНЕЙ челюсти, — артефакт
# зоны смыкания, а не пломба: моста между челюстями не бывает, они двигаются.
# Модель `sp` ловит на этом эмаль сомкнутых режущих краёв — врач заметил
# «металлическую штуку» между 11 и 41 у client5, которой нет ни в стороннем
# вьюере, ни у остальных наших моделей (у них там 0 вокселей).
#
# ЭТАЛОНОМ ToothFairy ЭТО НЕ КАЛИБРУЕТСЯ, и это отдельная находка: там
# челюсти РАЗОМКНУТЫ (медиана зазора 7.0 мм, ни одного тома с контактом из
# 24), а у наших пациентов сомкнуты (0.31 мм — один воксель). В эталоне
# правило выполняется тривиально: коснуться обеих челюстей физически нечем.
#
# Поэтому порог взят по НАШИМ кейсам с масками sp (client2, client4, client5),
# доля вокселей компоненты в пределах 1.5 мм от зубов каждой челюсти:
#
#     настоящие пломбы (14789, 8105, 228 вокс)   min(верх,низ) = 0 .. 17.5%
#     артефакты смыкания (604, 314, 56 вокс)     min(верх,низ) = 50 .. 100%
#
# Порог 35% стоит в разрыве. Оговорка: компонент всего восемь на трёх кейсах,
# поэтому дополнительно ограничиваем правило МЕЛКИМИ кусками — крупная
# реставрация не удаляется никогда, какой бы ни была её доля касания.
BITE_ARTIFACT_MAX_VOX = 2000
BITE_ARTIFACT_JAW_FRAC = 0.35
BITE_ARTIFACT_NEAR_MM = 1.5


def _drop_bite_artifacts(filtered: np.ndarray, mask: np.ndarray,
                         spacing) -> np.ndarray:
    """Убрать мелкие «реставрации», прижатые к зубам обеих челюстей сразу."""
    if not filtered.any() or spacing is None:
        return filtered
    import cc3d
    from scipy import ndimage as ndi
    up = np.isin(mask, [t for t in TEETH_CLASSES if t // 10 in (1, 2)])
    lo = np.isin(mask, [t for t in TEETH_CLASSES if t // 10 in (3, 4)])
    if up.sum() < 500 or lo.sum() < 500:
        return filtered            # одной челюсти нет — судить не о чем
    du = ndi.distance_transform_edt(~up, sampling=spacing)
    dl = ndi.distance_transform_edt(~lo, sampling=spacing)
    lab = cc3d.connected_components(filtered, connectivity=26)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    out = filtered.copy()
    for l in np.nonzero(sizes)[0]:
        if sizes[l] > BITE_ARTIFACT_MAX_VOX:
            continue
        comp = lab == l
        n = float(comp.sum())
        fu = float((du[comp] <= BITE_ARTIFACT_NEAR_MM).sum()) / n
        fl = float((dl[comp] <= BITE_ARTIFACT_NEAR_MM).sum()) / n
        if min(fu, fl) > BITE_ARTIFACT_JAW_FRAC:
            out[comp] = False
    return out


def _drop_off_bone(filtered: np.ndarray, bone_dist: np.ndarray,
                   thr: float = TOOTH_BONE_MAX_MM) -> np.ndarray:
    """Убрать куски, стоящие дальше `thr` от челюстной кости.

    Покомпонентно, а не по классу целиком: у класса реставраций в одном
    томе законно лежит несколько коронок, и дальний мусор не должен уносить
    с собой соседнюю настоящую пломбу.
    """
    if not filtered.any():
        return filtered
    import cc3d
    lab = cc3d.connected_components(filtered, connectivity=26)
    out = filtered.copy()
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    for l in np.nonzero(sizes)[0]:
        comp = lab == l
        if float(np.median(bone_dist[comp])) > thr:
            out[comp] = False
    return out


def canal_body(tooth: np.ndarray, pulp: np.ndarray) -> np.ndarray:
    """Зуб вместе с пульпой и со щелями, где модель не отметила ни то, ни
    другое: недорисованный канал модель метит фоном, и без замыкания в теле
    зуба вдоль канала остаётся дыра, куда рост не пустили бы."""
    from scipy import ndimage as ndi
    return ndi.binary_fill_holes(
        ndi.binary_closing(tooth | pulp, iterations=2)) | tooth | pulp


CANAL_NET_WEIGHTS = R3D.parent / "research/output/canal_net/best.pt"
_canal_net = {"key": None, "net": None, "device": None}


def canal_net():
    """Модель канала на вырезе зуба; None, если весов нет.

    Она ЛУЧШЕ трассировки по плотности — сверено одним протоколом на
    эталонных томах ToothFairy3, которых сеть не видела (95 премоляров и
    моляров): ошибка дохода 0.017 против 0.089, число каналов в срезе
    совпадает с эталоном в 91% против 71% при 11 лишних каналах против 40,
    покрытие отрезанного канала 0.99 против 0.84, точность прироста 0.99
    против 0.86, разрыв в середине канала закрыт целиком у 94 из 94 против
    57 из 94. Союз двух способов пробовали — он только теряет точность
    (0.89) и добавляет лишние каналы, поэтому берётся сеть одна.

    Трассировка остаётся запасным путём: без весов конвейер работает как
    раньше, а не падает.
    """
    if _canal_net["key"] is None:
        _canal_net["key"] = "loaded"
        if CANAL_NET_WEIGHTS.is_file():
            try:
                import torch
                import cbct_canal_apply as ca
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _canal_net["net"], _ = ca.load_net(CANAL_NET_WEIGHTS, dev)
                _canal_net["device"] = dev
            except Exception as e:                      # noqa: BLE001
                print(f"модель канала не загрузилась ({e}), работаем трассировкой")
    return _canal_net["net"], _canal_net["device"]


def _complete_canals(out: np.ndarray, vol: np.ndarray, spacing) -> None:
    """Дорисовать корневые каналы у всех зубов (на месте, в `out`).

    Сначала модель канала, и только без неё — трассировка по плотности.
    """
    import cbct_canals
    net, device = canal_net()
    zup = occlusal_up(out)
    sp = np.asarray(spacing, float)
    for fdi in TEETH_CLASSES:
        pulp_all = out == fdi + 100
        if not pulp_all.any():
            continue
        both = (out == fdi) | pulp_all
        idx = np.argwhere(both)
        lo = np.maximum(idx.min(0) - 4, 0)
        hi = np.minimum(idx.max(0) + 5, out.shape)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        sub = out[box]
        tooth, pulp = sub == fdi, sub == fdi + 100
        body = canal_body(tooth, pulp)
        sgn = 1 if (fdi // 10 in (1, 2)) == zup else -1
        if net is not None:
            import cbct_canal_apply as ca
            add = ca.canal_in_tooth(net, device, vol[box], body, pulp, sp, sgn) & ~pulp
        else:
            add = cbct_canals.complete_canals(vol[box], body, pulp, sp, sgn)
        # в канал превращаем только сам зуб и щели в нём, чужие метки не трогаем
        add &= (sub == fdi) | (sub == 0)
        if not add.any():
            continue
        # Приращение, оторванное от пульпы, в 3D висит крошкой — оставляем
        # только куски, касающиеся исходной пульпы, или заметные сами по себе.
        import cc3d
        lab = cc3d.connected_components(pulp | add, connectivity=26)
        sizes = np.bincount(lab.ravel())
        keep_ids = set(np.unique(lab[pulp]).tolist()) | set(
            np.nonzero(sizes >= 30)[0].tolist())
        keep_ids.discard(0)
        add &= np.isin(lab, list(keep_ids))
        sub[add] = fdi + 100


def filter_mask(mask: np.ndarray, vol: np.ndarray | None = None,
                spacing=None, canals: bool = True) -> np.ndarray:
    """HU-гейтинг (по vol) + CC-фильтрация всех классов. Исходники не трогаем.

    Порядок групп важен: зубы фильтруются раньше пульпы, поэтому правилу
    «пульпа внутри своего зуба» достаётся уже очищенный зуб.
    """
    out = np.zeros_like(mask)
    bone_dist = None            # считаем один раз, когда дойдём до зубов
    for g in GROUPS:
        key = g["key"]
        min_vox, max_comp = CC_RULES[GROUP_CC_RULE[key]]
        shape_rule = key in SHAPE_RULE_GROUPS
        gate = HU_GATE.get(key)
        for cls in g["classes"]:
            bm = mask == cls
            if gate and vol is not None:  # HU-гейт первым шагом
                op, thr = gate
                bm = bm & (vol >= thr) if op == "min" else bm & (vol <= thr)
            if key == "teeth":
                filtered = _filter_tooth(bm, spacing)
            elif key == "pulp" and spacing is not None:
                filtered = _filter_pulp(bm, out == (cls - 100), spacing)
            else:
                filtered = _filter_components(
                    bm, min_vox, max_comp, shape_rule=shape_rule,
                    merge_dist=(LARGE_MERGE_DIST
                                if GROUP_CC_RULE[key] == "large" else None))
            # Зуб и коронка сидят в кости — то, что висит от неё далеко,
            # это артефакт сегментации, а не анатомия. Челюсть в GROUPS
            # стоит раньше, поэтому в `out` она уже отфильтрована.
            if (key in BONE_ANCHORED_GROUPS and spacing is not None
                    and filtered.any()):
                if bone_dist is None:
                    from scipy import ndimage as ndi
                    # Расстояние меряем до челюсти ИЗ МАСКИ МОДЕЛИ, а не до
                    # того, что осталось от неё после нашего же HU-гейта.
                    # Гейт выгрызает 30-54% кости (замер по четырём кейсам),
                    # зубы «отдаляются» от огрызков, и порог, откалиброванный
                    # на ПОЛНЫХ эталонных масках, начинает резать живое: на
                    # client4 так терялось 13 настоящих нижних зубов из 32.
                    # Та же ловушка, что уже описана в проекте: база
                    # измерения на эталоне не равна базе на предсказании.
                    jaw = np.isin(mask, JAW_FOR_MIDLINE)
                    bone_dist = (ndi.distance_transform_edt(
                        ~jaw, sampling=spacing)
                        if jaw.sum() >= 1000 else False)
                if bone_dist is not False:
                    filtered = _drop_off_bone(filtered, bone_dist)
            if key == "prosthetics" and spacing is not None and filtered.any():
                filtered = _drop_bite_artifacts(filtered, out, spacing)
            out[filtered] = cls
    # Каналы — после всех групп: нужны и отфильтрованные зубы, и пульпа.
    # Модель недометит тонкие корневые каналы, а на снимке они видны —
    # см. cbct_canals.
    if canals and vol is not None and spacing is not None:
        _complete_canals(out, vol, spacing)
    return out


def filtered_mask_for(c: dict, model: str) -> np.ndarray:
    """Фильтрованная маска кейса (общий кэш с механизмом срезов)."""
    _arrays_for(c, model)
    return _arr_cache["mask"]


def arrays_for(c: dict, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Объём + фильтрованная маска (тот же кэш, что у срезов и мешинга).
    Нужно слою находок: он читает и HU объёма, и маску."""
    return _arrays_for(c, model)


# ================= кейсы =================

def _new_case(case_id: str, name: str, volume: Path,
              masks: dict | None = None, example: bool = False) -> dict:
    c = {
        "case_id": case_id,
        "name": name,
        "volume": str(volume),
        "masks": masks or {"ours": None, "pretrained": None},
        "model": None,            # модель последней успешной сегментации
        "example": example,
        "stage": "ready",         # ready|predicting|meshing|done|error
        "detail": "",
        "shape": None,            # [D, H, W]
        "spacing": None,          # [sz, sy, sx]
        "info": None,             # заполняется после meshing
        "findings": None,         # слой находок (cbct_findings), после info
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with cases_lock:
        cases[case_id] = c
    return c


def _save_meta(c: dict) -> None:
    d = CASES_DIR / c["case_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(
        {k: c.get(k) for k in ("case_id", "name", "volume", "masks", "model",
                               "example", "shape", "spacing", "info",
                               "findings", "review", "panorama", "meshes",
                               "scanner", "sides_applied", "sides_uncertain",
                               "sides_vote", "created_at")},
        ensure_ascii=False), encoding="utf-8")


def _load_persisted() -> None:
    """Поднимаем кейсы из meta.json (после перезапуска сервера)."""
    for meta in CASES_DIR.glob("*/meta.json"):
        try:
            c = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Кейс БЕЗ ТОМА законен и роняться на нём нельзя. Том весит 300+ МБ и
        # нужен только для срезов и пересчёта маски; кейс для разработки
        # (`fixtures/`) состоит из меша и уже посчитанных находок, и этого
        # хватает для 3D, находок, плана и PDF. Раньше `volume: None` валил
        # весь модуль на импорте — то есть и весь сервер, — а отсутствующий
        # файл молча выбрасывал кейс из списка.
        vol = c.get("volume")
        c["has_volume"] = bool(vol) and Path(vol).is_file()
        if not c["has_volume"] and not (c.get("findings") or c.get("meshes")):
            continue   # ни тома, ни посчитанного — показывать нечего
        # инвалидация устаревшего постпроцессинга: меши/инфо от старых правил
        # Правило сторон изменилось после сборки — всё пересчитывать заново.
        if (c.get("sides_applied") is not None
                and c.get("sides_applied")
                != bool(scanner_sides_state(c).get("mirrored"))):
            c["info"] = None
            c["findings"] = None
            c["panorama"] = None
            c["meshes"] = {}
            for glb in (CASES_DIR / c["case_id"]).glob("mesh_*.glb"):
                glb.unlink(missing_ok=True)
            for png in (CASES_DIR / c["case_id"]).glob("pano_*.png"):
                png.unlink(missing_ok=True)
        if c.get("info") and c["info"].get("postproc") != POSTPROC_VERSION:
            c["info"] = None
            c["findings"] = None   # находки считаются по той же маске
            c["panorama"] = None
            for png in (CASES_DIR / c["case_id"]).glob("pano_*.png"):
                png.unlink()
        # Версия каждого меша ОТДЕЛЬНО. `info` в кейсе один, а мешей три (по
        # модели), и раньше они инвалидировались скопом по `info`. Пока меш
        # всегда строился заново, это было безвредно; теперь готовый GLB
        # переиспользуется — и меш другой модели, собранный по прошлым
        # правилам, молча уехал бы врачу. Записи нет (кейс от старой версии) —
        # верим `info`, но только для ЕГО модели.
        meshes = dict(c.get("meshes") or {})
        for glb in (CASES_DIR / c["case_id"]).glob("mesh_*.glb"):
            model = glb.stem[len("mesh_"):]
            ver = meshes.get(model)
            if ver is None:
                info = c.get("info") or {}
                ver = (POSTPROC_VERSION if info.get("model") == model
                       and info.get("postproc") == POSTPROC_VERSION else None)
                if ver is not None:
                    meshes[model] = ver
            if ver != POSTPROC_VERSION:
                glb.unlink()
                meshes.pop(model, None)
        c["meshes"] = meshes
        c.setdefault("findings", None)
        c.setdefault("panorama", None)
        c["stage"] = "done" if c.get("info") else "ready"
        c["detail"] = ""
        with cases_lock:
            cases[c["case_id"]] = c


def register_examples() -> None:
    """Готовые примеры client2-5: маски уже посчитаны обеими моделями."""
    for n in range(2, 6):
        cid = f"client{n}"
        if cid in cases:
            continue
        vol = R3D / f"inference_input/{cid}_0000.mha"
        if not vol.is_file():
            continue
        masks = {}
        for model, suffix in (("ours", "_seg_ours"), ("pretrained", "_seg")):
            p = R3D / f"inference_output/{cid}{suffix}.mha"
            masks[model] = str(p) if p.is_file() else None
        c = _new_case(cid, f"Пример: пациент {n}", vol, masks, example=True)
        reader = sitk.ImageFileReader()  # читаем только заголовок
        reader.SetFileName(str(vol))
        reader.ReadImageInformation()
        c["shape"] = list(reader.GetSize()[::-1])
        c["spacing"] = list(reader.GetSpacing()[::-1])
        _save_meta(c)


def case_dir(c: dict) -> Path:
    """Каталог кейса. У фикстуры он в `fixtures/`, у обычного — в `web_cases`.

    Отдельная функция, потому что путь собирался в шести местах по одному
    шаблону: пропусти одно — и меш фикстуры «не найден», а причина
    неочевидна.
    """
    d = c.get("dir")
    return Path(d) if d else CASES_DIR / c["case_id"]


def register_fixtures() -> None:
    """Кейсы из `fixtures/` — чтобы демо было чем наполнить без снимков.

    Разработчику, взявшему проект из git, показывать нечего: снимки пациентов
    в репозиторий не кладутся. В `fixtures/` лежит кейс из публичного набора
    (CC BY 4.0) — меш и уже посчитанные находки, без тома. Этого хватает для
    3D, находок, плана, PDF и чата.

    Регистрируем ЧТЕНИЕМ ИЗ `fixtures/`, не копией в `web_cases`: копия
    завела бы вторую правду об одном кейсе, а инвалидация кэша удаляла бы
    файлы, лежащие в гите.
    """
    root = R3D.parent / "fixtures"
    if not root.is_dir():
        return
    for meta in sorted(root.glob("*/meta.json")):
        try:
            c = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cid = c.get("case_id")
        if not cid or cid in cases:
            continue
        c["fixture"] = True
        c["has_volume"] = False
        c["dir"] = str(meta.parent)
        c["stage"] = "done" if c.get("info") else "ready"
        c["detail"] = ""
        with cases_lock:
            cases[cid] = c


def get_case(case_id: str) -> dict | None:
    if not re.fullmatch(r"[\w\-]+", case_id):
        return None
    return cases.get(case_id)


def list_examples() -> list[dict]:
    """Примеры для списка на экране — и фикстуры тоже.

    Фикстуру надо показывать: на машине без снимков она единственный кейс, и
    без неё список примеров пуст. Пустой список читается как «ничего нет и
    всё сломано», хотя смотреть есть что.
    """
    return [{"case_id": c["case_id"], "name": c["name"],
             "fixture": bool(c.get("fixture"))}
            for c in sorted(cases.values(), key=lambda x: x["case_id"])
            if c.get("example") or c.get("fixture")]


# ================= загрузка DICOM =================

def import_upload(files: list) -> dict:
    """files: [(filename, bytes), ...] — один ZIP или набор .dcm.
    Возвращает запись кейса (stage=ready)."""
    cid = f"case_{int(time.time())}"
    cdir = CASES_DIR / cid
    raw = cdir / "dcm"
    raw.mkdir(parents=True)

    is_zip = len(files) == 1 and files[0][0].lower().endswith(".zip")
    if is_zip:
        zpath = raw / "upload.zip"
        zpath.write_bytes(files[0][1])
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)
        zpath.unlink()
    else:
        for fname, data in files:
            if not fname.lower().endswith(".dcm"):
                continue
            (raw / Path(fname).name).write_bytes(data)

    # каталог с серией: где больше всего .dcm
    best, best_n = None, 0
    for dirpath, _, filenames in os.walk(raw):
        n = sum(1 for f in filenames if f.lower().endswith(".dcm"))
        if n > best_n:
            best, best_n = dirpath, n
    if best is None or best_n < 5:
        shutil.rmtree(cdir, ignore_errors=True)
        raise ValueError("В архиве не найдено DICOM-серии (нужно от 5 файлов .dcm)")

    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(best)
    if not names:
        shutil.rmtree(cdir, ignore_errors=True)
        raise ValueError("SimpleITK не смог собрать серию из этих файлов")
    reader.SetFileNames(names)
    img = reader.Execute()

    vpath = cdir / "volume.mha"
    sitk.WriteImage(img, str(vpath))
    # Аппарат читаем ДО удаления сырых срезов. Раньше строка стояла после
    # rmtree: файл к тому моменту уже не существовал, read_scanner молча
    # возвращал пустоту, и у КАЖДОГО загруженного через демо кейса аппарат
    # выходил «неизвестен». Последствие тяжелее, чем пустая строка в панели:
    # без аппарата не находится подтверждённая конвенция сторон, зеркало не
    # применяется, и номера зубов уезжают на чужую сторону — ровно та ошибка,
    # которая в этом проекте стоила дороже всех остальных.
    scanner = read_scanner(names[0])
    shutil.rmtree(raw, ignore_errors=True)  # сырые dcm больше не нужны

    c = _new_case(cid, f"Загрузка {time.strftime('%d.%m %H:%M')}", vpath)
    c["shape"] = list(img.GetSize()[::-1])      # (W,H,D) -> [D,H,W]
    c["spacing"] = list(img.GetSpacing()[::-1])  # (x,y,z) -> [z,y,x]
    c["scanner"] = scanner
    _save_meta(c)
    return c


# ================= аппарат =================

# Определить левое и правое на симметричной челюсти НЕЛЬЗЯ ни одним
# алгоритмом — это внешнее знание, и в файл его пишет аппарат. Но у наших
# снимков он пишет заглушку: `ImagePositionPatient = (0,0,0)` и одинаковый
# `ImageOrientationPatient` у всех, независимо от укладки. Замер по четырём
# кейсам клиники:
#
#     client2, client4, client5 — Dentium rainbowCT
#     client3                   — Dentium brightCT
#     метаданные ориентации     — у всех ОДИНАКОВЫЕ и все заглушки
#
# Врач заметил расхождение сторон ровно на единственном снимке с другого
# аппарата. Поэтому модель аппарата запоминается в кейсе: конвенцию сторон
# достаточно подтвердить ОДИН РАЗ на один аппарат, а дальше применять по
# этому полю. Догадываться на каждом снимке нельзя — не по чему.
SCANNER_TAGS = (
    (0x0008, 0x0070),   # Manufacturer
    (0x0008, 0x1090),   # ManufacturerModelName
    (0x0018, 0x5100),   # PatientPosition
    (0x0008, 0x1010),   # StationName
)


def read_scanner(dcm_path: str) -> dict:
    """Кто снимал: производитель, модель, укладка. Пустые поля не скрываем."""
    try:
        import pydicom
        d = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    except Exception:                              # noqa: BLE001
        return {}
    out = {}
    for tag, key in zip(SCANNER_TAGS, ("manufacturer", "model",
                                       "patient_position", "station")):
        v = d.get(tag)
        out[key] = str(v.value).strip() if v is not None and v.value else ""
    iop = d.get((0x0020, 0x0037))
    ipp = d.get((0x0020, 0x0032))
    out["iop"] = [round(float(x), 3) for x in iop.value] if iop else None
    out["ipp_first"] = [round(float(x), 1) for x in ipp.value] if ipp else None
    # Заглушка в позиции — признак того, что ориентации в файле верить нельзя.
    out["position_is_stub"] = bool(
        out["ipp_first"] and all(abs(v) < 1e-6 for v in out["ipp_first"]))
    return out


# Подтверждённые врачом конвенции сторон, по одному правилу на аппарат.
# Файл, а не константа в коде: правило добывается на приёме, врачом, и должно
# переживать обновление программы.
SIDES_FILE = R3D / "scanner_sides.json"


def load_scanner_sides() -> dict:
    try:
        return json.loads(SIDES_FILE.read_text(encoding="utf-8"))
    except Exception:                              # noqa: BLE001
        return {}


def set_scanner_sides(key: str, mirrored: bool, by: str = "") -> dict:
    """Запомнить: снимки этого аппарата надо зеркалить (или не надо).

    Вызывается ОДИН раз на аппарат, после того как врач сверил снимок с
    заведомо верным источником. Дальше правило применяется ко всем снимкам
    с этого аппарата само.
    """
    data = load_scanner_sides()
    data[key] = {"mirrored": bool(mirrored), "by": by,
                 "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    SIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SIDES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return data[key]


def scanner_sides_state(c: dict) -> dict:
    """Что известно про стороны у аппарата этого кейса."""
    key = scanner_key(c)
    rec = load_scanner_sides().get(key)
    stub = bool((c.get("scanner") or {}).get("position_is_stub"))
    return {
        "scanner": key,
        "confirmed": rec is not None,
        "mirrored": bool(rec and rec.get("mirrored")),
        "by": (rec or {}).get("by", ""),
        "at": (rec or {}).get("at", ""),
        # Главное для интерфейса: можно ли вообще доверять файлу.
        "position_is_stub": stub,
        "uncertain": bool(c.get("sides_uncertain")),
        "vote": c.get("sides_vote"),
        "note": ("Аппарат пишет позицию пациента заглушкой — определить "
                 "сторону по файлу нельзя, нужно подтверждение врача."
                 if stub and rec is None else ""),
    }


def scanner_key(c: dict) -> str:
    """Короткий ключ аппарата — по нему хранится подтверждённая конвенция."""
    s = c.get("scanner") or {}
    parts = [s.get("manufacturer", ""), s.get("model", "")]
    return " ".join(p for p in parts if p).strip() or "неизвестен"


# ================= инференс nnU-Net =================

# Порог, выше которого карту вероятностей собираем на CPU, а не на GPU.
# nnU-Net при perform_everything_on_device=True держит на карте тензор
# «классы x весь объём»: при 78 классах и томе 599x536x536 это 26.8 ГБ в
# half precision, и вместе с самой сетью оно в 34.2 ГБ не помещается. Windows
# при этом не падает с OOM, а подменяет видеопамять оперативной — инференс
# client4 шёл 69 минут вместо трёх, GPU при этом показывал 100%.
# Замер карты вероятностей (78 классов, half):
#     client2/client5 (348 срезов) — 15.6 ГБ, помещается
#     client4         (599 срезов) — 26.8 ГБ, НЕ помещается
# 12 ГБ — с запасом от 15.6, чтобы «влезающие» кейсы считались на GPU.
PROB_MAP_GPU_LIMIT_GB = 12.0


def _fits_on_gpu(model: str, shape) -> bool:
    """Поместится ли карта вероятностей этой модели на видеокарту."""
    n_cls = (47 if model in SINGLE_PULP_MODELS
             else 78 if model in MODEL_NEEDS_REMAP else 49)
    gb = n_cls * float(np.prod(shape)) * 2 / 1e9      # half precision
    return gb <= PROB_MAP_GPU_LIMIT_GB


def get_predictor(model: str, on_device: bool = True):
    """Загрузка предиктора один раз (потокобезопасно).

    `on_device` — собирать ли карту вероятностей на GPU. Для больших томов её
    приходится собирать на CPU (см. PROB_MAP_GPU_LIMIT_GB): сам счёт остаётся
    на GPU, медленнее становится только агрегация, зато без спилла.
    """
    key = f"{model}:{'gpu' if on_device else 'cpu'}"
    with _pred_lock:
        if key in _predictors:
            return _predictors[key]
        # Освобождаем место ДО загрузки новой: иначе на карте окажутся обе.
        _evict_predictors(key)
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        p = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=on_device,
            device=torch.device("cuda", 0),
            verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(
            str(MODELS[model]), use_folds=(0,),
            checkpoint_name="checkpoint_best.pth")
        _predictors[key] = p
        return p


# Сколько предикторов держать в памяти одновременно.
#
# Раньше при старте грузились три модели, а дальше к ним добавлялись все, что
# врач открывал, и ничего не выгружалось: демо разрасталось до 22 ГБ ОЗУ и
# держало 21 ГБ видеопамяти. На рабочей машине врача это мешает всему
# остальному — он это и заметил.
#
# Одна модель весит около 4-5 ГБ. Держим только текущую: переключение модели
# станет ждать загрузку (около минуты), зато демо перестаёт занимать машину
# целиком. Прогрев при старте убран по той же причине — он грузил модели,
# которые могут и не понадобиться.
MAX_LOADED_PREDICTORS = 1


def _evict_predictors(keep_key: str) -> None:
    """Выгрузить всё, кроме нужного предиктора, и вернуть память карте."""
    drop = [k for k in _predictors if k != keep_key]
    if len(_predictors) <= MAX_LOADED_PREDICTORS and not drop:
        return
    for k in drop:
        _predictors.pop(k, None)
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:                              # noqa: BLE001
        pass


def warmup() -> None:
    """Прогрева больше нет: модель грузится при первом обращении к ней.

    Оставлено как точка вызова, чтобы не менять demo_server.
    """
    print("CBCT: модели грузятся по требованию "
          f"(в памяти держим {MAX_LOADED_PREDICTORS})")


# Парные структуры «лево/право». Конвенция снята с модели Dataset112, которая
# стороны различает верно: левые структуры лежат ПРАВЕЕ средней линии по оси x
# (обычная рентгенологическая раскладка — левая сторона пациента справа на
# изображении). Проверено на client4: канал 3 x=283, канал 4 x=157 при
# средней линии 252; пазухи 5 и 6 — 329 и 179.
PAIRED_SIDES = [(3, 4), (5, 6), (103, 104)]   # (левый класс, правый класс)
JAW_FOR_MIDLINE = (1, 2)                      # по чему считаем среднюю линию
SIDE_MIN_VOX = 50
MIN_TOOTH_VOX_SIDE = 300   # мелкий осколок голосовать о стороне не должен
# Доля голосов, ниже которой разворот НЕ делается. На верных масках
# голосование единогласно (25:0, 29:2, 31:1), спорные соотношения вроде 17:12
# встречаются только на слабых моделях, где номера и так недостоверны.
SIDE_VOTE_MIN_SHARE = 0.75

_remap_lut: dict[str, np.ndarray] = {}


QUADRANT_MIRROR = {1: 2, 2: 1, 3: 4, 4: 3}


def left_axis_sign(direction) -> int:
    """Куда по оси 2 массива растёт ЛЕВАЯ сторона пациента: +1 или -1.

    SimpleITK работает в LPS, поэтому столбцы direction говорят, куда смотрят
    оси изображения. Ищем ту, что отвечает за лево-право, и берём её знак.
    Геометрией сторону не вывести — левое и правое зеркально симметричны, —
    так что это единственный честный источник.
    """
    d = np.asarray(direction, float).reshape(3, 3)
    k = int(np.argmax(np.abs(d[0])))      # ось изображения вдоль L-R
    return 1 if d[0, k] > 0 else -1


def apply_scanner_mirror(vol, mask, case):
    """Отразить снимок по лево-правой оси, если аппарат пишет его зеркально.

    ЗАЧЕМ ИМЕННО ОТРАЖЕНИЕ, а не перестановка номеров. Врач с приёма: «ты
    строишь 3D неправильно и перепутал стороны» — и был прав. Зеркальна была
    САМА ГЕОМЕТРИЯ: зуб, стоящий у пациента слева, оказывался в модели справа.
    Переставлять при этом только номера бессмысленно — врач смотрит на снимок
    и сверяет его с исходным вьюером, а не читает подписи.

    Почему это не поймали проверки: они опирались на правило правой тройки
    («левая сторона = верх x перёд»), а на ЗЕРКАЛЬНОЙ модели это правило само
    даёт зеркальный ответ. Проверка снова оказалась замкнутой на ту же ошибку,
    что и раньше с ориентацией из файла.

    Отражаем ОБА массива согласованно, а вместе с ними переставляем номера
    квадрантов: сеть присвоила их на зеркальных данных, поэтому после
    отражения метка «12» окажется на зубе, который на самом деле 22.

    Правило берётся из подтверждённой врачом конвенции аппарата
    (`scanner_sides.json`), потому что по файлу сторону определить нельзя —
    аппараты пишут позицию пациента заглушкой.
    """
    if case is None or not scanner_sides_state(case).get("mirrored"):
        return vol, mask
    # vol=None допустим: проверкам бывает нужна только маска, а том они не
    # читают (он на 300 МБ и разворачивать его незачем).
    if vol is not None:
        vol = np.ascontiguousarray(vol[:, :, ::-1])
    mask = np.ascontiguousarray(mask[:, :, ::-1])
    out = mask.copy()
    for t in TEETH_CLASSES:
        mate = QUADRANT_MIRROR[t // 10] * 10 + t % 10
        out[mask == t] = mate
        out[mask == t + 100] = mate + 100
    for a, b in PAIRED_SIDES:
        out[mask == a] = b
        out[mask == b] = a
    return vol, out


def orient_sides(mask: np.ndarray, direction=None, case: dict | None = None) -> np.ndarray:
    """Привести номера сторон к анатомии пациента, а не к конвенции датасета.

    ЗАЧЕМ. Разметка ToothFairy (и 2, и 3) называет стороны зеркально
    относительно ориентации, записанной в её же файлах: замер по 80 томам —
    класс «Left IAC» во ВСЕХ лежит там, где по LPS правая сторона. Модель это
    честно воспроизводит, и на снимке с нормальной ориентацией зуб 22
    оказывается справа у пациента, то есть на месте зуба 12.

    Цена ошибки клиническая: план лечения уходит врачу с зеркальными номерами.
    Врач с приёма заметил это за минуту — «22-й стоит не с той стороны».

    Чиним не доверием к модели, а голосованием по геометрии: у зубов
    квадрантов 2 и 3 координата вдоль «влево» обязана быть больше средней
    линии. Если против этого большинство зубов — зеркалим ВСЕ номера разом
    (и зубы, и их пульпы, и парные каналы с пазухами).

    Решение принимается глобально, а не по каждому зубу: зеркало либо есть,
    либо нет, а поштучный разбор сломал бы верные модели на зубе, который
    просто уехал за среднюю линию.
    """
    sign = left_axis_sign(direction) if direction is not None else 1
    # Конвенцию аппарата здесь НЕ учитываем: она применяется раньше, в
    # `apply_scanner_mirror`, и уже отразила и геометрию, и номера. Голосование
    # ниже ловит другое — случай, когда сама МОДЕЛЬ путает стороны (так делают
    # сети, обученные с зеркалированием лево-право).
    jaw = np.isin(mask, JAW_FOR_MIDLINE)
    if jaw.sum() < 1000:
        return mask
    mid = float(np.where(jaw)[2].mean())

    agree = disagree = 0
    for t in TEETH_CLASSES:
        bm = mask == t
        if bm.sum() < MIN_TOOTH_VOX_SIDE:
            continue
        cx = float(np.where(bm)[2].mean())
        toward_left = (cx - mid) * sign > 0
        if toward_left == (t // 10 in (2, 3)):
            agree += 1
        else:
            disagree += 1
    if agree + disagree < 6 or disagree <= agree:
        return mask
    # Разворачиваем при любом большинстве, но УВЕРЕННОСТЬ запоминаем.
    #
    # Не разворачивать при слабом голосе — плохой выход: большинство зубов
    # тогда останется зеркальным. Замер по моделям одного кейса: у сильных
    # (sp, combo, pretrained) голосование единогласно, 93-100%, а у слабых
    # (ours, tf3) бывает 59-78% — там сеть путает стороны поштучно, и её
    # номерам нельзя верить независимо от разворота. Поэтому переворачиваем,
    # но помечаем кейс, чтобы врач видел предупреждение, а не узнавал о
    # ненадёжности из перепутанных номеров в плане.
    share = disagree / float(agree + disagree)
    if case is not None:
        case["sides_uncertain"] = bool(share < SIDE_VOTE_MIN_SHARE)
        case["sides_vote"] = [int(disagree), int(agree)]
    print(f"CBCT: стороны зеркальны ({disagree} зубов против {agree}, "
          f"{100*share:.0f}%) — переставляю номера квадрантов"
          + ("  ВНИМАНИЕ: голос неуверенный" if share < SIDE_VOTE_MIN_SHARE
             else ""))
    out = mask.copy()
    for t in TEETH_CLASSES:
        mate = QUADRANT_MIRROR[t // 10] * 10 + t % 10
        out[mask == t] = mate
        out[mask == t + 100] = mate + 100
    for a, b in PAIRED_SIDES:
        out[mask == a] = b
        out[mask == b] = a
    return out


def split_paired_sides(mask: np.ndarray, direction=None) -> np.ndarray:
    """Восстанавливает сторону у парных структур по средней линии челюсти.

    Модель Dataset113 при патче 64x160x160 не различает лево и право у почти
    зеркальных структур: оба канала и обе пазухи она кладёт в «левый» класс
    (замер: класс 3 занимает 55% слева и 44% справа от средней линии, класс 4
    почти пуст). Форму при этом берёт верно, поэтому сторону дешевле и
    надёжнее восстановить геометрией, чем переучивать модель с большим патчем.
    """
    import cc3d
    sign = left_axis_sign(direction) if direction is not None else 1
    jaw = np.isin(mask, JAW_FOR_MIDLINE)
    if jaw.sum() < 1000:
        return mask
    mid = float(np.where(jaw)[2].mean())
    out = mask
    for left_cls, right_cls in PAIRED_SIDES:
        bm = (mask == left_cls) | (mask == right_cls)
        if bm.sum() < SIDE_MIN_VOX:
            continue
        if out is mask:
            out = mask.copy()
        lab = cc3d.connected_components(bm, connectivity=26)
        sizes = np.bincount(lab.ravel())
        for l in range(1, len(sizes)):
            if sizes[l] < SIDE_MIN_VOX:
                continue
            comp = lab == l
            cx = float(np.where(comp)[2].mean())
            out[comp] = left_cls if (cx - mid) * sign > 0 else right_cls
    return out


def remap_prediction(seg: np.ndarray, model: str, spacing=None) -> np.ndarray:
    """Метки модели -> оригинальная нумерация ToothFairy (FDI, пульпа = зуб+100).

    Для моделей Dataset112 это тождество: они обучены прямо в FDI.

    Для моделей Dataset114 (пульпа единым классом) отображение своё: классы
    1-45 переводятся в FDI, а класс 46 — это ВСЯ пульпа сразу, и номер зуба ей
    возвращается геометрией, по ближайшему зубу (см. cbct_prepare_singlepulp).
    """
    if model in SINGLE_PULP_MODELS:
        import cbct_prepare_singlepulp as sp
        if model not in _remap_lut:
            mp, _labels = sp.build_mapping()
            back = {}
            for train_id, new_id in mp.items():
                if new_id and new_id != sp.SINGLE_PULP_ID:
                    back[new_id] = train_id
            lm = json.loads(LABEL_MAP_FILE.read_text(encoding="utf-8"))
            t2o = {int(k): int(v) for k, v in lm["train_to_orig"].items()}
            lut = np.zeros(sp.SINGLE_PULP_ID + 1, dtype=np.int16)
            for new_id, train_id in back.items():
                lut[new_id] = t2o[train_id]
            # Пульпе временный id, не пересекающийся с номерами зубов.
            lut[sp.SINGLE_PULP_ID] = sp.SINGLE_PULP_FDI
            _remap_lut[model] = lut
        lut = _remap_lut[model]
        out = lut[np.clip(seg, 0, len(lut) - 1)].astype(np.int16)
        return sp.assign_pulp_to_teeth(out, spacing=spacing)
    if model not in MODEL_NEEDS_REMAP:
        return seg
    if model not in _remap_lut:
        m = json.loads(LABEL_MAP_FILE.read_text(encoding="utf-8"))
        pairs = {int(k): int(v) for k, v in m["train_to_orig"].items()}
        lut = np.zeros(max(pairs) + 1, dtype=np.uint8)
        for train_id, orig in pairs.items():
            lut[train_id] = orig
        _remap_lut[model] = lut
    lut = _remap_lut[model]
    return lut[np.clip(seg, 0, len(lut) - 1)]


def _target_spacing(model: str) -> tuple | None:
    """Спейсинг 3d_fullres из планов модели, в порядке SimpleITK (x, y, z)."""
    try:
        plans = json.loads((MODELS[model] / "plans.json").read_text(encoding="utf-8"))
        sp = plans["configurations"]["3d_fullres"]["spacing"]   # (z, y, x)
        return tuple(float(v) for v in sp[::-1])
    except Exception:  # noqa: BLE001
        return None


def _resample(img, spacing_out, interpolator, default_value):
    size_in, sp_in = img.GetSize(), img.GetSpacing()
    size_out = [max(int(round(n * si / so)), 1)
                for n, si, so in zip(size_in, sp_in, spacing_out)]
    return sitk.Resample(img, size_out, sitk.Transform(), interpolator,
                         img.GetOrigin(), spacing_out, img.GetDirection(),
                         default_value, img.GetPixelID())


def _run_inference(c: dict, model: str) -> Path:
    """nnU-Net по volume.mha -> mask_<model>.mha (геометрия входа).

    Для модели с 78 классами объём заранее приводится к целевому спейсингу
    модели. Иначе nnU-Net на возврате логитов к исходной геометрии кастует
    весь 78-канальный массив в float64 разом: на client4 (154 млн вокселей)
    это 96 ГБ и падение по памяти. Когда формы совпадают, resample_data_or_seg
    возвращает данные как есть — каста нет, пик втрое ниже. Маску возвращаем
    к исходной геометрии сами, ближайшим соседом: это uint8, стоит копейки.
    """
    orig_img = sitk.ReadImage(c["volume"])
    sitk_img = orig_img
    target = (_target_spacing(model)
              if model in MODEL_NEEDS_REMAP or model in SINGLE_PULP_MODELS
              else None)
    if target and not np.allclose(orig_img.GetSpacing(), target, atol=1e-4):
        sitk_img = _resample(orig_img, target, sitk.sitkLinear, -1000.0)
        print(f"CBCT: объём приведён к {target} -> {sitk_img.GetSize()}")
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)[None]  # (1,D,H,W)
    # Решаем ПО РАЗМЕРУ ЭТОГО тома, где собирать карту вероятностей: у одной
    # и той же модели один кейс помещается на GPU, а другой нет.
    on_device = _fits_on_gpu(model, arr.shape[1:])
    if not on_device:
        n_cls = (47 if model in SINGLE_PULP_MODELS
             else 78 if model in MODEL_NEEDS_REMAP else 49)
        print(f"CBCT: карта вероятностей {n_cls}x{arr.shape[1:]} не влезает "
              f"в видеопамять — собираем на CPU (медленнее, но без спилла)")
    predictor = get_predictor(model, on_device)
    props = {
        "spacing": sitk_img.GetSpacing()[::-1],
        "sitk_stuff": {
            "spacing": sitk_img.GetSpacing(),
            "origin": sitk_img.GetOrigin(),
            "direction": sitk_img.GetDirection(),
        },
    }
    seg = predictor.predict_single_npy_array(arr, props, None, None, False)
    seg = remap_prediction(seg, model, spacing=props["spacing"])
    # Сторону выправляем ДО split_paired_sides: тот раскладывает парные
    # структуры по средней линии и на уже верной маске ничего не меняет.
    seg = orient_sides(seg, sitk_img.GetDirection(), c)
    if model in MODEL_NEEDS_REMAP:
        seg = split_paired_sides(seg, sitk_img.GetDirection())
    out = sitk.GetImageFromArray(seg.astype(np.uint8))
    out.SetSpacing(sitk_img.GetSpacing())
    out.SetOrigin(sitk_img.GetOrigin())
    out.SetDirection(sitk_img.GetDirection())
    if sitk_img is not orig_img:      # вернуть маску на исходную сетку
        out = sitk.Resample(out, orig_img, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, out.GetPixelID())
    mpath = CASES_DIR / c["case_id"] / f"mask_{model}.mha"
    sitk.WriteImage(out, str(mpath))
    return mpath


# ================= статусный воркер: сегментация + мешинг =================

def start_segment(case_id: str, model: str) -> tuple[bool, str]:
    c = get_case(case_id)
    if c is None:
        return False, "Кейс не найден"
    # combo своей сети не имеет — она собирается из двух других (build_combo_mask)
    if model not in MODELS and model != "combo":
        return False, "Неизвестная модель"
    if c["stage"] in ("predicting", "meshing"):
        return True, "already_running"
    c["model"] = model
    t = threading.Thread(target=_segment_worker, args=(c, model), daemon=True)
    t.start()
    return True, "ok"


def build_combo_mask(c: dict) -> Path:
    """Маска из двух моделей: анатомия и зубы от одной, пульпа от другой.

    Правило простое и проверяемое: пульпа принимается, только если её зуб есть
    у модели-донора анатомии И она действительно лежит внутри него. Иначе
    получилась бы пульпа без зуба или пульпа в чужом зубе — врач такое увидит
    сразу, а доверие к слою находок держится на том, что пульпа принадлежит
    именно этому зубу (эндодонтический статус считается «пульпа минус дентин
    ТОГО ЖЕ зуба»).

    Резцовый и язычный каналы (103-105) тоже берём у ToothFairy3: у модели
    ToothFairy2 таких классов нет вовсе.
    """
    from scipy import ndimage as ndi
    a_model, p_model = COMBO_SOURCES["anatomy"], COMBO_SOURCES["pulp"]
    for m in (a_model, p_model):
        if not (c["masks"].get(m) and Path(c["masks"][m]).is_file()):
            raise RuntimeError(f"для сводной модели нужна маска {m}")
    a_img = sitk.ReadImage(c["masks"][a_model])
    p_img = sitk.ReadImage(c["masks"][p_model])
    if a_img.GetSize() != p_img.GetSize():
        raise RuntimeError("маски разной геометрии, склеить нельзя")
    a = sitk.GetArrayFromImage(a_img).astype(np.int16)
    p = sitk.GetArrayFromImage(p_img).astype(np.int16)
    spacing = np.array(c["spacing"], float)

    out = a.copy()                      # анатомия и зубы — как есть
    kept = dropped = 0
    for t in TEETH_CLASSES:
        pm = p == (t + 100)
        if not pm.any():
            continue
        tooth = a == t
        if not tooth.any():
            dropped += 1                # зуба у донора анатомии нет
            continue
        # Пульпа обязана быть ВНУТРИ своего зуба: то же правило и тот же
        # порог, что в _filter_pulp, только зуб берётся из другой модели.
        dist = ndi.distance_transform_edt(~tooth, sampling=spacing)
        if float(np.median(dist[pm])) <= COMBO_PULP_DIST_MM:
            out[pm & (out == t)] = t + 100
            kept += 1
        else:
            dropped += 1
    for cls in (103, 104, 105):          # каналы, которых нет у ToothFairy2
        sel = p == cls
        if sel.any():
            out[sel & (out == 0)] = cls

    path = CASES_DIR / c["case_id"] / "mask_combo.mha"
    path.parent.mkdir(parents=True, exist_ok=True)
    res = sitk.GetImageFromArray(out)
    res.CopyInformation(a_img)
    sitk.WriteImage(res, str(path), useCompression=True)
    print(f"CBCT: сводная маска — анатомия {a_model}, пульпа {p_model}: "
          f"принято {kept} пульп, отклонено {dropped}")
    return path


def _segment_worker(c: dict, model: str) -> None:
    try:
        mpath = c["masks"].get(model)
        if mpath and Path(mpath).is_file():
            c["stage"] = "meshing"
            c["detail"] = "Готовая маска, строим 3D…"
        elif model == "combo":
            # Своей сети нет: досчитываем недостающие маски-источники и клеим.
            for src in (COMBO_SOURCES["anatomy"], COMBO_SOURCES["pulp"]):
                if not (c["masks"].get(src) and Path(c["masks"][src]).is_file()):
                    c["stage"] = "predicting"
                    c["detail"] = f"Сегментация ({MODEL_LABELS.get(src, src)})…"
                    c["masks"][src] = str(_run_inference(c, src))
            c["stage"] = "predicting"
            c["detail"] = "Сведение двух моделей…"
            mpath = str(build_combo_mask(c))
            c["masks"][model] = mpath
        else:
            c["stage"] = "predicting"
            c["detail"] = ("Загрузка модели…"
                           if model not in _predictors else "Сегментация…")
            mpath = str(_run_inference(c, model))
            c["masks"][model] = mpath
        # всё для этой модели уже посчитано — открываем мгновенно
        # (список примеров обещает «без ожидания сегментации»)
        glb = case_dir(c) / f"mesh_{model}.glb"
        glb_ok = (glb.is_file()
                  and (c.get("meshes") or {}).get(model) == POSTPROC_VERSION)
        info, found = c.get("info") or {}, c.get("findings") or {}
        if (glb_ok and info.get("model") == model
                and not findings_stale(found, model)):
            c["stage"] = "done"
            c["detail"] = ""
            return

        # Меш на диске переживает переключение модели, а `info` и находки —
        # нет: они хранятся по одному комплекту на кейс. Раньше из-за этого
        # при возврате к уже посчитанной модели меш строился ЗАНОВО — четыре
        # минуты вместо сорока секунд на пересчёт находок. Готовый GLB этой
        # версии постпроцессинга валиден: устаревшие удаляет `_load_persisted`.
        c["stage"] = "meshing"
        if glb_ok:
            c["detail"] = "3D-модель готова, пересчёт находок…"
        else:
            c["detail"] = "Построение 3D-модели…"
            build_mesh(c, model)
        c["info"] = compute_info(c, model)
        c["detail"] = "Анализ находок…"
        c["findings"] = compute_findings(c, model)
        _save_meta(c)
        c["stage"] = "done"
        c["detail"] = ""
    except Exception as exc:  # noqa: BLE001 — показываем причину фронту
        c["stage"] = "error"
        c["detail"] = str(exc)


# ================= мешинг (marching cubes -> GLB) =================

# Тонкие корневые каналы размытие перед marching cubes стирало. Врач с
# приёма: «у зубов с 3-4 каналами они не дорисованы — где один, где ни
# одного». Замер на client5/sp (17 премоляров и моляров, число каналов в
# срезах на 55/66/77% длины зуба):
#
#     сырая маска модели              85 участков каналов
#     меш как был (sigma 0.9)         58  — треть каналов пропадала в меше
#     меш с сохранением тонкого       86
#
# Канал в 1-2 вокселя после sigma 0.9 не дотягивает до изоуровня 0.5.
# Страховка ниже (снижение sigma) смотрит на максимум ВСЕЙ пульпы, а камера
# толстая и переживает размытие — поэтому она не срабатывала. Толстое
# оставляем как было (форма камеры не меняется), а то, что размытие съело
# бы и что НЕ прилегает к выжившему телу, размываем мягче.
THIN_SIGMA = 0.45    # 1-вокс. линия: пик после размытия 0.54 > 0.5
THIN_GROUPS = ("pulp",)


def _keep_thin_parts(vol: np.ndarray, sm: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    ball = ndi.generate_binary_structure(3, 1)
    solid = vol > 0
    lost = solid & (sm < 0.5)
    if not lost.any():
        return sm
    # Потерянное вплотную к выжившему телу — это скруглённые углы камеры,
    # их не возвращаем. Остальное — тонкие каналы.
    thin = lost & ~ndi.binary_dilation(sm >= 0.5, ball)
    # Короткий перешеек между двумя толстыми кусками правилу выше не
    # попадает (он вплотную к выжившему телу), а без него канал в 3D
    # отрывается от камеры. Скелет маски связность держит — его потерянные
    # воксели возвращаем всегда.
    from skimage.morphology import skeletonize
    thin |= lost & skeletonize(solid)
    if not thin.any():
        return sm
    # Прирастить на воксель внутри маски: канал не должен распасться на бусины
    # и обязан дотянуться до камеры.
    thin = (ndi.binary_dilation(thin, ball) & solid) | thin
    # Складываем, а не берём максимум: у короткого перешейка ни одно из
    # размытий в одиночку до изоуровня не дотягивает.
    return np.minimum(
        sm + ndi.gaussian_filter(thin.astype(np.float32), THIN_SIGMA), 1.0)


def _class_mesh(mask: np.ndarray, cls: int, spacing: tuple,
                cap: int, keep_thin: bool = False) -> "trimesh.Trimesh | None":
    """Меш одного класса: bbox-кроп (скорость) -> размытие маски ->
    marching cubes -> decimate до cap.

    Сглаживаем МАСКУ до marching cubes, а не меш после. Сглаживание меша
    через trimesh 5.0.0 давало те самые «шипы»: laplacian_calculation на
    упрощённом меше строит оператор со строками, сумма которых < 1 (замер:
    min 0.455 вместо 1.0), поэтому L·V − V смотрит в начало координат и
    утягивает вершину на десятки мм. На client5, класс 1: после decimate
    самое длинное ребро 10.8 мм, после filter_laplacian — 46.3 мм, 2003
    грани длиннее 10 мм и 1161 грань висит дальше 2 мм от маски. Тем же
    оператором пользуются filter_taubin и filter_humphrey — они ломают меш
    так же, поэтому сглаживания меша здесь нет вообще.

    Гауссово размытие бинарной маски даёт ту же гладкость без риска:
    поверхность остаётся на изоуровне 0.5, объём сохраняется (нерв 0.352 ->
    0.34 см³), ни одна грань не отходит от маски дальше 2 мм.
    """
    import trimesh
    from scipy import ndimage as ndi
    from skimage.measure import marching_cubes
    binmask = mask == cls
    if binmask.sum() < 30:
        return None
    # кроп по bounding box — marching cubes на полном объёме в 100× дороже
    idx = np.where(binmask)
    lo = [max(int(i.min()) - MESH_PAD, 0) for i in idx]
    hi = [int(i.max()) + MESH_PAD + 1 for i in idx]
    crop = binmask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # pad нулями закрывает меш, даже если структура упирается в край объёма
    vol = np.pad(crop, MESH_PAD).astype(np.float32)
    # Страховка на случай структуры тоньше самого размытия: если бы максимум
    # размытой маски опустился ниже изоуровня 0.5, marching_cubes упал бы с
    # «level вне диапазона» и уронил сборку всего кейса. Уменьшаем sigma,
    # пока структура переживает изоуровень; в пределе — без размытия.
    # Замер на client4 показал, что сегодня это не срабатывает ни разу:
    # у резцовых каналов (103/104) полутолщина 2.2-2.4 вокселя и максимум
    # после sigma 0.9 равен 0.96-0.98, у язычного (105) — 2.0 и 0.92,
    # у нерва — 4.9 и 1.00. То есть новые классы размытие переживают, и
    # ветка ниже стоит ради структур тоньше них, а не ради этих.
    sigma = MESH_BLUR_SIGMA
    sm = ndi.gaussian_filter(vol, sigma=sigma)
    while sm.max() < 0.6 and sigma > 0.15:
        sigma *= 0.5
        sm = ndi.gaussian_filter(vol, sigma=sigma)
    if keep_thin and sm.max() >= 0.6:
        sm = _keep_thin_parts(vol, sm)
    vol = sm if sm.max() >= 0.6 else vol
    verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=spacing)
    origin = (np.array(lo) - MESH_PAD) * np.array(spacing, dtype=np.float64)
    verts = verts + origin  # сдвиг кропа обратно в координаты объёма (мм)
    mesh = trimesh.Trimesh(verts, faces, process=False)

    if len(mesh.faces) > cap:
        import fast_simplification
        v, f = fast_simplification.simplify(
            mesh.vertices, mesh.faces,
            target_reduction=min(1.0 - cap / len(mesh.faces), 0.995))
        mesh = trimesh.Trimesh(v, f, process=False)
    if mesh.volume < 0:   # marching cubes отдаёт нормали внутрь, а фронт
        mesh.invert()     # рисует зубы/нерв с FrontSide — лицевые грани
    return mesh           # отсекались бы, поверхность светилась «наизнанку»


# Челюсть во вьюере должна выглядеть костью, а не решетом. HU-гейт нужен для
# ИЗМЕРЕНИЙ (иначе мягкие ткани сойдут за кость), но для МЕША он губителен:
# губчатая кость местами тоньше порога, гейт выгрызает её изнутри, и marching
# cubes честно обводит каждую полость. Замер на client5, нижняя челюсть:
#
#     с HU-гейтом (как было)     174 отдельные оболочки, площадь 345 см2
#     без гейта + заполнение       3 оболочки,            площадь 252 см2
#
# Те 174 куска врач и видел как «кость разваливается внутри челюсти».
# Гейт выгрызал 19% нижней челюсти и 38% верхней.
BONE_SHELL_GROUPS = ("mandible", "maxilla")


BONE_SHELL_CLOSE = 2          # итераций замыкания перед отбором компонент
BONE_SHELL_MIN_FRAC = 0.10    # компонента меньше этой доли крупнейшей — шум


def _bone_shell(bm: np.ndarray) -> np.ndarray:
    """Внешняя оболочка кости: замыкание, крупные куски, полости заполнены.

    Порядок важен. Замыкание идёт ДО отбора компонент: гейта здесь нет, но
    маска модели всё равно местами рвётся, и замыкание сшивает разрывы —
    иначе законная часть кости уходит в отдельный кусок и её приходится либо
    терять, либо тащить вместе с шумом.

    Компоненты отбираются по ДОЛЕ от крупнейшей, а не фиксированным числом:
    у верхней челюсти вторая по величине часть держит 15-17% объёма (замер на
    трёх кейсах) и выбрасывать её нельзя, а мелкие осколки — это шум.
    Замер на client5, нижняя челюсть, число оболочек в готовом меше:

        без обработки (как было)   174 оболочки, площадь 345 см2
        только крупнейшая           2 оболочки,  но теряет 24% объёма
        доля >= 10% + замыкание     2 оболочки,  объём сохранён
    """
    import cc3d
    from scipy import ndimage as ndi
    if not bm.any():
        return bm
    min_vox = CC_RULES["large"][0]
    # Работаем на кропе по bbox: замыкание и заполнение полостей по всему тому
    # (89 млн вокселей) занимают минуты, а челюсть занимает его малую часть.
    # Запас по краю нужен, чтобы замыкание не упёрлось в границу кропа.
    pad = 2 * BONE_SHELL_CLOSE + 2
    idx = np.where(bm)
    lo = [max(int(i.min()) - pad, 0) for i in idx]
    hi = [min(int(i.max()) + pad + 1, n) for i, n in zip(idx, bm.shape)]
    crop = bm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    closed = ndi.binary_closing(crop, ndi.generate_binary_structure(3, 1),
                                iterations=BONE_SHELL_CLOSE)
    lab = cc3d.connected_components(closed, connectivity=26)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    if not sizes.any():
        return bm
    keep = sizes >= max(min_vox, BONE_SHELL_MIN_FRAC * sizes.max())
    out = np.zeros_like(bm)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = ndi.binary_fill_holes(keep[lab])
    return out


def build_mesh(c: dict, model: str) -> Path:
    """Меши по группам -> один GLB (ноды: mandible, nerve, tooth_11…).
    Маска — фильтрованная (connected components против streak-артефактов)."""
    import trimesh
    mask_img = sitk.ReadImage(c["masks"][model])
    spacing = mask_img.GetSpacing()[::-1]            # (z,y,x)
    mask = filtered_mask_for(c, model)               # (D,H,W) uint8, CC-фильтр
    raw_mask = None                                  # сырая — только для оболочки кости

    scene = trimesh.Scene()
    for g in GROUPS:
        if g["key"] == "teeth":
            for cls in g["classes"]:
                m = _class_mesh(mask, cls, spacing, FACE_CAP["tooth"])
                if m is not None:
                    scene.add_geometry(m, geom_name=f"tooth_{cls}")
        else:
            meshes = []
            for cls in g["classes"]:
                src = mask
                if g["key"] in BONE_SHELL_GROUPS:
                    if raw_mask is None:
                        # ЗЕРКАЛО АППАРАТА ОБЯЗАТЕЛЬНО И ЗДЕСЬ.
                        #
                        # Зубы берутся из `filtered_mask_for`, то есть проходят
                        # `_arrays_for` -> `apply_scanner_mirror`. Кость же
                        # читалась с диска отдельным путём, и на аппаратах с
                        # `mirrored: true` (оба в клинике) она оставалась
                        # ЗЕРКАЛЬНОЙ относительно зубов. Врач с приёма это и
                        # видел: «зубы сместились с кости», дыры в лунках,
                        # пульпы будто снаружи — хотя каждая пульпа честно
                        # лежит внутри своего зуба, а разъехались зубы с
                        # челюстью.
                        #
                        # Тот же класс ошибки, что уже ловили дважды: часть
                        # данных прошла преобразование, часть нет. Правило —
                        # любая маска, попадающая в один кадр с зубами, обязана
                        # пройти ТУ ЖЕ цепочку.
                        _, raw_mask = apply_scanner_mirror(
                            None, sitk.GetArrayFromImage(mask_img), c)
                        raw_mask = orient_sides(
                            raw_mask, mask_img.GetDirection(), c)
                    src = np.where(_bone_shell(raw_mask == cls),
                                   cls, 0).astype(np.uint8)
                m = _class_mesh(src, cls, spacing,
                                FACE_CAP[g["key"]] // max(len(g["classes"]), 1),
                                keep_thin=g["key"] in THIN_GROUPS)
                if m is not None:
                    meshes.append(m)
            if meshes:
                combined = (meshes[0] if len(meshes) == 1
                            else trimesh.util.concatenate(meshes))
                scene.add_geometry(combined, geom_name=g["key"])

    # координаты (z,y,x) мм -> three.js: (x, z, y) — ось срезов смотрит вверх
    for geom in scene.geometry.values():
        geom.vertices = geom.vertices[:, [2, 0,1 ]]

    gpath = CASES_DIR / c["case_id"] / f"mesh_{model}.glb"
    scene.export(str(gpath))
    # Версию меша записываем на диск СРАЗУ после экспорта, не откладывая до
    # конца пересборки. Между export и _save_meta лежало окно в несколько
    # минут (находки, info), и всё это время в meta.json стояла ПРОШЛАЯ
    # версия: любой другой процесс, импортирующий этот модуль, видел «меш
    # устарел» и удалял только что собранный файл. Так пропали mesh_ours.glb
    # и mesh_pretrained.glb у client3 — сборка падала с «не найден файл», и
    # выглядело это как сбой сборки, хотя виноват был соседний импорт.
    # Отмечаем, по каким правилам собран ИМЕННО этот меш: `info` один на кейс,
    # а мешей несколько, и по нему их версии не различить.
    c.setdefault("meshes", {})[model] = POSTPROC_VERSION
    # Правило сторон живёт в данных (scanner_sides.json), а не в коде, поэтому
    # инвалидация по POSTPROC_VERSION его НЕ ловит: врач подтвердил конвенцию
    # аппарата — номера зубов поменялись, а версия осталась прежней, и демо
    # отдало бы старые меши. Запоминаем применённое правило рядом с версией.
    c["sides_applied"] = bool(scanner_sides_state(c).get("mirrored"))
    _save_meta(c)          # закрываем окно: на диске версия совпадает с мешем
    return gpath


# ================= инфо по кейсу =================

def occlusal_up(mask: np.ndarray) -> bool:
    """Растёт ли ось срезов вверх (верхние зубы имеют больший z).

    Определять ПО ДАННЫМ, а не предполагать. Наши четыре клинических кейса
    сняты одним аппаратом и лежат одинаково, а на ToothFairy ориентация
    обратная — измерение уровня кости из-за этого разваливалось полностью
    (0 зубов из 22 против 18 из 28 у нас): ось зуба смотрела от коронки к
    апексу, и перепад плотности эмаль-дентин получался отрицательным.
    """
    zu = [np.where(mask == t)[0].mean() for t in TEETH_CLASSES
          if t // 10 in (1, 2) and (mask == t).sum() >= 300]
    zl = [np.where(mask == t)[0].mean() for t in TEETH_CLASSES
          if t // 10 in (3, 4) and (mask == t).sum() >= 300]
    if not zu or not zl:
        return True
    return float(np.mean(zu)) > float(np.mean(zl))


def compute_findings(c: dict, model: str) -> dict | None:
    """Слой находок (cbct_findings). Импорт локальный — модуль сам импортирует
    этот, а сбой находок не должен ронять уже построенный меш."""
    try:
        import cbct_findings
        res = cbct_findings.analyze_case(c, model)
        if res is not None:
            res["v"] = FINDINGS_VERSION
        return res
    except Exception as exc:  # noqa: BLE001
        print(f"CBCT: находки не посчитаны для {c['case_id']}: {exc}")
        return None


def findings_stale(found: dict | None, model: str) -> bool:
    """Нужно ли пересчитать находки: нет их, чужая модель или старая версия."""
    return (not found or found.get("model") != model
            or found.get("v") != FINDINGS_VERSION)


def findings_for(c: dict, model: str) -> dict | None:
    """Находки кейса: из meta, иначе считаем и сохраняем.

    Сверяем МОДЕЛЬ и ВЕРСИЮ: раньше отдавалось что лежит в meta, поэтому
    находки, посчитанные другой моделью или прошлой версией правил, молча
    доживали до врача.
    """
    if findings_stale(c.get("findings"), model):
        res = compute_findings(c, model)
        if res is None:
            return None
        c["findings"] = res
        _save_meta(c)
    return c["findings"]


def compute_info(c: dict, model: str) -> dict:
    mask = filtered_mask_for(c, model)  # воксельные счётчики — после CC-фильтра
    ids, counts = np.unique(mask, return_counts=True)
    vc = dict(zip(ids.tolist(), counts.tolist()))

    groups = []
    for g in GROUPS:
        n = sum(vc.get(cls, 0) for cls in g["classes"])
        groups.append({"group": g["key"], "label_ru": g["label"],
                       "color": g["color"], "opacity": g["opacity"],
                       "voxel_count": int(n)})
    teeth = sorted(cls for cls in TEETH_CLASSES if vc.get(cls, 0) > 100)
    return {
        "name": c["name"],
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "shape": c["shape"],
        "spacing": [round(s, 4) for s in c["spacing"]],
        "groups": groups,
        "teeth_present": teeth,
        "nerve_left": vc.get(3, 0) > 100,
        "nerve_right": vc.get(4, 0) > 100,
        # Куда растёт ось срезов: по этому признаку фронт разворачивает
        # перекрестие на корональном и сагиттальном видах — там срез уже
        # развёрнут сервером (см. render_slice_png).
        "z_up": bool(occlusal_up(mask)),
        "postproc": POSTPROC_VERSION,
    }


# ================= срезы с оверлеем =================

def _slice_color_lut() -> np.ndarray:
    """LUT классов -> RGB для оверлея на срезах (с запасом под пульпу 111-148)."""
    lut = np.zeros((MAX_CLASS, 3), dtype=np.float32)
    for g in GROUPS:
        rgb = np.array([int(g["color"][i:i + 2], 16) for i in (1, 3, 5)],
                       dtype=np.float32)
        if g["key"] == "teeth":
            for cls in g["classes"]:
                qc = QUADRANT_COLORS[cls // 10]
                lut[cls] = [int(qc[i:i + 2], 16) for i in (1, 3, 5)]
        else:
            for cls in g["classes"]:
                lut[cls] = rgb
    return lut


_LUT = _slice_color_lut()


def _arrays_for(c: dict, model: str, canals: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """vol (int16) + ФИЛЬТРОВАННАЯ mask (uint8), кэш одного кейса в памяти.
    Сырая маска на диске не трогается — фильтрация только для рендера.

    `canals=False` отдаёт маску БЕЗ дотрассировки каналов по плотности: нужно
    для сверки трассировки с моделью канала (`cbct_canal_apply.py`), которой
    иначе достался бы уже дорисованный канал.
    """
    key = f'{c["case_id"]}:{model}:{int(canals)}'
    if _arr_cache["key"] != key:
        vol = sitk.GetArrayFromImage(sitk.ReadImage(c["volume"]))
        mimg = sitk.ReadImage(c["masks"][model])
        mask = sitk.GetArrayFromImage(mimg)
        # Сторону выправляем ЗДЕСЬ, а не только при инференсе: маски, уже
        # лежащие на диске, посчитаны до этой правки и зеркальны. Проверка
        # самокорректирующаяся — на верной маске не меняет ничего.
        vol, mask = apply_scanner_mirror(vol, mask, c)
        mask = orient_sides(mask, mimg.GetDirection(), c)
        mask = filter_mask(mask, vol.astype(np.float32, copy=False),
                           spacing=c["spacing"], canals=canals)
        _arr_cache.update(key=key, vol=vol, mask=mask)
    return _arr_cache["vol"], _arr_cache["mask"]


def panorama_for(c: dict, model: str) -> tuple:
    """Панорама кейса: с диска, иначе строим и кладём рядом.

    На запрос не считаем: развёртка идёт по всей высоте тома в 25 столбиков
    поперёк дуги и занимает десятки секунд — врач ждать не должен.
    """
    png_path = CASES_DIR / c["case_id"] / f"pano_{model}.png"
    meta = c.get("panorama")
    if png_path.is_file() and meta and meta.get("model") == model:
        return png_path.read_bytes(), meta
    import cbct_panorama
    png, meta = cbct_panorama.build(c, model)
    meta["model"] = model
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(png)
    c["panorama"] = meta
    _save_meta(c)
    return png, meta


CONTOUR_ALPHA = 0.9       # обводка структур в фокусе
CONTOUR_ALPHA_BG = 0.55   # всё остальное приглушено


def _label_edges(m: np.ndarray) -> np.ndarray:
    """Пиксели границы: метка отличается от соседа. Фон границей не считается."""
    e = np.zeros(m.shape, dtype=bool)
    d = m[:-1] != m[1:]
    e[:-1] |= d
    e[1:] |= d
    d = m[:, :-1] != m[:, 1:]
    e[:, :-1] |= d
    e[:, 1:] |= d
    return e & (m > 0)


def overlay_fill(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Полупрозрачная заливка классов — прежний режим срезов."""
    colored = m > 0
    if not colored.any():
        return rgb
    overlay = _LUT[np.clip(m, 0, MAX_CLASS - 1)]
    out = rgb.astype(np.float32)
    out[colored] = out[colored] * (1 - OVERLAY_ALPHA) + \
        overlay[colored] * OVERLAY_ALPHA
    return out.astype(np.uint8)


def overlay_contour(rgb: np.ndarray, m: np.ndarray, focus=()) -> np.ndarray:
    """Контуры структур вместо заливки.

    Заливка, даже полупрозрачная, скрывает саму плотность — а врач читает
    снимок именно по ней: кариес под пломбой, границу очага, рисунок губчатой
    кости. Обводка оставляет пиксели нетронутыми и при этом показывает, что
    именно сеть считает зубом. `focus` (свой зуб и его пульпа) обводится ярче
    и толще, остальное приглушено, иначе на карточке рябит от соседей.
    """
    from scipy import ndimage as ndi
    m = np.asarray(m)
    edges = _label_edges(m)
    if not edges.any():
        return rgb
    out = rgb.astype(np.float32)
    focus = tuple(int(f) for f in focus)
    rest = edges & ~np.isin(m, focus) if focus else edges
    if rest.any():
        col = _LUT[np.clip(m, 0, MAX_CLASS - 1)]
        out[rest] = out[rest] * (1 - CONTOUR_ALPHA_BG) + \
            col[rest] * CONTOUR_ALPHA_BG
    for cls in focus:
        fc = edges & (m == cls)
        if not fc.any():
            continue
        fc = ndi.binary_dilation(fc)      # 2 px: тонкая линия теряется на шуме
        rgbc = _LUT[min(cls, MAX_CLASS - 1)]
        out[fc] = out[fc] * (1 - CONTOUR_ALPHA) + rgbc * CONTOUR_ALPHA
    return out.astype(np.uint8)


# ================= проверка находок врачом и подпись =================
# Находка — это черновик до тех пор, пока её не посмотрел человек. Отметка
# врача живёт в meta.json рядом с самими находками и переживает перезапуск.
REVIEW_STATES = ("confirmed", "rejected")


def finding_key(f: dict) -> str:
    """Устойчивый ключ находки: код плюс зубы, к которым она относится.

    Порядкового номера в списке недостаточно: находки пересчитываются, список
    меняет длину, и отметка врача уехала бы на чужую строку.
    """
    if f.get("tooth"):
        return f'{f["code"]}:{f["tooth"]}'
    teeth = f.get("teeth") or []
    if teeth:
        return f'{f["code"]}:' + "-".join(str(t) for t in sorted(teeth))
    # Находка без зубов (качество маски: «канал слева не сегментирован»).
    # Одного кода мало — таких у одного кейса бывает несколько, и отметка
    # врача на одной легла бы сразу на все. Различаем по тексту.
    import zlib
    h = zlib.crc32(str(f.get("text", "")).encode("utf-8")) & 0xffffff
    return f'{f["code"]}:{h:06x}'


def review_for(c: dict) -> dict:
    r = c.get("review") or {}
    return {"items": r.get("items") or {}, "signed": r.get("signed")}


def set_review(c: dict, key: str, state: str | None, by: str = "") -> dict:
    """Отметить находку подтверждённой или отклонённой. state=None — снять.

    ПОДПИСЬ ПРИ ЭТОМ СНИМАЕТСЯ. Подписанный план не имеет права молча
    измениться под подписью: если врач меняет решение по находке, план
    пересобирается и подписывать его надо заново.
    """
    r = review_for(c)
    if state is None:
        r["items"].pop(key, None)
    else:
        r["items"][key] = {"state": state, "by": by,
                           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    r["signed"] = None
    c["review"] = r
    _save_meta(c)
    return r


def sign_case(c: dict, by: str, note: str = "") -> dict:
    """Подпись врача под текущим набором находок.

    Версия находок пишется в подпись: правила ещё меняются, и подпись,
    поставленная под прошлой версией, к новому набору не относится. Без этого
    поля отличить одно от другого было бы нечем.
    """
    r = review_for(c)
    r["signed"] = {"by": by, "note": note,
                   "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "findings_version": FINDINGS_VERSION}
    c["review"] = r
    _save_meta(c)
    return r


def apply_review(res: dict, rev: dict, drop_rejected: bool = True) -> dict:
    """Наложить отметки врача на набор находок.

    `drop_rejected` — для ПЛАНА: отклонённое врачом в план не идёт, в этом
    весь смысл проверки. В ленте находок, наоборот, показываем всё, включая
    отклонённое: врач должен видеть, что он уже отсмотрел.

    Копии, а не правка на месте: `res` лежит в кэше кейса и уходит в
    meta.json — состояние проверки внутри самих находок там оказаться не
    должно, иначе оно поедет вместе с кэшем и переживёт пересчёт.
    """
    items = rev.get("items") or {}

    def tag(f):
        # Ключ отдаём фронту готовым: если бы он считал его сам, две реализации
        # разъехались бы, и отметка врача легла бы не на ту находку.
        k = finding_key(f)
        return dict(f, key=k, review=(items.get(k) or {}).get("state"))

    tagged = [tag(f) for f in (res.get("findings") or [])]
    kept = [f for f in tagged if f["review"] != "rejected"] \
        if drop_rejected else tagged
    out = dict(res)
    out["findings"] = kept
    out["quality"] = [tag(f) for f in (res.get("quality") or [])]
    out["review"] = {
        "items": items, "signed": rev.get("signed"),
        "rejected": sum(1 for f in tagged if f["review"] == "rejected"),
        "confirmed": sum(1 for f in tagged if f["review"] == "confirmed"),
        "total": len(tagged),
        "findings_version": FINDINGS_VERSION,
    }
    return out


def tooth_card(c: dict, model: str, fdi: int) -> dict | None:
    """Геометрия зуба для карточки. None — зуба нет в маске."""
    import cbct_tooth
    vol, mask = _arrays_for(c, model)
    spacing = np.array(c["spacing"], float)
    return cbct_tooth.card(vol, mask, spacing, fdi, f'{c["case_id"]}:{model}')


def tooth_slice_png(c: dict, model: str, fdi: int, plane: str,
                    contour: bool = True, offset_mm: float = 0.0) -> tuple:
    """PNG одной плоскости зуба + её метаданные."""
    import cbct_tooth
    vol, mask = _arrays_for(c, model)
    spacing = np.array(c["spacing"], float)
    return cbct_tooth.render(vol, mask, spacing, fdi, plane,
                             f'{c["case_id"]}:{model}', contour=contour,
                             offset_mm=offset_mm)


def teeth_present(c: dict, model: str) -> list:
    """Зубы, для которых карточка строится (маска не огрызок)."""
    import cbct_tooth
    _vol, mask = _arrays_for(c, model)
    spacing = np.array(c["spacing"], float)
    return cbct_tooth.present(mask, spacing, f'{c["case_id"]}:{model}')


_zup_cache: dict[str, bool] = {}


def zup_for(c: dict, model: str) -> bool:
    """Растёт ли ось срезов вверх. Считается один раз на кейс: проход по маске.

    Нужно и срезам (иначе корональный и сагиттальный показываются вверх
    ногами), и фронту — он по этому же признаку разворачивает перекрестие.
    """
    key = f'{c["case_id"]}:{model}'
    if key not in _zup_cache:
        _zup_cache[key] = bool(occlusal_up(filtered_mask_for(c, model)))
    return _zup_cache[key]


def render_slice_png(c: dict, model: str, axis: str, idx: int,
                     overlay: str = "contour") -> bytes:
    """PNG среза: grayscale (окно HU_MIN..HU_MAX) + маска контуром или заливкой.

    Корональный и сагиттальный разворачиваются по вертикали, если ось z
    растёт вверх: строка 0 рисуется СВЕРХУ, то есть без разворота пациент
    оказывается вверх ногами. Ровно та же ошибка была на панораме.
    """
    import io
    from PIL import Image
    vol, mask = _arrays_for(c, model)
    D, H, W = vol.shape
    if axis == "z":
        idx = min(max(idx, 0), D - 1)
        v, m = vol[idx], mask[idx]
    elif axis == "y":
        idx = min(max(idx, 0), H - 1)
        v, m = vol[:, idx, :], mask[:, idx, :]
    else:
        idx = min(max(idx, 0), W - 1)
        v, m = vol[:, :, idx], mask[:, :, idx]
    if axis in ("y", "x") and zup_for(c, model):
        v, m = v[::-1], m[::-1]

    g = np.clip((v.astype(np.float32) - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)
    rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=-1)
    if overlay == "fill":
        rgb = overlay_fill(rgb, m)
    elif overlay != "none":
        rgb = overlay_contour(rgb, m)

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def mask_model_for(c: dict) -> str | None:
    """Какую маску показывать: последняя сегментированная, иначе любая готовая."""
    if c["model"] and c["masks"].get(c["model"]):
        return c["model"]
    for m in ("pretrained", "ours"):
        if c["masks"].get(m):
            return m
    return None


_load_persisted()
register_examples()
# Фикстуры последними: настоящий кейс с тем же номером всегда перевешивает.
register_fixtures()
