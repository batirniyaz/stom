"""Датасет для обучения, где пульпа — ОДИН класс, а не тридцать два.

Зачем. Мы учили сеть различать пульпу каждого зуба отдельно: класс «пульпа
зуба 11», «пульпа зуба 12» и так далее — 32 класса. Замер по нашим же данным
показал, во что это обходится: пульпа одного зуба это медиана **665 вокселей**,
то есть **4% от объёма зуба** (17 149), а у десятой части пульп меньше 250
вокселей. Тридцать два микроскопических класса на томе в 172 млн вокселей.

Так задачу не ставят даже авторы датасета. Официальный `evaluation.py` челленджа
ToothFairy3 содержит `convert_pulp_labels()`, который схлопывает все классы
пульпы в ОДИН — и в предсказании, и в эталоне — перед подсчётом Dice.
Оценивается 46 классов, пульпа среди них одна. Победитель челленджа учит так же
и возвращает номер зуба геометрией уже после сети.

Здесь готовится именно такая разметка: 78 классов -> 47.
  * 1-45   анатомия, зубы, резцовые и язычный каналы — без изменений
  * 46     ВСЯ пульпа одним классом (было 32 отдельных)

Номер зуба возвращается на постпроцессинге (`assign_pulp_to_teeth`): связная
компонента пульпы -> в каком зубе она лежит -> его номер. Компонента, не
попавшая ни в один зуб, отбрасывается как ложная — это то же правило, что уже
работает в `_filter_pulp`, только применяется после сети, а не внутри неё.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_prepare_singlepulp.py --check
    .venv/Scripts/python.exe research/scripts/cbct_prepare_singlepulp.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

R3D = Path(__file__).resolve().parents[2] / "research_3d"
SRC = R3D / "nnUNet_raw/Dataset113_ToothFairy3"
DST = R3D / "nnUNet_raw/Dataset114_ToothFairy3_SinglePulp"

# ДВА РАЗНЫХ ПРОСТРАНСТВА НУМЕРАЦИИ, и их нельзя путать.
#
# 46 — id пульпы В ОБУЧАЮЩЕЙ разметке: nnU-Net требует классы подряд от нуля,
# и после схлопывания непульповых классов ровно 45, так что пульпе достаётся 46.
#
# 150 — id пульпы ПОСЛЕ обратного отображения в FDI, где зубы это 11-48.
# Здесь 46 брать НЕЛЬЗЯ: это номер настоящего зуба. Ошибка стоила круговой
# проверки — функция принимала всю пульпу за зуб 46 и возвращала 1 класс
# вместо 26. То же значение 150 использует официальный evaluation.py челленджа.
SINGLE_PULP_ID = 46          # в обучающей разметке (nnU-Net)
SINGLE_PULP_FDI = 150        # после remap в FDI-нумерацию


def build_mapping() -> tuple[dict, dict]:
    """Отображение «старый обучающий id -> новый» и словарь меток для nnU-Net."""
    lm = json.loads((SRC / "label_map.json").read_text(encoding="utf-8"))
    t2o = {int(k): v for k, v in lm["train_to_orig"].items()}
    ds = json.loads((SRC / "dataset.json").read_text(encoding="utf-8"))
    name_by_train = {v: k for k, v in ds["labels"].items()}

    mapping, labels = {0: 0}, {"background": 0}
    nxt = 1
    for tr in sorted(t2o):
        if tr == 0:
            continue
        orig = t2o[tr]
        if orig >= 111:                      # любая пульпа
            mapping[tr] = SINGLE_PULP_ID
            continue
        mapping[tr] = nxt
        labels[name_by_train.get(tr, f"class_{orig}")] = nxt
        nxt += 1
    if nxt != SINGLE_PULP_ID:
        raise RuntimeError(
            f"непульповых классов {nxt-1}, ожидалось {SINGLE_PULP_ID-1} — "
            f"проверьте label_map")
    labels["pulp"] = SINGLE_PULP_ID
    return mapping, labels


def convert(limit: int | None = None) -> None:
    mapping, labels = build_mapping()
    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    lut = np.zeros(max(mapping) + 1, dtype=np.uint8)
    for a, b in mapping.items():
        lut[a] = b

    files = sorted((SRC / "labelsTr").glob("*.nii.gz"))
    if limit:
        files = files[:limit]
    for i, f in enumerate(files, 1):
        stem = f.name[:-len(".nii.gz")]
        out_lab = DST / "labelsTr" / f.name
        img_src = SRC / "imagesTr" / f"{stem}_0000.nii.gz"
        if not img_src.is_file():
            img_src = SRC / "imagesTr" / f"{stem}_0000.mha"
        out_img = DST / "imagesTr" / img_src.name
        if not out_lab.is_file():
            img = sitk.ReadImage(str(f))
            a = sitk.GetArrayFromImage(img)
            b = lut[np.clip(a, 0, len(lut) - 1)]
            res = sitk.GetImageFromArray(b)
            res.CopyInformation(img)
            sitk.WriteImage(res, str(out_lab), useCompression=True)
        # Снимки не копируем, а связываем: 26 ГБ второй копией ни к чему.
        if not out_img.exists() and img_src.is_file():
            try:
                out_img.hardlink_to(img_src)
            except OSError:
                shutil.copy2(img_src, out_img)
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    meta = {
        "channel_names": {"0": "CBCT"},
        "labels": labels,
        "numTraining": len(files),
        "file_ending": ".nii.gz",
        # ТОЛЬКО ASCII: nnU-Net читает dataset.json без указания кодировки,
        # на Windows это cp1252, и кириллица роняет препроцессинг ещё до старта
        # (UnicodeDecodeError на verify_dataset_integrity).
        "description": "ToothFairy3 with ALL pulp classes merged into one "
                       "(id 46). Tooth number is restored geometrically after "
                       "inference (assign_pulp_to_teeth).",
    }
    (DST / "dataset.json").write_text(
        json.dumps(meta, ensure_ascii=True, indent=1), encoding="utf-8")
    print(f"\nготово: {len(files)} томов, классов {len(labels)} "
          f"(было 78), пульпа -> id {SINGLE_PULP_ID}")
    print(f"каталог: {DST}")


def check() -> None:
    """Показать, что получится, не трогая диск."""
    mapping, labels = build_mapping()
    pulp_src = [a for a, b in mapping.items() if b == SINGLE_PULP_ID]
    print(f"классов на выходе: {len(labels)} (было 78)")
    print(f"схлопнуто в пульпу: {len(pulp_src)} классов")
    print(f"пульпа получает id {SINGLE_PULP_ID}")
    print("\nпервые метки:")
    for k, v in list(labels.items())[:8]:
        print(f"  {v:>3}  {k}")
    print("  ...")
    for k, v in list(labels.items())[-3:]:
        print(f"  {v:>3}  {k}")


def assign_pulp_to_teeth(mask: np.ndarray, pulp_id: int = SINGLE_PULP_FDI,
                         max_dist_mm: float = 2.0, spacing=None) -> np.ndarray:
    """Вернуть пульпе номер зуба: каждый воксель получает номер БЛИЖАЙШЕГО зуба.

    Постпроцессинг после сети, обученной на одном классе пульпы.

    Присваиваем по вокселю, а не голосованием по связной компоненте. Круговая
    проверка на client4 показала, почему: при голосовании соседние пульпы
    сливаются в одну компоненту и всю её метят одним номером — из 26 пульп
    правильно вернулись 20, шесть (11, 16, 17, 28, 42, 43) потерялись, и вместо
    них появился один разросшийся зуб 46. По ближайшему зубу такого слипания не
    бывает: граница проходит там, где меняется ближайший сосед.

    Воксель пульпы дальше `max_dist_mm` от любого зуба отбрасывается как ложный
    — то же правило, что в `_filter_pulp`, только применяется после сети.
    """
    from scipy import ndimage as ndi
    out = mask.copy()
    pulp = mask == pulp_id
    if not pulp.any():
        return out
    out[pulp] = 0                       # снимаем общий класс
    # Строго по списку номеров зубов, а не по диапазону: диапазон 11-48
    # захватил бы и id пульпы, если тот попадёт внутрь (так и вышло с 46).
    teeth = np.isin(mask, cb.TEETH_CLASSES) & (mask != pulp_id)
    if not teeth.any():
        return out
    sp = tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0)
    # Одно преобразование на весь том: для каждого вокселя — расстояние до
    # ближайшего зуба и координаты этого зуба.
    dist, idx = ndi.distance_transform_edt(~teeth, sampling=sp,
                                           return_indices=True)
    sel = pulp & (dist <= max_dist_mm)
    if not sel.any():
        return out
    zz, yy, xx = idx[0][sel], idx[1][sel], idx[2][sel]
    out[sel] = mask[zz, yy, xx].astype(out.dtype) + 100
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="показать план преобразования и выйти")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.check:
        check()
        return
    convert(a.limit)


if __name__ == "__main__":
    main()
