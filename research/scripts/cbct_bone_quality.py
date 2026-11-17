"""Качество кости в позиции импланта: BV/TV и кортикальная пластинка.

    .venv\Scripts\python.exe research\scripts\cbct_bone_quality.py client5 --model sp

ПОЧЕМУ НЕ D1-D4 ПО HU. Врач просил «распознавать тип кости d1, d2, d3, d4».
Классификация Миша стоит на единицах Хаунсфилда, которых КЛКТ не даёт:
систематический обзор отобрал 3 работы из 2710 и заключил, что переводить
серые значения в HU нельзя; после перекалибровки ошибка на 13 аппаратах
составляет 35..1562 единиц — шире, чем весь диапазон D1-D4 целиком. Значение
плывёт от поля зрения, укладки, металла ВНЕ кадра и даже от программы
просмотра (один и тот же скан в четырёх программах даёт разные HU). У
Diagnocat такой находки нет вовсе — ни HU, ни Misch, ни Lekholm.

ЧТО ВМЕСТО. Доля костной ткани в объёме (BV/TV) предсказывает первичную
стабильность импланта ЛУЧШЕ плотности: R² 0.61 против 0.39, обгоняя и момент
введения (0.50), и ISQ (0.38). И, что решает дело для нас, она переносима:
при переходе микро-КТ -> КЛКТ BV/TV смещается на +5%, площадь кортикала на
-0.3%, и обе не зависят от напряжения съёмки (p = 0.79..0.99), тогда как
толщина трабекул уезжает на +65%, а расстояние между ними на +43..54%.
Поэтому берём только первые две, а Tb.Th и Tb.Sp не считаем вовсе.

ПОРОГ КОСТИ — ОТ ДЕНТИНА ЭТОГО ЖЕ СКАНА, а не абсолютный. Тот же приём, что
в `cbct_endo`: некалиброванность шкалы сокращается при сравнении внутри
одного тома. Дентин выбран опорой потому, что он есть у каждого пациента и
из плотных тканей самый стабильный.

ЧТО ЭТО НЕ ЕСТЬ. Не диагноз и не класс кости. Отдаём измеренную долю и место
в собственной выборке по зоне челюсти — по той же причине, по которой
пародонт отдаёт перцентиль, а не стадию: разметки, на которой можно было бы
откалибровать клинический класс, у нас нет.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Доля дентинной медианы, выше которой воксель считается костью. Замер на
# client5: дентин 1900, порог 665 — между губчатой костью и мягкими тканями.
# Значение относительное, поэтому переживает разницу аппаратов.
DENTIN_FRAC = 0.35

# Радиус пробы: ложе импланта диаметром около 4 мм плюс стенки.
PROBE_R_MM = 6.0
# Насколько апикальнее центра зуба берётся проба — примерно середина тела
# импланта в этой позиции.
PROBE_DEPTH_MM = 6.0
# Меньше этого числа вокселей челюсти в пробе — точка вне кости, не считаем.
PROBE_MIN_JAW_VOX = 300

# Собственная база по зонам челюсти (client4 + client5, модель sp).
#
# Порядок воспроизводит анатомию, и это главная проверка метода: передний
# отдел нижней челюсти плотнее всего, задний отдел верхней рыхлее всего —
# ровно тот градиент, который Миш описывает качественно, а мы получили
# измерением. На client4 спад виден и внутри ряда: 11 даёт 0.89, 18 — 0.65,
# 21 — 0.89, 28 — 0.56.
#
# Проверено, что низкие значения у верхних моляров — НЕ пазуха: доля воздуха
# в пробе 0.00 у всех шестнадцати верхних зубов client4, то есть проба лежит
# в кости, а не в полости. Разница идёт за счёт «промежуточной» ткани между
# воздухом и порогом кости (0.11 у резцов против 0.57 у 26) — это и есть
# рыхлая губчатая кость.
#
# ЭТО ОРИЕНТИР НА ДВУХ КЕЙСАХ, А НЕ НОРМА. Межпациентная разница заметная:
# в боковом отделе верхней челюсти медиана client5 0.78 против 0.665 у
# client4. Поэтому словесная оценка называется от нашей выборки, а не
# клинически, и расширять базу надо по мере накопления кейсов.
ZONE_REF = {
    "lower_front": 0.95,     # client5; client4 в этой зоне не мерялся
    "upper_front": 0.86,     # client5 0.88, client4 0.845
    "lower_post": 0.83,      # client5
    "upper_post": 0.72,      # client5 0.78, client4 0.665
}
ZONE_REF_CASES = 2
ZONE_LABEL = {
    "lower_front": "передний отдел нижней челюсти",
    "upper_front": "передний отдел верхней челюсти",
    "lower_post": "боковой отдел нижней челюсти",
    "upper_post": "боковой отдел верхней челюсти",
}


def zone_of(fdi: int) -> str:
    """Зона челюсти по номеру зуба. Плотность кости различается между ними
    сильнее, чем между пациентами, — сравнивать надо внутри зоны."""
    upper = (fdi // 10) in (1, 2)
    front = (fdi % 10) <= 3
    return f'{"upper" if upper else "lower"}_{"front" if front else "post"}'


def dentin_threshold(vol: np.ndarray, mask: np.ndarray,
                     teeth_classes) -> float | None:
    """Порог «это кость» от медианы дентина того же скана."""
    sel = np.isin(mask, list(teeth_classes))
    if int(sel.sum()) < 2000:
        return None
    return float(np.median(vol[sel])) * DENTIN_FRAC


def bone_fraction(vol: np.ndarray, jaw: np.ndarray, spacing: np.ndarray,
                  point_vox: np.ndarray, thr: float,
                  radius_mm: float = PROBE_R_MM) -> dict | None:
    """BV/TV в шаре вокруг точки: доля костных вокселей среди вокселей челюсти.

    Знаменатель — челюсть по маске, а не весь шар: иначе результат зависел бы
    от того, сколько воздуха и мягких тканей попало в пробу, то есть от
    положения точки, а не от кости.
    """
    rad = np.maximum(np.round(radius_mm / spacing).astype(int), 1)
    lo = np.maximum(point_vox - rad, 0)
    hi = np.minimum(point_vox + rad + 1, np.array(vol.shape))
    if np.any(hi <= lo):
        return None
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    box_jaw = jaw[sl]
    n_jaw = int(box_jaw.sum())
    if n_jaw < PROBE_MIN_JAW_VOX:
        return None
    bone = (vol[sl] >= thr) & box_jaw
    return {"bvtv": float(bone.sum()) / n_jaw,
            "jaw_vox": n_jaw,
            "volume_mm3": n_jaw * float(np.prod(spacing))}


def describe(bvtv: float, zone: str) -> str:
    """Словесная оценка ОТ СОБСТВЕННОЙ ВЫБОРКИ, а не клиническая.

    Сказать «это D3» значит утверждать про болезнь и про шкалу, которой у нас
    нет. Сказать «на 0.09 ниже, чем обычно в этой зоне у наших пациентов» —
    проверяемое утверждение о нашем измерении.
    """
    ref = ZONE_REF.get(zone)
    if ref is None:
        return f"доля костной ткани {bvtv:.2f}"
    d = bvtv - ref
    if d <= -0.10:
        how = "заметно ниже обычного для этой зоны"
    elif d <= -0.05:
        how = "ниже обычного для этой зоны"
    elif d >= 0.05:
        how = "выше обычного для этой зоны"
    else:
        how = "в пределах обычного для этой зоны"
    return (f"доля костной ткани {bvtv:.2f} — {how} "
            f"(ориентир {ref:.2f}, {ZONE_LABEL[zone]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    import cbct_backend as cb
    from cbct_perio import tooth_axis

    c = cb.get_case(args.case)
    if not c:
        raise SystemExit(f"кейса {args.case} нет")
    model = args.model or ("sp" if "sp" in c["masks"] else sorted(c["masks"])[0])
    vol, mask = cb._arrays_for(c, model)
    spacing = np.array(c["spacing"], dtype=float)
    teeth = [int(t) for t in np.unique(mask) if 11 <= t <= 48]
    thr = dentin_threshold(vol, mask, teeth)
    if thr is None:
        raise SystemExit("дентина в маске мало — порог не построить")
    jaw = np.isin(mask, [1, 2])
    z_up = cb.occlusal_up(mask)
    print(f"{args.case} / {model}: порог кости {thr:.0f} "
          f"({DENTIN_FRAC:.2f} медианы дентина)")
    for t in sorted(teeth):
        w = np.array(np.where(mask == t))
        if w.shape[1] < 200:
            continue
        ax, org = tooth_axis(w, spacing, upper=(t // 10) in (1, 2), z_up=z_up)
        pt = np.round((org - ax * PROBE_DEPTH_MM) / spacing).astype(int)
        r = bone_fraction(vol, jaw, spacing, pt, thr)
        if not r:
            continue
        print(f"  {t}: {describe(r['bvtv'], zone_of(t))}")


if __name__ == "__main__":
    main()
