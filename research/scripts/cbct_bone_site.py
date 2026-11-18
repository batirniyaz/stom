"""Кость под имплант: высота до нерва или пазухи и ширина гребня.

    !!! НЕ ДОВЕДЕНО. Числа недостоверны, врачу не показывать. !!!

Результаты замера: высота до нижнечелюстного канала выходит 31–64 мм
при клинических 12–18, ширина гребня упирается в радиус окна. За десять
итераций правок каждая исправленная ошибка вскрывала следующую — это признак
того, что подход неверен в основе, а не в деталях.

Что по дороге выяснено и остаётся верным (это не пропадёт при переделке):

  * Гребень НЕЛЬЗЯ искать по маске челюсти: она его не покрывает вовсе — на
    глубине 0–3 мм от десны в неё попадает 0% точек.
  * Порог «это кость» — из своей базы (кость 1770, мягкие ткани до 180),
    литературные 300 HU для этой шкалы не значат ничего.
  * На КЛКТ рот ЗАКРЫТ, и кость непрерывна от нижней челюсти к верхней:
    поиск вверх уезжает в чужую челюсть (замер: плотность 3162 там, где
    алгоритм ещё считал «своё»).
  * Канал идёт не под зубом: смещение 6.3 мм по горизонтали на позиции 46.
  * Пазухи у моделей ToothFairy3 (`sp`) размечены примерно на десятую часть
    объёма — мерить по ним высоту нельзя, нужна `pretrained`.
  * Позицию отсутствующего зуба надо брать по дуге ряда, а не экстраполяцией
    от соседа: иначе столбик встаёт над соседним зубом.

Чем это лечится (следующий заход). Мерить не в объёме, а НА СРЕЗЕ вдоль оси
будущего импланта — так же, как это делает врач линейкой: строить
поперечный срез в позиции дефекта (как в `cbct_tooth`), находить гребень и
контур кости на нём и проверять результат ГЛАЗАМИ на нескольких кейсах,
прежде чем считать по всем. Плюс сверка с эталонной разметкой ToothFairy, где
кость размечена полностью.


Зачем. Решение «можно ли поставить имплант» держится на двух числах: сколько
кости в высоту (не упрёмся ли в нижнечелюстной канал или дно пазухи) и
сколько в ширину (хватит ли на диаметр винта). Сейчас врач меряет их линейкой
по срезам вручную.

Что уже есть в проекте и что нет. Расстояние от КОРНЯ до канала и до пазухи
считает `cbct_findings` — но это про существующий зуб. Здесь другое: высота
кости в МЕСТЕ ДЕФЕКТА, то есть там, где зуба нет и планируется имплант. В
CLAUDE.md это прямо записано как несделанное («правильнее мерить высоту кости
от гребня до дна пазухи, RBH — не сделано»).

Где меряем. В промежутках зубного ряда: позиция отсутствующего зуба между
соседями. Восьмёрки пропускаем — их не имплантируют.

Гребень ищем ПО ПЛОТНОСТИ, а не по маске челюсти. Замер при работе над
толщиной мягких тканей показал, что маска не покрывает альвеолярный гребень
вовсе: на глубине 0–3 мм от десны в неё попадает 0% точек. Порог плотности
взят из своей базы того же замера (кость — медиана 1770, мягкие ткани — до
180), поэтому 500 стоит в разрыве между ними, а литературные 300 HU для этой
шкалы не значат ничего.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_bone_site.py --case client4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb        # noqa: E402

BONE_HU = 500.0            # своя база, см. докстринг
NERVE_CLASSES = (3, 4)
SINUS_CLASSES = (5, 6)
# Ниже этого маску пазух считаем непригодной для измерений (см. analyze).
# Анатомический объём пазухи — порядка 500 тыс. вокселей при 0.3 мм.
SINUS_MIN_VOX = 150_000

# Ширину гребня меряем НИЖЕ его верхушки: сам гребень часто заострён, и
# ширина «на самом верху» получилась бы околонулевой у любого пациента.
WIDTH_BELOW_MM = 3.0
WIDTH_SPAN_MM = 1.0        # усредняем по полоске такой высоты

# Классификация остаточной высоты кости (RBH) при планировании в боковом
# отделе верхней челюсти — общепринятая, используется для выбора методики
# синус-лифтинга. Приводим как ориентир, а не как назначение.
RBH_CLASSES = ((10.0, "A", "высоты достаточно"),
               (7.0, "B", "умеренный дефицит"),
               (4.0, "C", "выраженный дефицит"),
               (0.0, "D", "кости почти нет"))

# Ниже этого высота недостоверна: сравнимо с вокселем и с ошибкой сегментации.
MIN_TRUST_MM = 1.0


def rbh_class(h: float) -> tuple:
    for lo, letter, text in RBH_CLASSES:
        if h >= lo:
            return letter, text
    return "D", "кости почти нет"


def _neighbours(present: list, fdi: int) -> list:
    """Соседи по ряду в том же квадранте."""
    q, i = fdi // 10, fdi % 10
    return [t for t in present if t // 10 == q and abs(t % 10 - i) == 1]


def defect_sites(present: list) -> list:
    """Позиции, где зуба нет, а соседи есть. Восьмёрки не берём."""
    out = []
    for q in (1, 2, 3, 4):
        for i in range(1, 8):            # 8 исключена намеренно
            fdi = q * 10 + i
            if fdi in present:
                continue
            if len(_neighbours(present, fdi)) >= 1:
                out.append(fdi)
    return out


def analyze(case_id: str, model: str | None = None) -> dict:
    c = cb.get_case(case_id)
    if c is None:
        raise SystemExit(f"нет кейса {case_id}")
    model = model or cb.mask_model_for(c)
    if model is None:
        raise SystemExit(f"у кейса {case_id} нет готовой маски")
    vol, mask = cb.arrays_for(c, model)
    spacing = np.array(c["spacing"], float)          # (z, y, x)
    z_up = cb.occlusal_up(mask)

    present = [t for t in cb.TEETH_CLASSES if (mask == t).sum() >= 300]
    sites = defect_sites(present)
    # Кость = плотное, но НЕ зуб и не реставрация. Без этого «гребнем»
    # оказывалась коронка соседнего зуба, попавшая в окно, и высота выходила
    # 26 мм вместо клинических 10–20.
    hard = np.isin(mask, list(cb.TEETH_CLASSES)
                   + [t + 100 for t in cb.TEETH_CLASSES] + [8, 9, 10])
    bone = (vol >= BONE_HU) & ~hard
    nerve = np.isin(mask, NERVE_CLASSES)
    sinus = np.isin(mask, SINUS_CLASSES)

    # Пазухи у моделей, обученных на ToothFairy3, размечены примерно на
    # десятую часть объёма — это замерено в проекте отдельно (медиана 49 тыс.
    # вокселей против анатомических ~500 тыс.). Мерить по такой маске высоту
    # до дна пазухи нельзя: «дно» окажется где угодно. Проверяем и честно
    # отказываемся, а не выдаём число.
    sinus_vox = int(sinus.sum())
    sinus_ok = sinus_vox >= SINUS_MIN_VOX

    out = []
    for fdi in sites:
        r = _site(fdi, present, mask, bone, nerve, sinus if sinus_ok else None,
                  spacing, z_up)  # маска челюстей берётся из mask внутри
        if r:
            out.append(r)
    return {"case_id": case_id, "model": model,
            "present": present, "sites": out, "draft": True,
            "sinus_voxels": sinus_vox, "sinus_usable": sinus_ok}


def _tooth_centre(mask, t: int) -> np.ndarray:
    return np.array(np.nonzero(mask == t), float).mean(1)


def _row_index(fdi: int) -> int:
    """Сквозная координата вдоль ряда: 18→11→21→28 идут монотонно.

    Нужна, чтобы положение зуба на дуге можно было интерполировать по номеру:
    внутри квадранта номера растут к середине, а между квадрантами
    переключаются, и без сквозной шкалы порядок ломается на средней линии.
    """
    q, i = fdi // 10, fdi % 10
    return -i if q in (1, 4) else i


def _site_centre(fdi: int, present: list, mask, spacing) -> np.ndarray | None:
    """Где стоял бы отсутствующий зуб — ПО ДУГЕ ряда.

    Экстраполяция от соседа (первая версия) промахивалась: для позиции 46
    столбик вставал над зубом 47, и нерв оказывался в 12 мм в стороне. Дуга
    же знает, как идёт ряд, и позиция между соседями получается там, где она
    и была бы у зуба.

    Дуга берётся из `cbct_panorama`: там она уже строится с отсевом промахов
    сегментации (зуб, уехавший на другую сторону, иначе утащит кривую).
    """
    import cbct_panorama as pano
    try:
        _pts, kept, _bad = pano.arch_points(mask, spacing)
    except Exception:                       # noqa: BLE001
        kept = {}
    jaw_q = (1, 2) if fdi // 10 in (1, 2) else (3, 4)
    mine = {t: c for t, c in kept.items() if t // 10 in jaw_q}
    if len(mine) >= 4:
        # Положение вдоль ряда описываем сквозным индексом и интерполируем
        # координаты по нему: между соседями это ровно середина промежутка.
        idx = np.array([_row_index(t) for t in mine])
        order = np.argsort(idx)
        idx = idx[order]
        # arch_points отдаёт центры в МИЛЛИМЕТРАХ, порядок осей (z, y, x).
        cent = np.array([mine[t] for t in mine], float)[order]
        want = _row_index(fdi)
        if idx.min() <= want <= idx.max():
            pos = np.array([float(np.interp(want, idx, cent[:, k]))
                            for k in range(3)])
        else:
            # За краем ряда: продолжаем по шагу последних двух зубов.
            at_end = want > idx.max()
            base = cent[-1] if at_end else cent[0]
            step = (cent[-1] - cent[-2]) if at_end else (cent[0] - cent[1])
            k = abs(want - (idx.max() if at_end else idx.min()))
            pos = base + step * k
        return pos / spacing            # мм -> воксели

    # Дуги не хватило (мало зубов) — старый способ: середина между соседями.
    nb = sorted(_neighbours(present, fdi), key=lambda t: t % 10)
    if len(nb) >= 2:
        return (_tooth_centre(mask, nb[0]) + _tooth_centre(mask, nb[1])) / 2
    if len(nb) == 1:
        return _tooth_centre(mask, nb[0])
    return None


def _site_axis(fdi: int, present: list, mask, spacing, z_up) -> np.ndarray | None:
    """Ось будущего импланта — средняя ось соседних зубов.

    Мерить вдоль оси тома нельзя, и это выяснилось замером: высота до канала
    выходила 23–26 мм при клинических 12–18. Зубы наклонены, гребень идёт
    вдоль них, и имплант ставят по оси соседей, а не по оси файла. Ровно
    поэтому в `cbct_perio` уровень кости считается вдоль СОБСТВЕННОЙ оси
    зуба — здесь та же причина.
    """
    import cbct_perio as perio
    upper = fdi // 10 in (1, 2)
    axes = []
    for t in _neighbours(present, fdi):
        idx = np.array(np.nonzero(mask == t))
        if idx.shape[1] < 300:
            continue
        try:
            ax, _origin = perio.tooth_axis(idx, spacing, t // 10 in (1, 2), z_up)
        except Exception:                       # noqa: BLE001
            continue
        axes.append(ax)
    if not axes:
        return None
    ax = np.mean(axes, axis=0)
    n = np.linalg.norm(ax)
    return ax / n if n > 1e-6 else None


def _site(fdi: int, present: list, mask, bone, nerve, sinus, spacing, z_up) -> dict | None:
    """Высота и ширина кости в одной позиции, вдоль оси будущего импланта."""
    nb = _neighbours(present, fdi)
    if not nb:
        return None
    upper = fdi // 10 in (1, 2)
    centre = _site_centre(fdi, present, mask, spacing)
    if centre is None:
        return None
    axis = _site_axis(fdi, present, mask, spacing, z_up)
    if axis is None:
        return None
    if sinus is None and upper:
        return {"fdi": fdi, "neighbours": sorted(nb),
                "height_mm": None, "width_mm": None, "width_trusted": False,
                "limited_by": "дно гайморовой пазухи",
                "note": ("высота не измерена: пазухи этой моделью почти не "
                         "размечены — нужна модель pretrained")}

    centre_mm = centre * spacing            # ось зуба живёт в миллиметрах
    target = sinus if upper else nerve
    t_idx = np.array(np.nonzero(target), float)
    b_idx = np.array(np.nonzero(bone), float)
    if not b_idx.size:
        return None

    def along(idx_arr):
        """Координаты вдоль оси и поперёк неё, в мм."""
        pts = idx_arr.T * spacing - centre_mm
        s_ax = pts @ axis
        radial = np.linalg.norm(pts - np.outer(s_ax, axis), axis=1)
        return s_ax, radial

    # Кость в узком цилиндре вокруг оси — это и есть будущее ложе импланта.
    s_b, r_b = along(b_idx)
    core = r_b <= 3.0
    if core.sum() < 50:
        return None
    s_core = s_b[core]

    # Гребень: край кости в сторону коронки. Ось tooth_axis направлена К
    # КОРОНКЕ, значит гребень — максимум координаты вдоль неё. Берём 98-й
    # перцентиль, а не максимум: одиночный воксель иначе задаёт уровень.
    crest = float(np.percentile(s_core, 98))

    height_mm = None
    if t_idx.size:
        s_t, r_t = along(t_idx)
        near = (r_t <= 8.0) & (s_t < crest)      # канал/пазуха глубже гребня
        if near.any():
            height_mm = float(crest - s_t[near].max())

    # Ширина: поперечник кости на WIDTH_BELOW_MM ниже гребня.
    # Полоса ТОЛЬКО вокруг позиции: без радиального ограничения она проходит
    # через всю челюсть, и «ширина гребня» выходила 100 мм.
    band = (np.abs(s_b - (crest - WIDTH_BELOW_MM)) <= WIDTH_SPAN_MM / 2)         & (r_b <= 7.0)
    width_mm = None
    if band.sum() >= 20:
        pts = b_idx.T[band] * spacing - centre_mm
        flat = pts - np.outer(pts @ axis, axis)   # проекция в плоскость гребня
        # Ширина — поперёк ряда. Направление вдоль ряда даёт длину промежутка,
        # а не толщину гребня, поэтому берём МЕНЬШУЮ из главных осей сечения.
        u, s_val, _vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
        proj = (flat - flat.mean(0)) @ _vt[1]     # вторая главная ось
        width_mm = float(np.percentile(proj, 97.5) - np.percentile(proj, 2.5))

    res = {"fdi": fdi, "neighbours": sorted(nb),
           "height_mm": round(height_mm, 1) if height_mm else None,
           "width_mm": round(width_mm, 1) if width_mm else None,
           "width_trusted": bool(width_mm and 2.0 <= width_mm <= 15.0),
           "limited_by": ("дно гайморовой пазухи" if upper
                          else "нижнечелюстной канал")}
    if height_mm and height_mm >= MIN_TRUST_MM:
        letter, text = rbh_class(height_mm)
        res["rbh_class"] = letter
        res["rbh_text"] = text
    elif height_mm is None:
        res["note"] = ("на пути ни канала, ни пазухи — высота по ним "
                       "не ограничена")
    return res


def main() -> None:
    print("ВНИМАНИЕ: измерение НЕ ДОВЕДЕНО, числа недостоверны "
          "(см. докстринг модуля). Врачу не показывать.\n")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    res = analyze(a.case, a.model)
    print(f"кейс {res['case_id']}, модель {res['model']}, "
          f"зубов {len(res['present'])}")
    if not res["sites"]:
        print("мест под имплант не найдено: пропусков в ряду нет "
              "(восьмёрки не считаем — их не имплантируют)")
    for s in res["sites"]:
        print(f"\nпозиция {s['fdi']} (соседи {s['neighbours']})")
        if s["height_mm"] is None:
            print(f"  высота: {s.get('note', 'не измерена')}")
        else:
            cls = f" — класс {s['rbh_class']}, {s['rbh_text']}" \
                if "rbh_class" in s else ""
            print(f"  высота кости до {s['limited_by']}: "
                  f"{s['height_mm']} мм{cls}")
        print(f"  ширина гребня (на {WIDTH_BELOW_MM} мм ниже верхушки): "
              f"{s['width_mm']} мм" if s["width_mm"] else "  ширина не измерена")
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"\nсохранено: {a.json}")


if __name__ == "__main__":
    main()
