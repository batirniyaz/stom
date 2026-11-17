"""Эндодонтический статус зуба по КЛКТ: канал запломбирован или зуб витален.

Считать КАНАЛЫ по маске мы уже пробовали и закрыли отрицательно (маска сливает
корни и крошится у апекса, см. CLAUDE.md). Но эндодонтический статус — вопрос
не топологии, а плотности, и решается он на тех же масках.

Ключ к методу: сравнивать зуб САМ С СОБОЙ, а не с абсолютной шкалой.
Абсолютный порог здесь обречён дважды:
  * шкала КЛКТ не количественная — на client4 воздух даёт -1035 (правдоподобно),
    но дентин читается 1400-3200 при настоящих 1200-1800, а эмаль 3043 при
    2500-3000. Порог, снятый с одного аппарата, на другом уедет;
  * эмаль сама плотнее любого разумного порога, поэтому «плотное внутри зуба»
    находится у КАЖДОГО зуба. Замер: у заведомо живого 11 в корне 3485 вокселей
    выше 2200 HU.

Поэтому меряем РАЗНИЦУ медиан: плотность внутри маски пульпы минус плотность
дентина того же зуба. Живая пульпа — мягкая ткань, она ЛЕГЧЕ дентина;
пломбировочный материал ТЯЖЕЛЕЕ. Некалиброванность шкалы при вычитании
сокращается.

Замер на client2 + client4 + client5, 51 зуб с читаемой пульпой:
    живые (45 зубов):  медиана -1073, самый высокий  +24
    пломбы (6 зубов):  медиана +1591, самая низкая  +366
    ЗАЗОР между группами: 342 HU, перекрытия нет ни одного зуба
Картина клинически связная: у client2 пролечен сектор 37, 43, 44, 45, 47,
у client4 один зуб 35, у client5 ни одного.

ВАЖНО: отсутствие маски пульпы НЕ означает «канал пролечен». На client4 без
пульпы остались 17, 28, 43, 46, 48, и у 17, 46, 48 маска зуба — огрызок с
корнем 2.7-3.6 мм, то есть провал сегментации, а не клиника. Такие зубы уходят
в ручную проверку, а не в диагноз.

ЧЕГО МОДУЛЬ НЕ ДЕЛАЕТ: не судит о качестве пломбирования. «Пломба не доходит
до апекса» померить не удалось — контроль показал, что зазор до апекса
ОДИНАКОВ у пролеченных и у живых зубов (медианы 2.7 и 2.4 мм на client2,
2.0 и 2.7 мм на client4). То есть это не недопломбировка, а маска пульпы не
тянет тонкий апикальный канал — то же ограничение, из-за которого закрыт
подсчёт каналов. Зазор считается и лежит в `fill_gap_mm`/`pulp_gap_mm`, но
находкой НЕ становится. Вернуться к нему можно только с моделью, которая
тянет канал до апекса.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_endo.py client4
    .venv/Scripts/python.exe research/scripts/cbct_endo.py client2 --model tf3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402
import cbct_perio as P  # noqa: E402

MIN_TOOTH_VOX = 300
MIN_PULP_VOX = 30       # меньше — маска пульпы недостоверна
MIN_CORE_VOX = 50       # дентина должно остаться на честную медиану

# Порог разницы медиан «пульпа минус дентин». Стоит В ЗАЗОРЕ между группами
# (+24 у худшего живого, +366 у самой слабой пломбы), поэтому его точное
# значение внутри 100..350 результата не меняет. Подбирать не пришлось.
FILLED_DIFF_HU = 300
# Уверенный уровень. Ниже него разницу мог дать металлоартефакт, а не пломба:
# проверка глазами показала, что единственный слабый случай (зуб 37 client2,
# +366) стоит вплотную к массивной металлической реставрации, и белая засветка
# от неё занимает пол-кадра. Сильные случаи (+2028, +2004) видно как яркий тяж
# по ходу канала, у живого контроля тот же тяж тёмный.
FILLED_STRONG_HU = 600
METAL_NEAR_MM = 5.0     # ближе этого к коронке/импланту засветка вероятна

CORE_ERODE = 2          # снять эмалевую кайму и частичный объём на границе
PULP_PAD = 2            # отступ от пульпы, чтобы не захватить её край
PROSTH_CLASSES = (8, 9, 10)


def _tooth_rows(vol, mask, spacing):
    """Померить каждый зуб: разница плотностей пульпы и дентина."""
    rows, no_pulp = [], []
    # Расстояние до ближайшего металла: коронки, мосты, импланты дают streak,
    # и он поднимает плотность внутри соседнего зуба. Нужно, чтобы отличить
    # пломбу в канале от засветки.
    z_up = cb.occlusal_up(mask)     # ориентация тома — по данным
    prosth = np.isin(mask, PROSTH_CLASSES)
    dist_metal = (ndi.distance_transform_edt(~prosth, sampling=spacing)
                  if prosth.any() else None)
    for t in sorted(cb.TEETH_CLASSES):
        bm = mask == t
        if bm.sum() < MIN_TOOTH_VOX:
            continue
        pm = mask == (t + 100)
        if pm.sum() < MIN_PULP_VOX:
            no_pulp.append(t)
            continue
        # дентин: тело зуба без пульпы и без внешнего слоя
        core = ndi.binary_erosion(bm, iterations=CORE_ERODE) & \
            ~ndi.binary_dilation(pm, iterations=PULP_PAD)
        if core.sum() < MIN_CORE_VOX:
            no_pulp.append(t)
            continue

        pulp_hu = float(np.median(vol[pm]))
        dent_hu = float(np.median(vol[core]))
        diff = pulp_hu - dent_hu

        # Насколько пломба (или сама пульпа) не доходит до апекса. Меряем
        # ВСЕГДА, в том числе у живых зубов: у живого это зазор самой маски
        # пульпы, и он служит контролем — без него «недопломбировка» была бы
        # неотличима от того, что маска не тянет тонкий апикальный канал.
        idx = np.array(np.where(bm))
        upper = t // 10 in (1, 2)
        axis, origin = P.tooth_axis(idx, spacing, upper, z_up)
        apex = float((((idx.T * spacing) - origin) @ axis).min())
        p_idx = np.array(np.where(pm))
        p_proj = ((p_idx.T * spacing) - origin) @ axis
        pulp_gap = float(p_proj.min() - apex)

        filled = diff > FILLED_DIFF_HU
        fill_gap = None
        if filled:
            dense = vol[pm] > dent_hu + FILLED_DIFF_HU
            if dense.sum() >= 10:
                fill_gap = float(p_proj[dense].min() - apex)

        metal_mm = None if dist_metal is None else float(dist_metal[bm].min())
        # Близость металла гасит ТОЛЬКО слабую разницу. Гасить ею всё подряд
        # нельзя: на client2 металл лежит в 0.3 мм почти от каждого зуба, и
        # такое правило погасило бы даже зуб 44 с разницей +2028, видимый
        # глазом. Контроль там же: зубы на тех же 0.3 мм, но живые, читаются
        # -630, -1154, -427 — значит соседство с металлом само по себе
        # плотность внутри канала не задирает.
        near_metal = metal_mm is not None and metal_mm < METAL_NEAR_MM
        if not filled:
            conf = "high"
        elif diff > FILLED_STRONG_HU:
            conf = "high"
        else:
            conf = "low" if near_metal else "medium"

        rows.append({
            "tooth": t, "pulp_hu": round(pulp_hu), "dentin_hu": round(dent_hu),
            "diff_hu": round(diff), "pulp_vox": int(pm.sum()),
            "status": "filled" if filled else "vital",
            "confidence": conf,
            "metal_mm": None if metal_mm is None else round(metal_mm, 1),
            "pulp_gap_mm": round(pulp_gap, 1),
            "fill_gap_mm": None if fill_gap is None else round(fill_gap, 1),
        })
    return rows, no_pulp


def analyze_case(case, model: str = "tf3") -> dict:
    """Эндодонтический статус всех зубов кейса."""
    c = cb.get_case(case) if isinstance(case, str) else case
    vol, mask = cb.arrays_for(c, model)
    spacing = np.array(c["spacing"], float)
    rows, no_pulp = _tooth_rows(vol, mask, spacing)
    return {
        "case": c.get("id") if isinstance(c, dict) else str(case),
        "model": model, "teeth": rows, "no_pulp": sorted(no_pulp),
        "filled": sorted(r["tooth"] for r in rows if r["status"] == "filled"),
        "draft": True,
    }


def build_findings(res: dict) -> list:
    """Находки эндодонта. Формат общий с cbct_findings."""
    out = []
    filled = [r for r in res["teeth"] if r["status"] == "filled"]
    sure = [r for r in filled if r["confidence"] != "low"]
    weak = [r for r in filled if r["confidence"] == "low"]
    if sure:
        teeth = sorted(r["tooth"] for r in sure)
        out.append({
            "code": "endo_filled", "tooth": None, "teeth": teeth,
            "value": len(teeth), "unit": None, "severity": "info",
            "text": ("Каналы запломбированы у зубов "
                     + ", ".join(map(str, teeth))
                     + " (плотность внутри канала выше дентина того же зуба)"),
            "action": "оценить качество пломбирования по срезам; "
                      "при жалобах или очаге у апекса — перелечивание",
            "route": "therapy", "confidence": "medium",
            # Слабейший зуб группы: по нему видно, насколько находка близка
            # к порогу. Абсолютные HU здесь бессмысленны (шкала КЛКТ не
            # калибрована) — значение имеет только разница с СОБСТВЕННЫМ
            # дентином зуба, см. докстринг модуля.
            "basis": [
                {"label": "слабейшая разница пульпа − дентин",
                 "value": min(r["diff_hu"] for r in sure), "unit": "HU",
                 "rule": f"порог {FILLED_DIFF_HU} HU",
                 "note": "замер на 51 зубе: живые до +24, пломбы от +366 — "
                         "порог стоит в пустом зазоре 342 HU"},
                {"label": "зубов в находке", "value": len(teeth),
                 "unit": "шт.", "rule": "каждый сравнивается сам с собой"},
            ],
        })
    if weak:
        # Металл рядом поднимает плотность внутри соседнего зуба. Такой зуб
        # не диагностируем, а показываем врачу на срезах.
        teeth = sorted(r["tooth"] for r in weak)
        out.append({
            "code": "endo_uncertain", "tooth": None, "teeth": teeth,
            "value": None, "unit": None, "severity": "info",
            "text": ("Возможно пролеченные каналы у зубов "
                     + ", ".join(map(str, teeth))
                     + ": плотность повышена слабо, и рядом металлическая "
                       "реставрация — это может быть засветка, а не пломба"),
            "action": "посмотреть срезы вручную",
            "route": "review", "confidence": "low",
            "basis": [
                {"label": "наибольшая разница пульпа − дентин",
                 "value": max(r["diff_hu"] for r in weak), "unit": "HU",
                 "rule": f"уверенно с {FILLED_STRONG_HU} HU",
                 "note": "металл рядом гасит только слабый сигнал: сильный "
                         "гасить нельзя, проверено на client2"},
                {"label": "ближайший металл",
                 "value": round(min(r["metal_mm"] for r in weak
                                    if r["metal_mm"] is not None), 1)
                 if any(r["metal_mm"] is not None for r in weak) else None,
                 "unit": "мм", "rule": f"настораживает ближе {METAL_NEAR_MM} мм"},
            ],
        })
    if res["no_pulp"]:
        out.append({
            "code": "endo_no_pulp", "tooth": None,
            "teeth": sorted(res["no_pulp"]), "value": None, "unit": None,
            "severity": "info",
            "text": (f"Пульпа не выделена у зубов {sorted(res['no_pulp'])}: "
                     f"статус канала не определён. Это НЕ значит, что зуб "
                     f"депульпирован — чаще это провал сегментации"),
            "action": "посмотреть срезы вручную",
            "route": "review", "confidence": "high",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case")
    ap.add_argument("--model", default="tf3")
    a = ap.parse_args()

    res = analyze_case(a.case, a.model)
    print(f"{'зуб':>4} {'пульпа HU':>10} {'дентин HU':>10} {'разница':>8} "
          f"{'до металла':>11} {'зазор пульпы':>13} {'зазор пломбы':>13}  статус")
    for r in res["teeth"]:
        fg = "—" if r["fill_gap_mm"] is None else f"{r['fill_gap_mm']:.1f} мм"
        mm = "—" if r["metal_mm"] is None else f"{r['metal_mm']:.1f} мм"
        st = "ПЛОМБА" if r["status"] == "filled" else "витален"
        if r["status"] == "filled" and r["confidence"] == "low":
            st = "пломба? (металл рядом)"
        print(f"{r['tooth']:>4} {r['pulp_hu']:10d} {r['dentin_hu']:10d} "
              f"{r['diff_hu']:+8d} {mm:>11} {r['pulp_gap_mm']:10.1f} мм "
              f"{fg:>13}  {st}")

    # Контроль: зазор до апекса у живых и у пломбированных. Если он одинаков,
    # «недопломбировка» неотличима от того, что маска не тянет тонкий канал,
    # и такую находку выдавать нельзя.
    vital = [r["pulp_gap_mm"] for r in res["teeth"] if r["status"] == "vital"]
    fil = [r["fill_gap_mm"] for r in res["teeth"]
           if r["fill_gap_mm"] is not None]
    if vital:
        print(f"\nконтроль, зазор маски пульпы до апекса у ЖИВЫХ: "
              f"медиана {np.median(vital):.1f} мм, "
              f"размах {min(vital):.1f}..{max(vital):.1f}")
    if fil:
        print(f"зазор пломбы до апекса у ПРОЛЕЧЕННЫХ: "
              f"медиана {np.median(fil):.1f} мм, "
              f"размах {min(fil):.1f}..{max(fil):.1f}")
    if res["no_pulp"]:
        print(f"\nпульпа не выделена: {res['no_pulp']} — статус НЕ определён")
    for f in build_findings(res):
        print(f"\n[{f['route']}] {f['text']}")


if __name__ == "__main__":
    main()
