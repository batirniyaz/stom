"""Проверка порога эндодонтии на ЧУЖИХ данных с эталонной пульпой.

Зачем. `FILLED_DIFF_HU = 300` (разница медиан «пульпа минус дентин того же
зуба», выше которой канал считается пролеченным) стоит на 51 зубе трёх
пациентов с одного аппарата. Зазор между живыми и пломбами вышел 342 HU без
единого перекрытия — но это три человека, и порог может оказаться удачей
выборки, как уже случалось с пародонтом и пазухами.

Pulpy3D даёт 96 томов с ЭТАЛОННОЙ разметкой пульпы. Клинической правды («этот
канал пролечен») там нет, поэтому проверяется распределение: где стоит порог
относительно сотен живых зубов и нет ли у распределения второго горба.

ЧТО ЭТИМИ ДАННЫМИ ПРОВЕРИТЬ НЕЛЬЗЯ, и это выяснилось на замере: в наборе с
разметкой пульпы пролеченного зуба нет ПО ПОСТРОЕНИЮ — канал заполнен
пломбировочным материалом, пульпы там нет, размечать нечего. Замер на 499
пульпах это подтвердил: ни одного значения выше порога, самое высокое −65 при
пороге +300. То есть проверяется только СПЕЦИФИЧНОСТЬ (нет ложных срабатываний
на живых зубах), а чувствительность остаётся на наших шести пломбах.

ДВЕ ОСОБЕННОСТИ ДАННЫХ, из-за которых мерить приходится иначе.

1. В Pulpy3D нет масок ЗУБОВ, только пульпа и нижнечелюстной канал. Наш метод
   сравнивает пульпу с дентином ТОГО ЖЕ зуба, то есть маска зуба нужна. Вместо
   сегментации берём дентин КОЛЬЦОМ вокруг пульпы: расширяем маску пульпы и
   вычитаем её саму. Кольцо лежит внутри того же зуба по построению.

2. У набора потеряны метаданные о вокселе — везде записан 1.0 мм при реальных
   ~0.3. Поэтому всё считается В ВОКСЕЛЯХ, а миллиметры не используются вовсе.
   Порогу это безразлично: он в единицах плотности.

Кольцевой способ сначала сверяется с нашим на СВОИХ кейсах, где есть и зубы, и
пульпа. Если два способа расходятся, проверять чужие данные бессмысленно.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_validate_endo.py --check-ours
    .venv/Scripts/python.exe research/scripts/cbct_validate_endo.py --limit 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PULPY = (ROOT / "research_3d/datasets/kagglehub_cache/datasets/gingeryy"
         / "pulpy3d/versions/1/Dataset007_Pulpy3D_Full")

PULP_ID = 1                 # класс пульпы в Pulpy3D
MIN_PULP_VOX = 150          # компонента мельче — обрывок, не зуб
RING_ITERS = 3              # толщина кольца дентина, вокселей
RING_MIN_VOX = 200          # кольцо тоньше этого не характеризует дентин


def ring_diff(vol: np.ndarray, pulp: np.ndarray) -> tuple[float, int, float]:
    """Разница медиан «пульпа минус кольцо дентина вокруг неё», в единицах шкалы."""
    from scipy import ndimage as ndi
    grown = ndi.binary_dilation(pulp, iterations=RING_ITERS)
    ring = grown & ~pulp
    if int(ring.sum()) < RING_MIN_VOX:
        return float("nan"), int(pulp.sum()), float("nan")
    inside = float(np.median(vol[pulp]))
    around = float(np.median(vol[ring]))
    return inside - around, int(pulp.sum()), inside


def check_ours(model: str) -> None:
    """Сверка кольцевого способа с нашим на кейсах, где есть маски зубов."""
    import cbct_endo
    print("СВЕРКА способов на наших кейсах (у них есть и зубы, и пульпа)\n")
    print(f"{'кейс':<9}{'зуб':>5}{'наш способ':>13}{'кольцевой':>12}   вывод")
    rows = []
    for cid in ("client2", "client4", "client5"):
        c = cb.get_case(cid)
        if c is None:
            continue
        mdl = model if c["masks"].get(model) else cb.mask_model_for(c)
        if mdl is None:
            continue
        cb._arr_cache["key"] = None
        vol, mask = cb.arrays_for(c, mdl)
        for t in cb.TEETH_CLASSES:
            pulp = mask == t + 100
            tooth = mask == t
            if int(pulp.sum()) < MIN_PULP_VOX or int(tooth.sum()) < 300:
                continue
            dentin = tooth & ~pulp
            if int(dentin.sum()) < 300:
                continue
            ours = float(np.median(vol[pulp]) - np.median(vol[dentin]))
            ring, _n, _ins = ring_diff(vol, pulp)
            if not np.isfinite(ring):
                continue
            rows.append((cid, t, ours, ring))
            agree = (ours >= cb_filled()) == (ring >= cb_filled())
            print(f"{cid:<9}{t:>5}{ours:>13.0f}{ring:>12.0f}   "
                  f"{'совпало' if agree else 'РАСХОЖДЕНИЕ'}")
    if not rows:
        print("нет зубов с пульпой")
        return
    a = np.array([r[2] for r in rows])
    b = np.array([r[3] for r in rows])
    same = int(((a >= cb_filled()) == (b >= cb_filled())).sum())
    print(f"\nзубов {len(rows)}, вердикт совпал у {same} "
          f"({100*same/len(rows):.0f}%)")
    print(f"корреляция способов: {np.corrcoef(a, b)[0,1]:.3f}")
    print(f"наш способ:  медиана {np.median(a):>7.0f}, "
          f"пролеченных по порогу {int((a >= cb_filled()).sum())}")
    print(f"кольцевой:   медиана {np.median(b):>7.0f}, "
          f"пролеченных по порогу {int((b >= cb_filled()).sum())}")


def cb_filled() -> float:
    import cbct_endo
    return float(getattr(cbct_endo, "FILLED_DIFF_HU", 300.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--model", default="sp")
    ap.add_argument("--check-ours", action="store_true",
                    help="сверить кольцевой способ с нашим и выйти")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    if a.check_ours:
        check_ours(a.model)
        return

    import cc3d
    labels = sorted((PULPY / "labelsTr").glob("*.nii*"))[:a.limit]
    if not labels:
        raise SystemExit(f"нет данных в {PULPY}")
    print(f"томов к обработке: {len(labels)}\n")

    diffs, sizes = [], []
    per_case = []
    for i, fl in enumerate(labels, 1):
        stem = fl.name.split(".")[0]
        fi = PULPY / "imagesTr" / f"{stem}_0000.nii"
        if not fi.is_file():
            continue
        try:
            vol = sitk.GetArrayFromImage(sitk.ReadImage(str(fi))).astype(np.float32)
            lab = sitk.GetArrayFromImage(sitk.ReadImage(str(fl))).astype(np.int16)
        except Exception as exc:  # noqa: BLE001
            print(f"  {stem}: пропуск — {exc}")
            continue
        pulp_all = lab == PULP_ID
        if pulp_all.sum() < MIN_PULP_VOX:
            continue
        # Пульпа размечена одним классом; отдельные зубы — это её связные куски.
        cc = cc3d.connected_components(pulp_all, connectivity=26)
        counts = np.bincount(cc.ravel())
        counts[0] = 0
        n_here = 0
        for l in np.nonzero(counts)[0]:
            if counts[l] < MIN_PULP_VOX:
                continue
            d, n, _ins = ring_diff(vol, cc == l)
            if np.isfinite(d):
                diffs.append(d)
                sizes.append(n)
                n_here += 1
        per_case.append((stem, n_here))
        if i % 5 == 0:
            print(f"  {i}/{len(labels)} томов, пульп {len(diffs)}", flush=True)

    d = np.array(diffs)
    if d.size == 0:
        raise SystemExit("не удалось измерить ни одной пульпы")
    thr = cb_filled()
    print(f"\n{'='*64}")
    print(f"РАСПРЕДЕЛЕНИЕ разницы «пульпа минус дентин» на {len(d)} пульпах "
          f"({len(per_case)} томов)")
    print(f"{'='*64}")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("  перцентили: " + ", ".join(
        f"{q}%={v:.0f}" for q, v in zip(qs, np.percentile(d, qs))))
    print(f"  выше порога {thr:.0f}: {int((d >= thr).sum())} "
          f"({100*(d >= thr).mean():.1f}%)")

    # Двугорбость: ищем впадину в гистограмме между главным горбом и хвостом.
    lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)
    hist, edges = np.histogram(d, bins=40, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = int(np.argmax(hist))
    right = hist[peak + 1:]
    print("\n  гистограмма (столбец = доля пульп):")
    for h, c0 in zip(hist, centers):
        bar = "#" * max(int(round(60 * h / hist.max())), 1 if h else 0)
        mark = "  <- порог" if abs(c0 - thr) <= (edges[1] - edges[0]) else ""
        print(f"   {c0:>7.0f} {bar}{mark}")
    if right.size:
        valley = peak + 1 + int(np.argmin(right))
        after = hist[valley + 1:]
        second = valley + 1 + int(np.argmax(after)) if after.size else None
        print(f"\n  главный горб при {centers[peak]:.0f}")
        print(f"  минимум справа от него при {centers[valley]:.0f} "
              f"(в нём {hist[valley]} пульп)")
        # Второй горб считается настоящим, только если в нём заметная доля
        # выборки. Иначе хвост из десятка значений при тысяче объявляется
        # «горбом» и скрипт делает вывод о пороге на шуме — так и вышло на
        # первом полном прогоне: 10 пульп из 1212 были названы вторым горбом.
        second_min = max(3, int(0.02 * len(d)))
        if (second is not None and hist[second] > hist[valley] * 2
                and hist[second] >= second_min):
            print(f"  ВТОРОЙ ГОРБ при {centers[second]:.0f} "
                  f"({hist[second]} пульп) — распределение двугорбое")
            print(f"  наш порог {thr:.0f} "
                  + ("лежит между горбами — подтверждается"
                     if centers[peak] < thr < centers[second]
                     else "НЕ лежит между горбами — требует пересмотра"))
        else:
            print("  второго горба нет.")

    # Отсутствие значений выше порога — это НЕ «порог завышен». В наборе с
    # разметкой пульпы пролеченного зуба быть не может по построению: канал
    # заполнен пломбировочным материалом, пульпы там нет, и размечать нечего.
    # То есть положительного класса в данных НЕТ, и чувствительность ими не
    # проверяется — проверяется только специфичность.
    above = int((d >= thr).sum())
    print()
    if above == 0:
        print(f"  ВЫШЕ ПОРОГА НИ ОДНОЙ из {len(d)} пульп, самая высокая "
              f"{d.max():.0f} при пороге {thr:.0f}.")
        print("  Это проверка СПЕЦИФИЧНОСТИ, и она пройдена: ложных "
              "«канал пролечен» нет ни одного.")
        print("  Чувствительность этими данными НЕ проверяется: там, где "
              "канал запломбирован, пульпы")
        print("  нет и размечать нечего — положительного класса в наборе не "
              "существует.")
    else:
        print(f"  выше порога {above} пульп — их стоит посмотреть глазами: "
              "либо это настоящие")
        print("  пролеченные каналы, попавшие в разметку, либо порог завышен.")

    if a.json:
        a.json.write_text(json.dumps(
            {"diffs": [round(float(x), 1) for x in d],
             "sizes": [int(x) for x in sizes],
             "threshold": thr}, ensure_ascii=False), encoding="utf-8")
        print(f"\n  сырые числа: {a.json}")


if __name__ == "__main__":
    main()
