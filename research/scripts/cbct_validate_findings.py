"""Проверка слоя находок КЛКТ на МНОГИХ пациентах, а не на четырёх.

Зачем. Пороги пародонта и эндодонтии калибровались на client2/client4/client5 —
это четыре клинических кейса с одного аппарата. На треке сканов такая же
проверка, но на 180 пациентах, перевернула выводы: ориентиры оказались хуже
центроидов, а неверная вертикаль дуги оказалась причиной провала ВСЕХ
измерений прикуса. На четырёх кейсах ни то, ни другое не всплыло бы.

Берём ToothFairy3 (532 тома с разметкой) и гоняем измерения на ЭТАЛОННЫХ
масках. Это отделяет ошибку измерения от ошибки сегментации: если метод
разваливается уже на идеальной маске, виноват метод.

Клинической правды («у этого зуба пломба», «здесь убыль 4 мм») в датасете нет,
поэтому проверяем то, что проверяется без неё:

  * ЭНДОДОНТИЯ. Разделение групп не должно быть удачей четырёх кейсов. Если
    разница «пульпа минус дентин» на сотнях пациентов двугорбая с пустой
    впадиной — метод переносится. Если это сплошной континуум, наш «зазор
    342 HU» был совпадением, и порог придётся признать необоснованным.
  * ПАРОДОНТ. Какая доля зубов вообще измеряется (у client4 — 17 из 28) и
    сколько зубов на пациента помечается убылью. Если убыль находится у
    половины зубов у КАЖДОГО пациента, порог даёт ложную тревогу, и врач
    перестанет читать раздел — та же ловушка, из-за которой на прикусе пороги
    выбирались по специфичности ~90%, а не по максимуму разделения.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_validate_findings.py --limit 40
    .venv/Scripts/python.exe research/scripts/cbct_validate_findings.py --limit 200 --json out.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_endo as E  # noqa: E402
import cbct_perio as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "research_3d/nnUNet_raw/Dataset113_ToothFairy3"
IMAGES, LABELS = DS / "imagesTr", DS / "labelsTr_fdi"


def _load(stem: str):
    """Том и эталонная маска одного пациента, в (z, y, x) и мм."""
    img = sitk.ReadImage(str(IMAGES / f"{stem}_0000.mha")
                         if (IMAGES / f"{stem}_0000.mha").is_file()
                         else str(IMAGES / f"{stem}_0000.nii.gz"))
    lab = sitk.ReadImage(str(LABELS / f"{stem}.nii.gz"))
    vol = sitk.GetArrayFromImage(img).astype(np.int16)
    mask = sitk.GetArrayFromImage(lab).astype(np.int16)
    # SimpleITK отдаёт spacing как (x, y, z) — нам нужен порядок осей массива
    spacing = np.array(img.GetSpacing(), float)[::-1]
    return vol, mask, spacing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    stems = sorted(p.name[:-len(".nii.gz")] for p in LABELS.glob("*.nii.gz"))
    stems = stems[:a.limit]
    print(f"пациентов к проверке: {len(stems)} из {len(list(LABELS.glob('*.nii.gz')))}")

    endo_diff, endo_filled = [], []
    perio_rate, perio_hits, perio_teeth = [], [], []
    perio_mm, perio_frac = [], []
    failed = []
    for i, stem in enumerate(stems, 1):
        try:
            vol, mask, spacing = _load(stem)
        except Exception as exc:  # noqa: BLE001
            failed.append((stem, f"чтение: {exc}"))
            continue
        try:
            rows, _no_pulp = E._tooth_rows(vol, mask, spacing)
            for r in rows:
                endo_diff.append(r["diff_hu"])
                if r["status"] == "filled":
                    endo_filled.append(r["diff_hu"])
        except Exception as exc:  # noqa: BLE001
            failed.append((stem, f"эндо: {exc}"))
        try:
            res = P.analyze_arrays(vol, mask, spacing)
            n_ok, n_skip = len(res["teeth"]), len(res["not_measured"])
            if n_ok + n_skip:
                perio_rate.append(n_ok / (n_ok + n_skip))
            hit = sum(1 for row in res["teeth"]
                      for s in row["sites"].values()
                      if s["severity"] != "норма")
            perio_hits.append(hit)
            perio_teeth.append(n_ok)
            for row in res["teeth"]:
                for s in row["sites"].values():
                    perio_mm.append(s["mm"])
                    if s["root_fraction"] is not None:
                        perio_frac.append(s["root_fraction"])
        except Exception as exc:  # noqa: BLE001
            failed.append((stem, f"перио: {exc}"))
        if i % 5 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)}: зубов с пульпой {len(endo_diff)}, "
                  f"помечено пломб {len(endo_filled)}", flush=True)

    print("\n================ ЭНДОДОНТИЯ ================")
    if endo_diff:
        d = np.array(endo_diff, float)
        print(f"зубов с читаемой пульпой: {len(d)}")
        print(f"разница пульпа-дентин: медиана {np.median(d):+.0f}, "
              f"перцентили 1/5/95/99: "
              + ", ".join(f"{v:+.0f}" for v in np.percentile(d, [1, 5, 95, 99])))
        print(f"помечено пломбированными: {len(endo_filled)} "
              f"({100*len(endo_filled)/len(d):.1f}%)")
        # Двугорбость: сколько зубов попало в окрестность порога. У метода,
        # который переносится, там должна быть ВПАДИНА.
        for lo, hi in ((0, 300), (300, 600), (600, 1000)):
            n = int(((d >= lo) & (d < hi)).sum())
            print(f"  в полосе {lo:+5d}..{hi:+5d}: {n:5d} зубов "
                  f"({100*n/len(d):.2f}%)")
        near = int(((d > -200) & (d < 600)).sum())
        print(f"ВПАДИНА вокруг порога (-200..+600): {near} зубов "
              f"({100*near/len(d):.2f}%) — чем меньше, тем надёжнее порог")

    print("\n================ ПАРОДОНТ ================")
    if perio_rate:
        r = np.array(perio_rate, float)
        h = np.array(perio_hits, float)
        tn = np.array(perio_teeth, float)
        print(f"пациентов: {len(r)}")
        print(f"доля ИЗМЕРЕННЫХ зубов: медиана {np.median(r):.2f}, "
              f"квартили {np.percentile(r,25):.2f}..{np.percentile(r,75):.2f}")
        print(f"измеренных зубов на пациента: медиана {np.median(tn):.0f}")
        print(f"участков с убылью на пациента: медиана {np.median(h):.0f}, "
              f"квартили {np.percentile(h,25):.0f}..{np.percentile(h,75):.0f}, "
              f"максимум {h.max():.0f}")
        clean = int((h == 0).sum())
        print(f"пациентов БЕЗ единой находки: {clean} "
              f"({100*clean/len(h):.0f}%) — если их почти нет, порог "
              f"тревожит на каждом")

    if perio_mm:
        mm = np.array(perio_mm, float)
        print(f"\nРАССТОЯНИЕ ШЕЙКА-ГРЕБЕНЬ по всем участкам ({len(mm)}):")
        qs = [5, 25, 50, 75, 85, 90, 95, 99]
        print("  перцентили " + ", ".join(
            f"{q}%={v:.1f}" for q, v in zip(qs, np.percentile(mm, qs))))
        print(f"  текущие пороги: норма<{P.NORMAL_MM}, лёгкая<{P.MILD_MM}, "
              f"умеренная<{P.MODERATE_MM}")
        for name, thr in (("норма", P.NORMAL_MM), ("лёгкая", P.MILD_MM),
                          ("умеренная", P.MODERATE_MM)):
            print(f"  выше порога {name} ({thr} мм): "
                  f"{100*(mm > thr).mean():.0f}% участков")
        print("  ЕСЛИ БЫ пороги ставились по населению "
              "(специфичность как на прикусе):")
        for spec in (80, 85, 90, 95):
            print(f"    {spec}% участков в норме -> порог "
                  f"{np.percentile(mm, spec):.1f} мм")
    if perio_frac:
        fr = np.array(perio_frac, float)
        print(f"\nДОЛЯ ДЛИНЫ КОРНЯ ({len(fr)}): перцентили " + ", ".join(
            f"{q}%={v:.2f}" for q, v in zip([50, 85, 90, 95],
                                            np.percentile(fr, [50, 85, 90, 95]))))

    if failed:
        print(f"\nсбоев: {len(failed)}")
        for stem, why in failed[:10]:
            print(f"  {stem}: {why}")

    if a.json:
        a.json.write_text(json.dumps({
            "n_patients": len(stems), "endo_diff": endo_diff,
            "perio_rate": perio_rate, "perio_hits": perio_hits,
            "perio_mm": perio_mm, "perio_frac": perio_frac,
            "failed": failed,
        }, ensure_ascii=False), encoding="utf-8")
        print(f"\nсырые числа: {a.json}")


if __name__ == "__main__":
    main()
