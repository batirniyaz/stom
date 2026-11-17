"""База измерения «корень верхнего зуба против дна гайморовой пазухи».

Зачем. Мы выдавали находку «корень близко к пазухе» при расстоянии ≤2 мм, и на
client5 это сработало у 62% верхних моляров, на client4 — у 40%. Литература
объясняет почему: контакт корня с дном пазухи есть у 75-78% моляров, то есть
это НОРМА, а не находка. Тревога у двух третей пациентов — ровно та ловушка, из
которой уже вытаскивали пародонт.

Но литературные 75% брать напрямую нельзя: у нас своя сегментация, свой
воксель и своя граница пазухи. Меряем СВОЮ базу на эталонных масках
ToothFairy3, как это уже сделано для уровня кости.

Что считаем на каждый верхний премоляр и моляр:
  * расстояние от корня до пазухи (0, если корень касается или входит);
  * ПРОТРУЗИЮ — насколько глубоко корень заходит в пазуху, если заходит.
    Именно она клинически значима: контакт есть у большинства, а вот глубокое
    вдавление в пазуху меняет тактику удаления и имплантации.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_calibrate_sinus.py --limit 60
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "research_3d/nnUNet_raw/Dataset113_ToothFairy3"
LABELS = DS / "labelsTr_fdi"

SINUS_CLASSES = (5, 6)
MIN_TOOTH_VOX = 300
# Верхние премоляры и моляры — только они соседствуют с пазухой.
UPPER_POSTERIOR = [t for t in cb.TEETH_CLASSES
                   if t // 10 in (1, 2) and t % 10 >= 4]


def measure(mask: np.ndarray, spacing: np.ndarray) -> list:
    """(зуб, расстояние до пазухи, глубина протрузии) для верхних жевательных."""
    sinus = np.isin(mask, SINUS_CLASSES)
    if sinus.sum() < 500:
        return []
    # Расстояние снаружи пазухи и глубина внутри неё — двумя полями.
    out_d = ndi.distance_transform_edt(~sinus, sampling=spacing)
    in_d = ndi.distance_transform_edt(sinus, sampling=spacing)
    rows = []
    for t in UPPER_POSTERIOR:
        bm = mask == t
        if bm.sum() < MIN_TOOTH_VOX:
            continue
        gap = float(out_d[bm].min())
        # Протрузия: самая глубокая точка корня ВНУТРИ пазухи. Ноль, если
        # корень до пазухи не дошёл.
        inside = bm & sinus
        prot = float(in_d[inside].max()) if inside.any() else 0.0
        rows.append((int(t), gap, prot))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    stems = sorted(p.name[:-len(".nii.gz")] for p in LABELS.glob("*.nii.gz"))
    stems = stems[:a.limit]
    print(f"эталонных томов к обработке: {len(stems)}")

    gaps, prots, per_case = [], [], []
    for i, stem in enumerate(stems, 1):
        try:
            img = sitk.ReadImage(str(LABELS / f"{stem}.nii.gz"))
            mask = sitk.GetArrayFromImage(img).astype(np.int16)
            spacing = np.array(img.GetSpacing(), float)[::-1]
        except Exception as exc:  # noqa: BLE001
            print(f"  {stem}: пропуск — {exc}")
            continue
        rows = measure(mask, spacing)
        if not rows:
            continue
        n_touch = sum(1 for _t, g, _p in rows if g <= 0.001)
        per_case.append((len(rows), n_touch))
        for _t, g, p in rows:
            gaps.append(g)
            prots.append(p)
        if i % 10 == 0:
            print(f"  {i}/{len(stems)}: зубов {len(gaps)}", flush=True)

    g = np.array(gaps)
    p = np.array(prots)
    n_teeth = len(g)
    print(f"\nверхних премоляров и моляров: {n_teeth} "
          f"в {len(per_case)} томах")

    print("\n=== КОНТАКТ С ПАЗУХОЙ (расстояние 0) ===")
    touch = int((g <= 0.001).sum())
    print(f"  касаются или входят: {touch} ({100*touch/max(n_teeth,1):.0f}%)")
    print(f"  для сравнения, литература: 75-78% моляров")
    per = np.array([b / max(a_, 1) for a_, b in per_case])
    print(f"  на пациента: медиана {100*np.median(per):.0f}% зубов, "
          f"квартили {100*np.percentile(per,25):.0f}..{100*np.percentile(per,75):.0f}%")

    print("\n=== РАССТОЯНИЕ до пазухи, мм (у тех, кто НЕ касается) ===")
    far = g[g > 0.001]
    if far.size:
        qs = [10, 25, 50, 75, 90]
        print("  перцентили: " + ", ".join(
            f"{q}%={v:.1f}" for q, v in zip(qs, np.percentile(far, qs))))
    for thr in (0.5, 1.0, 2.0, 3.0):
        print(f"  ближе {thr} мм (включая контакт): "
              f"{100*(g <= thr).mean():.0f}% зубов")

    print("\n=== ПРОТРУЗИЯ в пазуху, мм — вот что редко ===")
    pos = p[p > 0.001]
    print(f"  корней, заходящих в пазуху: {len(pos)} "
          f"({100*len(pos)/max(n_teeth,1):.0f}%)")
    if pos.size:
        qs = [50, 75, 90, 95, 99]
        print("  глубина, перцентили: " + ", ".join(
            f"{q}%={v:.1f}" for q, v in zip(qs, np.percentile(pos, qs))))
        print(f"  максимум {pos.max():.1f} мм")
        print("\n  ПОРОГИ ПО ВЫБОРКЕ (доля зубов ниже порога):")
        for q in (85, 90, 95, 99):
            print(f"    {q}-й перцентиль ВСЕХ зубов: "
                  f"{np.percentile(p, q):.1f} мм")

    if a.json:
        a.json.write_text(json.dumps(
            {"n_teeth": n_teeth, "gaps": gaps, "protrusions": prots},
            ensure_ascii=False), encoding="utf-8")
        print(f"\nсырые числа: {a.json}")


if __name__ == "__main__":
    main()
