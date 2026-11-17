"""Проверка модели на ЧУЖИХ данных, на которых она не училась.

Зачем. До сих пор мы оценивали модели на ToothFairy3 — том же датасете, где и
учили, — а «сколько зубов нашла» сравнивали между своими же моделями. Это
замкнутый круг: если разметка неполная (а у ToothFairy3 медиана 22 зуба на том
против 30 у DentVoxel), то и метрика, и модель наследуют один и тот же изъян.

DentVoxel даёт выход: 100 томов, воксель 0.3 мм как у нас, 38 классов,
разметка существенно полнее. Модель на нём не училась, поэтому число здесь
честное.

Считаем то, что видит врач:
  * сколько зубов найдено против эталона (recall) и сколько выдумано (precision);
  * Dice по зубам — насколько совпадает форма;
  * отдельно по челюстям: у ToothFairy3 верхняя размечена хуже, и если модель
    унаследовала перекос, он проявится именно здесь.

ВАЖНО про нумерацию. У DentVoxel и ToothFairy3 классы пронумерованы ОБРАТНО:
у DentVoxel id 1 это верхняя челюсть, у ToothFairy3 — нижняя; каналы и пазухи
тоже зеркально. Соответствие задаётся явной таблицей ниже, вслепую сравнивать
нельзя.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_eval_external.py --model tf3 --limit 20
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DV = ROOT / "research_3d/datasets/dentvoxel/DentVoxel_Dataset"

# id зуба в DentVoxel -> номер FDI. В их схеме зубы идут подряд с 3:
# 3..10 = 11..18, 11..18 = 21..28, 19..26 = 31..38, 27..34 = 41..48.
#
# ПРО СТОРОНЫ. Раньше здесь стоял разворот квадрантов: модель нумеровала по
# конвенции ToothFairy, а та зеркальна относительно ориентации, записанной в
# файлах. Теперь сторону приводит к анатомии сам backend (`orient_sides`), и
# эталон разворачивать больше не нужно — DentVoxel анатомически верен.
# Проверено: на 8 томах DentVoxel зубы квадранта 2 лежат слева пациента
# согласно direction, во всех восьми; у ToothFairy во всех 120 проверенных
# томах наоборот.
#
# Так метрика меряет ровно то, что уходит врачу: разворот сторон входит в
# оцениваемый путь, а не компенсируется отдельно в скрипте оценки.
DV_TOOTH_TO_FDI = {}
_fdi = [t for q in (1, 2, 3, 4) for t in range(q * 10 + 1, q * 10 + 9)]
for _i, _f in enumerate(_fdi):
    DV_TOOTH_TO_FDI[3 + _i] = _f

MIN_TOOTH_VOX = 300


def load_dv(stem: str):
    img = sitk.ReadImage(str(DV / "image" / f"{stem}.nii.gz"))
    lab = sitk.ReadImage(str(DV / "label" / f"{stem.replace('img', 'label')}.nii.gz"))
    return img, sitk.GetArrayFromImage(lab).astype(np.int16)


def gt_teeth(dv_mask: np.ndarray) -> dict:
    """Эталонные зубы DentVoxel в НАШЕЙ нумерации FDI."""
    out = {}
    for dv_id, fdi in DV_TOOTH_TO_FDI.items():
        bm = dv_mask == dv_id
        n = int(bm.sum())
        if n >= MIN_TOOTH_VOX:
            out[fdi] = bm
    return out


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="tf3", help="модель из cbct_backend.MODELS")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    # Отсеиваем служебные файлы macOS («._имя», 4 КБ ресурсной вилки): они
    # попали в архив и по маске *.nii.gz неотличимы от данных, но SimpleITK
    # на них падает с «Unable to determine ImageIO reader».
    stems = sorted(p.name[:-len(".nii.gz")]
                   for p in (DV / "image").glob("*.nii.gz")
                   if not p.name.startswith("._"))[:a.limit]
    if not stems:
        raise SystemExit(f"нет данных в {DV}")
    print(f"внешняя проверка модели «{a.model}» на DentVoxel: "
          f"{len(stems)} томов\n")

    rows = []
    for i, stem in enumerate(stems, 1):
        img, dv = load_dv(stem)
        gt = gt_teeth(dv)
        if not gt:
            continue
        # Инференс нашей моделью через тот же путь, что и в демо.
        arr = sitk.GetArrayFromImage(img).astype(np.float32)[None]
        props = {"spacing": img.GetSpacing()[::-1],
                 "sitk_stuff": {"spacing": img.GetSpacing(),
                                "origin": img.GetOrigin(),
                                "direction": img.GetDirection()}}
        on_dev = cb._fits_on_gpu(a.model, arr.shape[1:])
        pred = cb.get_predictor(a.model, on_dev).predict_single_npy_array(
            arr, props, None, None, False)
        # spacing обязателен: у моделей с единым классом пульпы номер зуба
        # возвращается по РАССТОЯНИЮ до ближайшего зуба, а оно в миллиметрах.
        pred = cb.remap_prediction(pred, a.model, spacing=props["spacing"])
        # Тот же порядок, что в cbct_backend.predict.
        pred = cb.orient_sides(pred, img.GetDirection())
        if a.model in cb.MODEL_NEEDS_REMAP:
            pred = cb.split_paired_sides(pred, img.GetDirection())

        found = {t: (pred == t) for t in cb.TEETH_CLASSES
                 if (pred == t).sum() >= MIN_TOOTH_VOX}
        hit = sorted(set(gt) & set(found))
        # Dice ПО НОМЕРАМ ЗУБОВ, а не списком. Доверительный интервал считается
        # отдельно по типам зубов — у клыков случаев мало, и интервал там
        # заведомо широкий; с плоским списком номер потерян и такую группу уже
        # не выделить. TP/FP/FN названы явно, чтобы их не приходилось
        # выводить из gt/pred/hit и не ошибиться знаком.
        row = {
            "case": stem,
            "gt": len(gt), "pred": len(found), "hit": len(hit),
            "tp": len(hit),
            "fp": len(found) - len(hit),
            "fn": len(gt) - len(hit),
            "missed": sorted(set(gt) - set(found)),
            "extra": sorted(set(found) - set(gt)),
            "dice": {str(t): float(dice(gt[t], found[t])) for t in hit},
            "gt_upper": sum(1 for t in gt if t // 10 in (1, 2)),
            "hit_upper": sum(1 for t in hit if t // 10 in (1, 2)),
            "gt_lower": sum(1 for t in gt if t // 10 in (3, 4)),
            "hit_lower": sum(1 for t in hit if t // 10 in (3, 4)),
        }
        rows.append(row)
        dvals = list(row["dice"].values())
        d = np.mean(dvals) if dvals else float("nan")
        print(f"  {i:>3}/{len(stems)} {stem}: эталон {row['gt']:>2}, "
              f"нашли {row['pred']:>2}, совпало {row['hit']:>2}, "
              f"Dice {d:.3f}", flush=True)

    if not rows:
        raise SystemExit("не удалось обработать ни одного тома")

    gt_all = sum(r["gt"] for r in rows)
    hit_all = sum(r["hit"] for r in rows)
    pred_all = sum(r["pred"] for r in rows)
    all_dice = [x for r in rows for x in r["dice"].values()]
    print(f"\n{'='*66}")
    print(f"ИТОГ по {len(rows)} томам, модель «{a.model}»")
    print(f"{'='*66}")
    print(f"  зубов в эталоне:        {gt_all}")
    print(f"  найдено моделью:        {pred_all}")
    print(f"  совпало по номеру:      {hit_all}")
    print(f"  ПОЛНОТА (recall):       {100*hit_all/max(gt_all,1):.1f}%  "
          f"<- сколько реальных зубов нашли")
    print(f"  ТОЧНОСТЬ (precision):   {100*hit_all/max(pred_all,1):.1f}%  "
          f"<- сколько найденного существует")
    if all_dice:
        print(f"  Dice по зубам:          медиана {np.median(all_dice):.3f}, "
              f"среднее {np.mean(all_dice):.3f}")
    gu = sum(r["gt_upper"] for r in rows)
    hu = sum(r["hit_upper"] for r in rows)
    gl = sum(r["gt_lower"] for r in rows)
    hl = sum(r["hit_lower"] for r in rows)
    print(f"\n  по челюстям (перекос разметки ToothFairy3 проявится здесь):")
    print(f"    верхняя: {hu}/{gu} = {100*hu/max(gu,1):.1f}%")
    print(f"    нижняя:  {hl}/{gl} = {100*hl/max(gl,1):.1f}%")
    miss = {}
    for r in rows:
        for t in r["missed"]:
            miss[t] = miss.get(t, 0) + 1
    if miss:
        worst = sorted(miss.items(), key=lambda kv: -kv[1])[:8]
        print(f"\n  чаще всего теряются зубы: "
              + ", ".join(f"{t} ({n})" for t, n in worst))

    if a.json:
        # Файл самодостаточный: по нему считается доверительный интервал, и
        # через месяц должно быть понятно, какая модель и какая выборка дали
        # эти числа. Без этого таблица в отчёте — утверждение без источника.
        out = {
            "metric_source": "cbct_eval_external.py",
            "model": a.model,
            "dataset": "DentVoxel (внешняя выборка, модель её не видела)",
            "unit_of_analysis": "пациент (один том = один пациент)",
            "min_tooth_vox": MIN_TOOTH_VOX,
            "created": datetime.now().isoformat(timespec="seconds"),
            "cases": rows,
        }
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"\n  подробности: {a.json}  ({len(rows)} пациентов)")
        print("  бутстреп считать ПО ПАЦИЕНТАМ: зубы одного человека зависимы,\n"
              "  и выборка по зубам даст искусственно узкий интервал.")


if __name__ == "__main__":
    main()
