"""Проверка ВСЕХ кейсов разом: что маска физически возможна.

Зачем этот файл вообще появился. Врач находил ошибку на одном кейсе, я её
чинил и проверял на тех же трёх-четырёх примерах — и следующая правка ломала
пятый. Так всплыли: зеркальные стороны у одной модели, отсев настоящих зубов
фильтром артефактов, потерянные нижние зубы. Каждая из них жила до тех пор,
пока врач не открывал КОНКРЕТНЫЙ кейс и не замечал глазом.

Причина не в правилах — они общие и не привязаны к кейсу. Причина в том, что
ПРОВЕРКА была выборочной. Этот скрипт закрывает именно это: гоняет одни и те
же инварианты по всем кейсам, какие есть на диске, включая пары из внешнего
датасета. Запускать после КАЖДОЙ правки фильтрации, ориентации или порогов.

Что проверяется — только то, что верно для любого человека, без порогов и
вкусовщины:

  * СТОРОНЫ. Зубы квадрантов 2 и 3 обязаны лежать со стороны «влево пациента»
    (направление берётся из ориентации файла). Допускается отклонение у
    центральных резцов: они стоят на средней линии, и по ним судить нельзя.
  * ОБЕ ЧЕЛЮСТИ. Если в снимке видна нижняя челюсть, а зубов на ней нет —
    это отказ сегментации или наш фильтр, но не анатомия.
  * ФИЛЬТР НЕ СЪЕДАЕТ ЖИВОЕ. Сколько зубов было в маске модели и сколько
    осталось после нашей обработки. Потеря больше пары зубов — красный флаг.
  * РАЗМЕРЫ. Зуб мельче порога огрызка недостоверен, и таких не должно быть
    много.

Скрипт ничего не чинит и не трогает данные — только считает и печатает.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_selfcheck.py
    .venv/Scripts/python.exe research/scripts/cbct_selfcheck.py --model sp --limit 8
    .venv/Scripts/python.exe research/scripts/cbct_selfcheck.py --json check.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

MIN_TOOTH_VOX = 300
MIDLINE_TOLERANCE_MM = 3.0   # резцы стоят на средней линии — по ним не судим
LOST_TEETH_ALARM = 2         # потеря больше двух зубов на фильтре — тревога


def _stub_mm3() -> float:
    """Порог огрызка живёт в слое находок, а не в backend."""
    import cbct_findings as cf
    return float(getattr(cf, "STUB_VOLUME_MM3", 70.0))


def check_case(case_id: str, model: str) -> dict | None:
    c = cb.get_case(case_id)
    if c is None or not c["masks"].get(model):
        return None
    img = sitk.ReadImage(c["masks"][model])
    sign = cb.left_axis_sign(img.GetDirection())
    raw = cb.orient_sides(sitk.GetArrayFromImage(img).astype(np.int16),
                          img.GetDirection())
    # Разворот аппарата обязателен и здесь. Без него `raw` остаётся в
    # НОМЕРАХ ДО зеркала, а `m` приходит уже развёрнутой, и сравнение
    # наборов зубов идёт в двух разных системах: на client2 это выглядело как
    # «фильтр съел 5 зубов» (24, 26, 38, 45, 46), хотя ровно те же пять
    # появились под зеркальными номерами (14, 16, 48, 35, 36), и всего зубов
    # как было 25, так и осталось. Такая ложная тревога опаснее молчания: она
    # маскирует НАСТОЯЩУЮ потерю зубов, ради которой этот скрипт и написан.
    _, raw = cb.apply_scanner_mirror(None, raw, c)
    cb._arr_cache["key"] = None
    vol, m = cb.arrays_for(c, model)
    spacing = np.array(c["spacing"], float)

    before = [t for t in cb.TEETH_CLASSES if (raw == t).sum() >= MIN_TOOTH_VOX]
    after = [t for t in cb.TEETH_CLASSES if (m == t).sum() >= MIN_TOOTH_VOX]
    lost = sorted(set(before) - set(after))

    jaw = np.isin(m, cb.JAW_FOR_MIDLINE)
    mid = float(np.where(jaw)[2].mean()) if jaw.sum() >= 1000 else None
    wrong = []
    if mid is not None:
        for t in after:
            x = float(np.where(m == t)[2].mean())
            if abs(x - mid) * spacing[2] < MIDLINE_TOLERANCE_MM:
                continue                      # центральные резцы
            if ((x - mid) * sign > 0) != (t // 10 in (2, 3)):
                wrong.append(t)

    upper = [t for t in after if t // 10 in (1, 2)]
    lower = [t for t in after if t // 10 in (3, 4)]
    # Нижняя челюсть в кадре? Класс 1 — нижняя, 2 — верхняя.
    has_mandible = int((m == 1).sum()) > 50000
    stub = [t for t in after
            if (m == t).sum() * float(np.prod(spacing)) < _stub_mm3()]

    # Версия обработки. Меши и находки инвалидируются по ней, и если правила
    # фильтрации поменяли, а номер забыли поднять, демо честно отдаёт врачу
    # СТАРЫЕ файлы. Так и случилось: фильтр артефактов был исправлен, версия
    # осталась прежней, и у пациента в 3D по-прежнему не было нижних зубов.
    meshes = (c.get("meshes") or {})
    mesh_v = meshes.get(model)
    stale_mesh = mesh_v is not None and mesh_v != cb.POSTPROC_VERSION
    fnd = (c.get("findings") or {})
    fnd_v = fnd.get("v")
    stale_find = fnd_v is not None and fnd_v != cb.FINDINGS_VERSION

    # Стороны на этом аппарате подтверждены врачом?
    sides = cb.scanner_sides_state(c)

    problems = []
    if stale_mesh:
        problems.append(f"меш собран версией {mesh_v}, текущая "
                        f"{cb.POSTPROC_VERSION}")
    if stale_find:
        problems.append(f"находки версии {fnd_v}, текущая "
                        f"{cb.FINDINGS_VERSION}")
    if sides["position_is_stub"] and not sides["confirmed"]:
        problems.append(f"стороны аппарата «{sides['scanner']}» не подтверждены "
                        f"(позиция в файле — заглушка)")
    if wrong:
        problems.append(f"стороны: {len(wrong)} зубов не на своей половине "
                        f"({', '.join(str(t) for t in wrong[:6])})")
    if len(lost) > LOST_TEETH_ALARM:
        problems.append(f"фильтр съел {len(lost)} зубов "
                        f"({', '.join(str(t) for t in lost[:6])})")
    if has_mandible and not lower:
        problems.append("нижняя челюсть в кадре, но зубов на ней НЕТ")
    elif has_mandible and len(lower) < 4:
        problems.append(f"на нижней челюсти всего {len(lower)} зубов")
    if not upper:
        problems.append("верхних зубов нет вовсе")
    if len(stub) > 3:
        problems.append(f"огрызков: {len(stub)}")

    return {
        "case": case_id, "model": model,
        "teeth_raw": len(before), "teeth": len(after),
        "upper": len(upper), "lower": len(lower),
        "lost": lost, "wrong_side": wrong, "stub": stub,
        "has_mandible": has_mandible,
        "mesh_version": mesh_v, "findings_version": fnd_v,
        "scanner": sides["scanner"], "sides_confirmed": sides["confirmed"],
        "problems": problems,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None,
                    help="какую модель проверять; по умолчанию — все готовые")
    ap.add_argument("--limit", type=int, default=0,
                    help="ограничить число кейсов (0 — все)")
    ap.add_argument("--only", nargs="*", help="конкретные кейсы")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    ids = a.only or sorted(cb.cases)
    if a.limit:
        ids = ids[:a.limit]
    rows, bad = [], 0
    print(f"{'кейс':<18}{'модель':<12}{'зубов':>7}{'верх':>6}{'низ':>6}"
          f"{'потеряно':>10}  замечания")
    print("-" * 96)
    for cid in ids:
        c = cb.get_case(cid)
        if c is None:
            continue
        models = ([a.model] if a.model
                  else [k for k, v in c["masks"].items() if v])
        for mdl in models:
            try:
                r = check_case(cid, mdl)
            except Exception as exc:      # noqa: BLE001
                print(f"{cid:<18}{mdl:<12}  ошибка: {exc}")
                continue
            if r is None:
                continue
            rows.append(r)
            note = "; ".join(r["problems"]) if r["problems"] else "ок"
            if r["problems"]:
                bad += 1
            print(f"{cid:<18}{mdl:<12}{r['teeth']:>7}{r['upper']:>6}"
                  f"{r['lower']:>6}{len(r['lost']):>10}  {note}")

    print("-" * 96)
    print(f"проверено пар кейс/модель: {len(rows)}, с замечаниями: {bad}")
    if rows:
        wrong_total = sum(len(r["wrong_side"]) for r in rows)
        lost_total = sum(len(r["lost"]) for r in rows)
        print(f"суммарно зубов не на своей стороне: {wrong_total}, "
              f"съедено фильтром: {lost_total}")
    if a.json:
        a.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"подробности: {a.json}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
