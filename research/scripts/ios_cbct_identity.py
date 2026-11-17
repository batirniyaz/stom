"""Тот ли это пациент: сверка скана и КЛКТ по «отпечатку» зубного ряда.

Зачем. Измерение на настоящих парах показало, что остаток ICP пациента НЕ
различает: чужой скан дал 0.93 мм по зубам — лучше, чем свой в двух случаях
из трёх. Дуги у разных людей слишком похожи, и совмещение сходится на чужом
почти так же хорошо, как на своём. Пускать такой план врачу нельзя: он
построится по чужой анатомии и внешне будет выглядеть безупречно.

Здесь считаются признаки, которые совмещение НЕ выравнивает, потому что они
не про положение, а про состав и размеры ряда:

  * СОСТАВ — какие зубы вообще есть. У человека свой набор отсутствующих,
    и он виден и на скане, и на КЛКТ. Мера — коэффициент Жаккара.
  * ШАГ ДУГИ — расстояния между центрами соседних по номеру зубов. Это
    индивидуальный профиль: у одного пациента шаг между 14 и 15 свой, у
    другого другой. Совмещение (поворот и сдвиг) расстояния сохраняет, то
    есть подделать их нельзя.
  * РАЗМЕР ЗУБА — поперечник коронки. Тоже сохраняется при совмещении.

Пороги здесь НЕ ставятся заранее: скрипт считает разделение своих и чужих на
имеющихся парах и печатает, где проходит граница и проходит ли вообще.

РЕЗУЛЬТАТ ЗАМЕРА (20 пар, 20 своих против 20 чужих):

    признак                медиана свои / чужие     AUC
    шаг дуги                   0.34 / 0.58 мм      0.718
    состав ряда (Жаккар)       0.92 / 0.89         0.578
    размер зуба                0.30 / 0.37 мм      0.468
    шаг + размер вместе                            0.671

Сигнал есть только у шага дуги, и он слабый: при пороге 0.3 мм принимается
40% своих ценой 5% чужих, при 0.4 мм — 70% своих ценой уже 35% чужих.
Разбросы полностью перекрываются (у обеих групп 0.18..1.07 мм).

**Вывод: подтвердить тождество пациента этими признаками нельзя.** Признак
состава почти не работает по конкретной причине — у всех пациентов набора
figshare 26965903 ПОЛНЫЕ ряды по 28 зубов (возраст 19-28 лет), различать их
по составу нечем. У пациентов клиники, где есть удалённые зубы, импланты и
коронки, состав должен работать заметно лучше, но проверить это на данном
наборе невозможно. До тех пор тождество подтверждается организационно — по
карточке, а не расчётом.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_cbct_identity.py
    .venv/Scripts/python.exe research/scripts/ios_cbct_identity.py --json out.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402
import ios_cbct_register as reg  # noqa: E402

MIN_COMMON = 6          # меньше — судить не о чем


def _centroids(d: dict) -> dict:
    return {t: np.asarray(v).mean(axis=0) for t, v in d.items()}


def _spread(d: dict) -> dict:
    """Поперечник коронки: медиана расстояний от центра, устойчиво к выбросам."""
    out = {}
    for t, v in d.items():
        p = np.asarray(v)
        out[t] = float(np.median(np.linalg.norm(p - p.mean(axis=0), axis=1)))
    return out


def _steps(cen: dict) -> dict:
    """Расстояния между центрами соседних по номеру зубов, мм."""
    out = {}
    for t in cen:
        nxt = t + 1
        if nxt % 10 <= 8 and nxt in cen:
            out[(t, nxt)] = float(np.linalg.norm(cen[nxt] - cen[t]))
    return out


def compare(cbct_case: str, ios_case: str, model: str | None = None) -> dict:
    tgt = reg.cbct_crowns(cbct_case, model)
    src = reg.ios_crowns(ios_case)
    both = sorted(set(tgt) & set(src))
    union = sorted(set(tgt) | set(src))
    jac = len(both) / len(union) if union else 0.0

    c_t, c_s = _centroids(tgt), _centroids(src)
    st_t, st_s = _steps(c_t), _steps(c_s)
    common_steps = sorted(set(st_t) & set(st_s))
    step_err = (float(np.median([abs(st_t[k] - st_s[k]) for k in common_steps]))
                if len(common_steps) >= MIN_COMMON else float("nan"))

    sp_t, sp_s = _spread(tgt), _spread(src)
    size_err = (float(np.median([abs(sp_t[t] - sp_s[t]) for t in both]))
                if len(both) >= MIN_COMMON else float("nan"))

    return {
        "cbct": cbct_case, "ios": ios_case,
        "teeth_cbct": len(tgt), "teeth_ios": len(src),
        "common": len(both), "jaccard": round(jac, 3),
        "n_steps": len(common_steps),
        "step_err_mm": None if np.isnan(step_err) else round(step_err, 2),
        "size_err_mm": None if np.isnan(size_err) else round(size_err, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="sp")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    ids = sorted(c["case_id"][4:-5] for c in cb.cases.values()
                 if c["case_id"].startswith("pair")
                 and c["case_id"].endswith("_cbct"))
    ready = [p for p in ids
             if (reg.IOS_CASES / f"pair{p}_ios" / "scan.glb").is_file()]
    if len(ready) < 2:
        raise SystemExit(f"готовых пар {len(ready)}, нужно ≥2")
    print(f"пар готово: {len(ready)}\n")

    own, cross = [], []
    for i, pid in enumerate(ready):
        try:
            r = compare(f"pair{pid}_cbct", f"pair{pid}_ios", a.model)
        except SystemExit as exc:
            print(f"  {pid}: {exc}")
            continue
        r["kind"] = "свой"
        own.append(r)
        other = ready[(i + 1) % len(ready)]
        if other != pid:
            try:
                rc = compare(f"pair{pid}_cbct", f"pair{other}_ios", a.model)
                rc["kind"] = "чужой"
                cross.append(rc)
            except SystemExit:
                pass

    def show(rows, title):
        print(f"=== {title} ===")
        for r in rows:
            print(f"  {r['cbct'][4:7]} + {r['ios'][4:7]}: общих {r['common']:>2}, "
                  f"Жаккар {r['jaccard']:.2f}, шаг дуги {r['step_err_mm']} мм, "
                  f"размер {r['size_err_mm']} мм")
        print()
    show(own, "СВОИ пары")
    show(cross, "ЧУЖИЕ пары (контроль)")

    print(f"{'='*62}")
    print("РАЗДЕЛЕНИЕ признаков")
    print(f"{'='*62}")
    for key, name, better in (("jaccard", "Жаккар по составу", "больше"),
                              ("step_err_mm", "ошибка шага дуги, мм", "меньше"),
                              ("size_err_mm", "ошибка размера, мм", "меньше")):
        a_own = np.array([r[key] for r in own if r[key] is not None], float)
        a_cr = np.array([r[key] for r in cross if r[key] is not None], float)
        if a_own.size == 0 or a_cr.size == 0:
            continue
        if better == "больше":
            gap = a_own.min() - a_cr.max()
        else:
            gap = a_cr.min() - a_own.max()
        print(f"  {name:<24} свои {np.median(a_own):>6.2f} "
              f"({a_own.min():.2f}..{a_own.max():.2f})   "
              f"чужие {np.median(a_cr):>6.2f} ({a_cr.min():.2f}..{a_cr.max():.2f})")
        print(f"  {'':24} зазор {gap:+.2f} — "
              + ("РАЗДЕЛЯЕТ" if gap > 0 else "не разделяет"))
    if a.json:
        a.json.write_text(json.dumps({"own": own, "cross": cross},
                                     ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"\n  подробности: {a.json}")


if __name__ == "__main__":
    main()
