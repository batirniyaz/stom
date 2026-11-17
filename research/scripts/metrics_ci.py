"""Доверительные интервалы для метрик по таблице результатов.

Зачем. «Точность 96.9%» посчитана на выборке; на другой такой же выборке
число было бы другим. Интервал честно показывает, в каком диапазоне лежит
настоящая точность, и этого требуют стандарты отчётности медицинского ИИ
(STARD-AI, TRIPOD+AI, CLAIM). Без интервала наше число ничем не отличается от
чужих «97%», которые мы сами критикуем за отсутствие методологии.

ГЛАВНОЕ: БУТСТРЕП ПО ПАЦИЕНТАМ, А НЕ ПО ЗУБАМ. Зубы одного человека зависимы
— одна и та же маска, один аппарат, одна анатомия. Выборка по зубам считает
их независимыми наблюдениями и даёт искусственно узкий интервал, то есть врёт
в нашу пользу. Единица анализа — пациент, и она записана в самом файле
результатов (`unit_of_analysis`).

Широкий интервал НЕ прячем, а помечаем. У клыков случаев мало, интервал там
заведомо шире, и это надо показать: «мало случаев, оценка ненадёжна» —
проверяемое утверждение, а усреднённое число по всем зубам скрыло бы его.

Вход — JSON от `cbct_eval_external.py --json`. Моделей не требует.

Запуск:

    .venv/Scripts/python.exe research/scripts/metrics_ci.py \\
        research/output/metrics/teeth_sp_dentvoxel.json \\
        --out research/output/metrics/teeth_sp_dentvoxel_ci.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ITERS = 2000          # для долей хватает с запасом; для mAP берут 500
LOW, HIGH = 2.5, 97.5  # 95-процентный интервал

# Широким считаем интервал шире 10 процентных пунктов ЛИБО группу меньше 30
# зубов. Оба признака про одно — наблюдений мало, числу верить нельзя.
WIDE_PP = 10.0
MIN_TEETH = 30

GROUPS = {
    "резцы": (1, 2),
    "клыки": (3,),
    "премоляры": (4, 5),
    "моляры": (6, 7, 8),
}


def _pos(fdi: int) -> int:
    """Позиция зуба в квадранте: 1-8. Тип зуба определяется ею."""
    return abs(int(fdi)) % 10


def _counts(case: dict, positions: tuple | None = None) -> tuple[int, int, int]:
    """TP, FP, FN по одному пациенту, при желании только по типу зубов."""
    dice = case.get("dice") or {}
    missed = case.get("missed") or []
    extra = case.get("extra") or []
    if positions is None:
        return len(dice), len(extra), len(missed)
    tp = sum(1 for t in dice if _pos(t) in positions)
    fp = sum(1 for t in extra if _pos(t) in positions)
    fn = sum(1 for t in missed if _pos(t) in positions)
    return tp, fp, fn


def _dices(case: dict, positions: tuple | None = None) -> list[float]:
    d = case.get("dice") or {}
    if positions is None:
        return [float(v) for v in d.values()]
    return [float(v) for t, v in d.items() if _pos(t) in positions]


def _ci(samples: list[float]) -> tuple[float, float]:
    a = np.asarray([s for s in samples if not np.isnan(s)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(a, LOW)), float(np.percentile(a, HIGH))


def bootstrap(cases: list[dict], positions: tuple | None = None,
              iters: int = ITERS, seed: int = 0) -> dict:
    """Метрики с интервалом. Пациенты берутся с возвращением."""
    rng = np.random.default_rng(seed)
    n = len(cases)

    tp0 = fp0 = fn0 = 0
    for c in cases:
        a, b, d = _counts(c, positions)
        tp0 += a
        fp0 += b
        fn0 += d
    dice0 = [x for c in cases for x in _dices(c, positions)]

    rec, prec, dmed = [], [], []
    for _ in range(iters):
        pick = rng.integers(0, n, n)
        tp = fp = fn = 0
        dd = []
        for i in pick:
            a, b, d = _counts(cases[i], positions)
            tp += a
            fp += b
            fn += d
            dd += _dices(cases[i], positions)
        rec.append(100.0 * tp / (tp + fn) if tp + fn else float("nan"))
        prec.append(100.0 * tp / (tp + fp) if tp + fp else float("nan"))
        dmed.append(float(np.median(dd)) if dd else float("nan"))

    teeth = tp0 + fn0
    out = {
        "пациентов": n,
        "зубов_в_эталоне": teeth,
        "полнота": {"значение": 100.0 * tp0 / (tp0 + fn0) if tp0 + fn0 else None},
        "точность": {"значение": 100.0 * tp0 / (tp0 + fp0) if tp0 + fp0 else None},
        "dice_медиана": {"значение": float(np.median(dice0)) if dice0 else None},
    }
    for key, samples in (("полнота", rec), ("точность", prec),
                         ("dice_медиана", dmed)):
        lo, hi = _ci(samples)
        out[key]["ди_низ"] = lo
        out[key]["ди_верх"] = hi
    # Ненадёжность определяем по САМОЙ ШИРОКОЙ из долей: если хоть одна
    # разъехалась, группе верить нельзя целиком.
    widths = [out[k]["ди_верх"] - out[k]["ди_низ"]
              for k in ("полнота", "точность")
              if out[k]["ди_верх"] == out[k]["ди_верх"]]
    out["ненадёжно"] = bool(teeth < MIN_TEETH
                            or (widths and max(widths) > WIDE_PP))
    out["почему_ненадёжно"] = (
        f"зубов меньше {MIN_TEETH}" if teeth < MIN_TEETH else
        f"интервал шире {WIDE_PP:.0f} п.п." if out["ненадёжно"] else None)
    return out


def _line(name: str, m: dict, unit: str = "%") -> str:
    v, lo, hi = m["значение"], m["ди_низ"], m["ди_верх"]
    if v is None:
        return f"  {name}: нет данных"
    if unit == "%":
        s = f"  {name}: {v:.1f}% (95% ДИ {lo:.1f}–{hi:.1f}%)"
    else:
        s = f"  {name}: {v:.3f} (95% ДИ {lo:.3f}–{hi:.3f})"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--iters", type=int, default=ITERS)
    a = ap.parse_args()

    data = json.loads(a.results.read_text(encoding="utf-8"))
    cases = data.get("cases") or []
    if not cases:
        raise SystemExit("в файле нет поля cases")

    print(f"модель: {data.get('model')}  выборка: {data.get('dataset')}")
    print(f"единица анализа: {data.get('unit_of_analysis')}")
    print(f"бутстреп: {a.iters} итераций по пациентам, перцентили "
          f"{LOW}/{HIGH}\n")

    report = {
        "источник": str(a.results.name),
        "модель": data.get("model"),
        "выборка": data.get("dataset"),
        "единица_анализа": data.get("unit_of_analysis"),
        "итераций": a.iters,
        "посчитано": datetime.now().isoformat(timespec="seconds"),
        "всего": bootstrap(cases, None, a.iters),
        "по_типам_зубов": {},
    }

    tot = report["всего"]
    print(f"ВСЕ ЗУБЫ — n = {tot['пациентов']} пациентов, "
          f"{tot['зубов_в_эталоне']} зубов")
    print(_line("полнота", tot["полнота"]))
    print(_line("точность", tot["точность"]))
    print(_line("Dice (медиана)", tot["dice_медиана"], unit=""))

    print("\nПО ТИПАМ ЗУБОВ")
    for name, pos in GROUPS.items():
        g = bootstrap(cases, pos, a.iters)
        report["по_типам_зубов"][name] = g
        flag = ("   <- мало случаев, оценка ненадёжна"
                if g["ненадёжно"] else "")
        print(f"\n  {name}: {g['зубов_в_эталоне']} зубов{flag}")
        print("  " + _line("полнота", g["полнота"]).strip())
        print("  " + _line("точность", g["точность"]).strip())
        print("  " + _line("Dice", g["dice_медиана"], unit="").strip())

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"\nтаблица для интерфейса: {a.out}")


if __name__ == "__main__":
    main()
