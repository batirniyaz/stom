"""Сравнение моделей КЛКТ по тому, что врач замечает первым: сколько зубов.

Зачем. 27.08 врач открыл один и тот же DICOM-каталог в стороннем вьюере и в
нашем демо и увидел разницу: у нас пропал почти весь четвёртый квадрант.
Разбор показал, что данные и конвертация ни при чём — модели просто находят
РАЗНОЕ число зубов на одних и тех же данных, и ни одна не лучшая:

    кейс        ours  pretrained  tf3
    client2       23        20     23
    client4       26        32     29
    client5       26        30     23

Такое сравнение делалось вручную и потому не делалось. Здесь оно одной
командой — и по маскам, которые уже лежат на диске, и с инференсом новой
модели, которой маски ещё нет.

Считаем то, что проверяемо: наличие зуба в маске и его объём. Правды о том,
сколько зубов у пациента на самом деле, у нас нет (разметки клиники нет),
поэтому за ориентир берём ОБЪЕДИНЕНИЕ моделей: зуб, найденный хоть кем-то,
скорее есть, чем нет — выдумать зуб на пустом месте труднее, чем пропустить.
Это ориентир, а не истина, и в выводе он так и назван.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_compare_models.py
    .venv/Scripts/python.exe research/scripts/cbct_compare_models.py client5
    .venv/Scripts/python.exe research/scripts/cbct_compare_models.py --predict tf3_resencl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

MIN_TOOTH_VOX = 300        # тот же порог присутствия, что в слое находок
DEFAULT_CASES = ["client2", "client4", "client5"]


def teeth_of(path: str) -> dict:
    """Номера зубов в маске и их объёмы в вокселях."""
    m = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.int16)
    out = {}
    for t in cb.TEETH_CLASSES:
        n = int((m == t).sum())
        if n >= MIN_TOOTH_VOX:
            out[t] = n
    return out


def predict_missing(case_id: str, model: str) -> str | None:
    """Инференс модели, если её маски у кейса ещё нет. Возвращает путь."""
    c = cb.get_case(case_id)
    if c is None:
        print(f"  {case_id}: кейса нет")
        return None
    have = c["masks"].get(model)
    if have and Path(have).is_file():
        return have
    if model not in cb.MODELS:
        print(f"  модель {model} не объявлена в cbct_backend.MODELS")
        return None
    print(f"  {case_id}: маски модели {model} нет, считаем (2-4 минуты)…",
          flush=True)
    path = str(cb._run_inference(c, model))
    c["masks"][model] = path
    cb._save_meta(c)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case_id", nargs="*", help=f"по умолчанию {DEFAULT_CASES}")
    ap.add_argument("--predict", action="append", default=[],
                    help="посчитать этой моделью, если маски нет")
    a = ap.parse_args()
    cases = a.case_id or DEFAULT_CASES

    for model in a.predict:
        for case in cases:
            predict_missing(case, model)

    print(f"\n{'кейс':<12} {'модель':<14} {'зубов':>6} {'верх':>5} {'низ':>5}"
          f"   не нашла из объединения")
    totals: dict[str, list] = {}
    for case in cases:
        c = cb.get_case(case)
        if c is None:
            print(f"{case:<12} кейс не найден")
            continue
        found = {}
        for model, path in sorted(c["masks"].items()):
            if path and Path(path).is_file():
                found[model] = teeth_of(path)
        if not found:
            print(f"{case:<12} масок нет")
            continue
        union = set().union(*(set(v) for v in found.values()))
        for model, teeth in found.items():
            up = [t for t in teeth if t // 10 in (1, 2)]
            lo = [t for t in teeth if t // 10 in (3, 4)]
            miss = sorted(union - set(teeth))
            totals.setdefault(model, []).append(len(teeth))
            print(f"{case:<12} {model:<14} {len(teeth):>6} {len(up):>5} "
                  f"{len(lo):>5}   {miss if miss else '—'}")
        print(f"{'':<12} {'объединение':<14} {len(union):>6}"
              f"   ← ориентир, а не истина: разметки клиники нет")
        print()

    if totals:
        print("в среднем по кейсам:")
        for model, counts in sorted(totals.items(),
                                    key=lambda kv: -np.mean(kv[1])):
            print(f"   {model:<14} {np.mean(counts):>5.1f} зуба "
                  f"({', '.join(str(c) for c in counts)})")
        print("\nБольше зубов — НЕ автоматически лучше: модель может выдумать "
              "лишний\nобъект. Смотреть вместе с объёмами и глазами по срезам.")


if __name__ == "__main__":
    main()
