"""Прогресс обучения ToothFairy3 по группам классов.

nnU-Net печатает Pseudo dice одним массивом на 77 классов — глазами в нём
ничего не видно. Скрипт разбивает его по смыслу и отдельно показывает пульпу:
именно её мы добавляли ради просьбы рентгенолога, и если она не учится, это
надо увидеть на ранних эпохах, а не через 16 часов.

Раскладка train-id (после convert_toothfairy3.py):
      1-10  структуры (челюсти, каналы, пазухи, глотка, реставрации)
     11-42  зубы FDI
     43-45  резцовые и язычный каналы
     46-77  пульпа (orig 111-148)

Запуск:
    .venv/Scripts/python.exe research/scripts/tf3_train_progress.py
    .venv/Scripts/python.exe research/scripts/tf3_train_progress.py --last 5
"""

import argparse
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "output" / "nnunet_tf3_train.log"

# (название, первый train-id, последний train-id) — границы включительно
GROUPS = [
    ("структуры 1-10", 1, 10),
    ("зубы 11-42", 11, 42),
    ("каналы 43-45", 43, 45),
    ("ПУЛЬПА 46-77", 46, 77),
]


def parse(text: str) -> tuple[list, list, list]:
    """-> (номера эпох, времена эпох, массивы pseudo dice)."""
    epochs = [int(x) for x in re.findall(r": Epoch (\d+)\s*$", text, re.M)]
    times = [float(x) for x in re.findall(r"Epoch time: ([\d.]+) s", text)]
    dices = []
    for line in re.findall(r"Pseudo dice \[(.*?)\]", text, re.S):
        vals = []
        for tok in re.findall(r"np\.float32\(([^)]+)\)", line):
            vals.append(float("nan") if tok == "nan" else float(tok))
        if vals:
            dices.append(np.array(vals, dtype=float))
    return epochs, times, dices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=3, help="сколько последних эпох показать")
    ap.add_argument("--log", default=str(LOG))
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="ignore")
    epochs, times, dices = parse(text)
    if not dices:
        print("эпох с Pseudo dice пока нет")
        return

    done = len(dices)
    if len(times) > 1:
        avg = float(np.mean(times[1:]))
        left = (1000 - done) * avg / 3600
        print(f"эпох: {done}/1000   среднее {avg:.1f} с/эпоху   "
              f"осталось ~{left:.1f} ч")
    print(f"классов в массиве dice: {len(dices[-1])}\n")

    print(f"{'эпоха':>6}", end="")
    for name, _, _ in GROUPS:
        print(f"{name:>16}", end="")
    print()
    for d, ep in list(zip(dices, range(done)))[-args.last:]:
        print(f"{ep:>6}", end="")
        for _, lo, hi in GROUPS:
            sl = d[lo - 1:hi]                  # индекс i = класс i+1
            val = np.nanmean(sl) if not np.all(np.isnan(sl)) else float("nan")
            print(f"{val:>16.3f}", end="")
        print()

    # худшие классы последней эпохи — где именно проседает
    d = dices[-1]
    order = np.argsort(np.where(np.isnan(d), 2.0, d))
    print("\nхудшие 8 классов (train-id: dice):")
    print("  " + "  ".join(f"{int(i)+1}:{d[i]:.2f}" for i in order[:8]))
    pulp = d[45:77]
    alive = int(np.sum(~np.isnan(pulp) & (pulp > 0.05)))
    print(f"\nпульпа: обучается {alive} классов из 32 "
          f"(dice > 0.05), медиана {np.nanmedian(pulp):.3f}")


if __name__ == "__main__":
    main()
