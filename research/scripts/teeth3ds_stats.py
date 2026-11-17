"""Статистика Teeth3DS+: что реально лежит в датасете.

От этих цифр зависят решения препроцессинга: сколько точек сэмплировать
(разброс плотности сеток), какие классы вообще встречаются (есть ли восьмёрки),
у скольких пациентов есть обе челюсти (нужно для анализа прикуса).

Запуск:
    .venv/Scripts/python.exe research/scripts/teeth3ds_stats.py
    .venv/Scripts/python.exe research/scripts/teeth3ds_stats.py --limit 200
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "research_ios" / "datasets" / "teeth3ds"
SCANS = ROOT / "scans"


def scan_one(p: Path):
    """-> (челюсть, пациент, число вершин, набор зубов)"""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    lab = np.asarray(d["labels"], dtype=np.int16)
    teeth = sorted({int(x) for x in np.unique(lab) if x != 0})
    return d.get("jaw"), d.get("id_patient") or p.parent.name, len(lab), teeth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="ограничить число сканов")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    files = sorted(p for p in SCANS.rglob("*.json")
                   if not p.name.endswith("__kpt.json"))
    if args.limit:
        files = files[:args.limit]
    print(f"файлов разметки: {len(files)}")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, r in enumerate(ex.map(scan_one, files, chunksize=8), 1):
            if r:
                rows.append(r)
            if n % 400 == 0:
                print(f"  {n}/{len(files)}")

    jaws = Counter(r[0] for r in rows)
    patients = {}
    for jaw, pid, _, _ in rows:
        patients.setdefault(pid, set()).add(jaw)
    both = sum(1 for v in patients.values() if len(v) >= 2)
    print(f"\nсканов: {len(rows)}   {dict(jaws)}")
    print(f"пациентов: {len(patients)}, из них с обеими челюстями: {both} "
          f"({both/max(len(patients),1)*100:.0f}%)")

    verts = np.array([r[2] for r in rows])
    print(f"\nвершин в сетке: медиана={int(np.median(verts)):,}  "
          f"p5={int(np.percentile(verts,5)):,}  p95={int(np.percentile(verts,95)):,}  "
          f"мин={verts.min():,}  макс={verts.max():,}")

    nteeth = np.array([len(r[3]) for r in rows])
    print(f"зубов на скан: медиана={int(np.median(nteeth))}  "
          f"мин={nteeth.min()}  макс={nteeth.max()}")

    freq = Counter()
    for _, _, _, teeth in rows:
        freq.update(teeth)
    print("\nчастота зубов по FDI (в скольких сканах встречается):")
    for q in (1, 2, 3, 4):
        line = "  ".join(f"{q}{i}:{freq.get(q*10+i,0):>4}" for i in range(1, 9))
        print(f"   квадрант {q}:  {line}")
    wisdom = sum(freq.get(q * 10 + 8, 0) for q in (1, 2, 3, 4))
    print(f"\nвосьмёрки суммарно: {wisdom} появлений на {len(rows)} сканов")

    # доля вершин десны — насколько несбалансированы классы
    print("\nразбиение по числу зубов (гистограмма):")
    h = Counter(int(x) for x in nteeth)
    for k in sorted(h):
        print(f"   {k:>2} зубов: {'#' * min(h[k]//10, 60)} {h[k]}")


if __name__ == "__main__":
    main()
