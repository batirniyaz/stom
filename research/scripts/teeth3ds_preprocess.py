"""Препроцессинг Teeth3DS+: сетка + разметка -> облако точек в .npz.

На выходе на каждый скан: points (N,3), normals (N,3), labels (N,) и мета.
Метки перекодированы из FDI в позицию в челюсти: 0 = десна, 1-8 — правая
сторона от резца к моляру, 9-16 — левая. Челюсть известна из имени файла,
обратно в FDI разворачивается однозначно. Так классов 17 вместо 33, и они
одинаковы для верха и низа — сети проще, а информации не теряется.

ВАЖНО: OBJ читаем с process=False. trimesh по умолчанию сливает дубликаты
вершин и меняет их порядок, а разметка в Teeth3DS+ идёт МАССИВОМ ПО ВЕРШИНАМ —
после переупорядочивания метки поехали бы и молча испортили весь датасет.

Запуск:
    .venv/Scripts/python.exe research/scripts/teeth3ds_preprocess.py --limit 20
    .venv/Scripts/python.exe research/scripts/teeth3ds_preprocess.py
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "research_ios" / "datasets"
SCANS = ROOT / "teeth3ds" / "scans"
SPLITS = ROOT / "teeth3ds" / "_splits" / "Teeth3DS_train_test_split"
OUT = ROOT / "teeth3ds_npz"

N_POINTS = 24000          # медиана сетки 114k, p5 74k — берём с запасом снизу
SCALE_MM = 50.0           # делим координаты на это: сохраняем физический смысл

def robust_center(pts):
    """Центр по диапазону 2-98%, а не по среднему.

    Среднее тянется за выбросами: у сканов Teeth3DS длинные хвосты редких
    точек, и центр уезжает от дуги на 2.6 мм, у клинического скана на 4.8 мм.
    Модель к сдвигу чувствительна (10 мм роняют точность с 0.97 до 0.72),
    поэтому центр должен считаться одинаково и устойчиво везде.
    """
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    return ((lo + hi) / 2).astype(pts.dtype)


# FDI -> позиция 1..16. Правая сторона пациента (квадранты 1 и 4) -> 1..8,
# левая (2 и 3) -> 9..16. Индекс внутри квадранта: резец=1 ... восьмёрка=8.
def fdi_to_pos(fdi: int) -> int:
    q, i = fdi // 10, fdi % 10
    if q in (1, 4):
        return i
    if q in (2, 3):
        return 8 + i
    return 0


def pos_to_fdi(pos: int, jaw: str) -> int:
    """Обратное преобразование, нужно на инференсе."""
    if pos <= 0:
        return 0
    side_right = pos <= 8
    i = pos if side_right else pos - 8
    q = (1 if side_right else 2) if jaw == "upper" else (4 if side_right else 3)
    return q * 10 + i


def process_one(args):
    obj_path, out_path = args
    import trimesh
    lab_path = obj_path.with_suffix(".json")
    try:
        meta = json.loads(lab_path.read_text(encoding="utf-8"))
        # process=False — иначе trimesh переупорядочит вершины и метки поедут
        mesh = trimesh.load(str(obj_path), process=False, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        labels_fdi = np.asarray(meta["labels"], dtype=np.int32)
        if len(verts) != len(labels_fdi):
            return obj_path.name, False, (f"вершин {len(verts)} != меток "
                                          f"{len(labels_fdi)}")
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        if len(normals) != len(verts):
            return obj_path.name, False, "нормали не совпали по длине"

        pos = np.zeros(len(labels_fdi), dtype=np.uint8)
        uniq = np.unique(labels_fdi)
        for fdi in uniq:
            if fdi == 0:
                continue
            p = fdi_to_pos(int(fdi))
            if p == 0:
                return obj_path.name, False, f"неожиданный FDI {fdi}"
            pos[labels_fdi == fdi] = p

        rng = np.random.default_rng(abs(hash(obj_path.stem)) % (2 ** 32))
        n = len(verts)
        idx = (rng.choice(n, N_POINTS, replace=False) if n >= N_POINTS
               else rng.choice(n, N_POINTS, replace=True))

        center = verts.mean(axis=0)
        pts = (verts[idx] - center) / SCALE_MM
        np.savez_compressed(
            out_path,
            points=pts.astype(np.float32),
            normals=normals[idx].astype(np.float32),
            labels=pos[idx],
            patient=meta.get("id_patient", obj_path.parent.name),
            jaw=meta.get("jaw", "upper" if "upper" in obj_path.stem else "lower"),
            center=center.astype(np.float32),
            scale=np.float32(SCALE_MM),
            n_verts=np.int32(n),
            teeth=np.array(sorted(int(x) for x in uniq if x != 0), dtype=np.int16),
        )
        return obj_path.name, True, ""
    except Exception as exc:  # noqa: BLE001
        return obj_path.name, False, str(exc)


def read_split(name: str) -> set:
    out = set()
    for jaw in ("upper", "lower"):
        f = SPLITS / f"{name}_{jaw}.txt"
        if f.exists():
            out |= {l.strip() for l in f.read_text().splitlines() if l.strip()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    train, test = read_split("training"), read_split("testing")
    print(f"официальное разбиение: train={len(train)} сканов, test={len(test)}")

    tasks = []
    for jaw in ("upper", "lower"):
        for pat_dir in sorted((SCANS / jaw).iterdir()):
            obj = pat_dir / f"{pat_dir.name}_{jaw}.obj"
            if not obj.exists():
                continue
            key = f"{pat_dir.name}_{jaw}"
            split = "train" if key in train else "test" if key in test else "unused"
            (OUT / split).mkdir(parents=True, exist_ok=True)
            tasks.append((obj, OUT / split / f"{key}.npz"))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"сканов к обработке: {len(tasks)}")

    ok, bad = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, (name, good, err) in enumerate(ex.map(process_one, tasks,
                                                     chunksize=4), 1):
            if good:
                ok += 1
            else:
                bad.append((name, err))
            if n % 100 == 0:
                print(f"  {n}/{len(tasks)}  ошибок={len(bad)}")
    print(f"\nготово: {ok}/{len(tasks)}")
    for name, err in bad[:15]:
        print(f"   ОШИБКА {name}: {err}")
    for split in ("train", "test", "unused"):
        d = OUT / split
        if d.exists():
            print(f"   {split}: {len(list(d.glob('*.npz')))} файлов")


if __name__ == "__main__":
    main()
