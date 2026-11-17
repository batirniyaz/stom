"""Перенумерация меток ToothFairy3 в подряд идущие id (требование nnU-Net).

ToothFairy3 нумерует классы по FDI с дырами: зубы 11-18, 21-28, 31-38, 41-48,
пульпа = зуб+100 (111-148), плюс каналы 103-105. nnU-Net требует, чтобы id шли
подряд (0,1,2,...) — иначе verify_dataset_integrity падает на
«Labels must be in consecutive order».

Забивать дыры заглушками NA_, как сделано в ToothFairy2, здесь нельзя: максимум
148 против 78 реальных классов, то есть сеть получила бы 149 выходных каналов
вместо 78 — вдвое больше памяти на softmax при том же результате.

Скрипт: сохраняет оригинальные маски в labelsTr_fdi/, переписывает labelsTr/ с
подряд идущими id, чинит dataset.json и кладёт label_map.json с обратным
преобразованием. На инференсе маска возвращается к FDI (см. cbct_backend), так
что весь остальной код продолжает работать с привычными номерами зубов.

Запуск:
    .venv/Scripts/python.exe research/scripts/convert_toothfairy3.py
    .venv/Scripts/python.exe research/scripts/convert_toothfairy3.py --check
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2] / "research_3d"
DS = ROOT / "nnUNet_raw" / "Dataset113_ToothFairy3"
LABELS = DS / "labelsTr"
BACKUP = DS / "labelsTr_fdi"
MAP_FILE = DS / "label_map.json"


def build_map(dataset_json: dict) -> tuple[dict, dict]:
    """orig_id -> train_id (подряд) и обратно."""
    orig_ids = sorted(int(v) for v in dataset_json["labels"].values())
    fwd = {o: i for i, o in enumerate(orig_ids)}
    back = {i: o for o, i in fwd.items()}
    return fwd, back


def _remap_one(args) -> tuple[str, bool, str, int, list]:
    """-> (имя, успех, ошибка, число классов в маске, отброшенные id)."""
    src, dst, lut, known = args
    try:
        img = sitk.ReadImage(str(src))
        arr = sitk.GetArrayFromImage(img)
        present = np.unique(arr)
        # Одиночные воксели с id вне dataset.json (60, 100, 160 — суммарно
        # 5 вокселей на весь датасет) — шум разметки, гасим в фон.
        dropped = [int(v) for v in present if int(v) not in known]
        # таблица должна покрывать и lut, и реальный максимум файла
        big = np.zeros(max(len(lut), int(present.max()) + 1), dtype=np.uint8)
        big[:len(lut)] = lut               # неизвестные id остаются нулями
        out = big[arr]
        new = sitk.GetImageFromArray(out.astype(np.uint8))
        new.CopyInformation(img)          # геометрия обязана совпасть со снимком
        sitk.WriteImage(new, str(dst), useCompression=True)
        # пульпа в исходной нумерации — 111..148; считаем, чтобы понять,
        # в каких подмножествах она вообще размечена
        pulp = sum(1 for v in present if 111 <= int(v) <= 148)
        return src.name, True, "", int(len(present)), dropped, pulp
    except Exception as exc:  # noqa: BLE001
        return src.name, False, str(exc), 0, [], 0


def convert(workers: int, force: bool) -> None:
    dj = json.loads((DS / "dataset.json").read_text(encoding="utf-8"))
    fwd, back = build_map(dj)
    print(f"классов: {len(fwd)}  (id 0..{max(back)})")
    gaps = [o for o in sorted(fwd) if o != fwd[o]]
    print(f"перенумеровано: {len(gaps)} меток, первая сдвинутая — "
          f"{gaps[0] if gaps else '—'} -> {fwd[gaps[0]] if gaps else '—'}")

    if BACKUP.exists() and not force:
        print(f"ОТМЕНА: {BACKUP.name} уже есть — конвертация выполнялась. "
              f"Повтор с оригиналов: --force")
        sys.exit(1)
    if not BACKUP.exists():
        files = sorted(LABELS.glob("*.nii.gz"))
        if not files:
            sys.exit(f"нет масок в {LABELS}")
        print(f"масок: {len(files)}  -> резервная копия в {BACKUP.name}")
        shutil.copytree(LABELS, BACKUP)
    else:
        print(f"повтор с оригиналов из {BACKUP.name}")

    files = sorted(BACKUP.glob("*.nii.gz"))
    lut = np.zeros(max(fwd) + 1, dtype=np.uint8)
    for o, i in fwd.items():
        lut[o] = i
    known = set(fwd)

    tasks = [(BACKUP / f.name, LABELS / f.name, lut, known) for f in files]
    ok, dropped_all = 0, {}
    per_prefix: dict[str, list] = {}
    pulp_prefix: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (name, good, err, ncls, dropped, pulp) in enumerate(
                ex.map(_remap_one, tasks), 1):
            if good:
                ok += 1
                pref = name.split("_")[0]
                per_prefix.setdefault(pref, []).append(ncls)
                pulp_prefix.setdefault(pref, []).append(pulp)
            else:
                print(f"  ОШИБКА {name}: {err}")
            if dropped:
                dropped_all[name] = dropped
            if n % 100 == 0:
                print(f"  {n}/{len(tasks)}")
    print(f"переписано: {ok}/{len(files)}")
    if dropped_all:
        print(f"погашено в фон (шум разметки) в {len(dropped_all)} файлах:")
        for k, v in dropped_all.items():
            print(f"   {k}: id {v}")
    print("\nклассов в маске (полнота разметки по подмножествам):")
    for pref, counts in sorted(per_prefix.items()):
        c = sorted(counts)
        p = sorted(pulp_prefix[pref])
        with_pulp = sum(1 for x in p if x > 0)
        print(f"   {pref}: файлов={len(c):3d}  классов мин/медиана/макс="
              f"{c[0]:2d}/{c[len(c)//2]:2d}/{c[-1]:2d}   "
              f"с пульпой: {with_pulp:3d} файлов, пульп медиана={p[len(p)//2]:2d}")
    if ok != len(files):
        sys.exit("часть файлов не сконвертирована — препроцессинг не запускать")

    # dataset.json с подряд идущими id
    dj["labels"] = {name: fwd[int(v)] for name, v in dj["labels"].items()}
    (DS / "dataset.json").write_text(
        json.dumps(dj, ensure_ascii=False, indent=1), encoding="utf-8")
    MAP_FILE.write_text(json.dumps(
        {"train_to_orig": {str(i): o for i, o in back.items()},
         "orig_to_train": {str(o): i for o, i in fwd.items()},
         "note": "orig — нумерация ToothFairy3 (FDI, пульпа = зуб+100)"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"обновлены dataset.json и {MAP_FILE.name}")


def check() -> None:
    """Сверяет, что в масках только ожидаемые id и геометрия совпадает."""
    dj = json.loads((DS / "dataset.json").read_text(encoding="utf-8"))
    expect = set(int(v) for v in dj["labels"].values())
    print(f"ожидаем id 0..{max(expect)} ({len(expect)} классов)")
    files = sorted(LABELS.glob("*.nii.gz"))
    for f in files[:5] + files[-2:]:
        lab = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(lab)
        vals = set(np.unique(arr).tolist())
        img = sitk.ReadImage(str(DS / "imagesTr" / f.name.replace(".nii.gz", "_0000.nii.gz")))
        geom = (lab.GetSize() == img.GetSize() and
                np.allclose(lab.GetSpacing(), img.GetSpacing()))
        extra = vals - expect
        print(f"  {f.name}: значений={len(vals)} max={max(vals)} "
              f"лишних={extra or 'нет'} геометрия={'ок' if geom else 'РАСХОЖДЕНИЕ'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="повторить конвертацию с оригиналов в labelsTr_fdi")
    ap.add_argument("--check", action="store_true",
                    help="только проверить текущее состояние масок")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        convert(args.workers, args.force)


if __name__ == "__main__":
    main()
