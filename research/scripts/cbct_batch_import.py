"""Пакетный ввоз КЛКТ с флешки: серии -> кейсы -> сегментация -> разметка.

    .venv\\Scripts\\python.exe research\\scripts\\cbct_batch_import.py --src E:\\CBCT --dry-run
    .venv\\Scripts\\python.exe research\\scripts\\cbct_batch_import.py --src E:\\CBCT
    .venv\\Scripts\\python.exe research\\scripts\\cbct_batch_import.py --src E:\\CBCT --no-prepare

ЗАЧЕМ. Снимки приходят пачками по 40-50 штук — флешкой или телеграмом, — и
загружать их по одному через демо бессмысленно: каждый занимает 300-600 МБ, а
бесплатный Cloudflare режет тело запроса на 100 МБ. Здесь пачка заливается
локально: ввоз, сегментация и нарезка зубов под разметку идут одной ночью
(около 5 минут на снимок), а утром врачам есть что размечать.

МЕШИ И НАХОДКИ НЕ СТРОЯТСЯ. Для разметки каналов нужны только том и маска
зубов; меш и слой находок добавляют к снимку ещё 7 минут и не нужны никому,
пока врач не открыл кейс в демо. Демо построит их само при первом открытии.

ЧТО ПРИХОДИТ С ФЛЕШКИ — ЗООПАРК, и это главная работа скрипта:
  * структура папок у каждого экспорта своя, имена файлов без расширения
    (`IM000001`), поэтому серии ищутся через GDCM, а не по маске `*.dcm`;
  * в одной папке лежит по нескольку серий, включая обзорные снимки и
    отчётные картинки — берём только объёмные (`MIN_SLICES`, шаг вокселя);
  * один пациент запросто приедет дважды — и флешкой, и телеграмом. Дубли
    ловятся по СОДЕРЖИМОМУ тома: имя папки ничего не гарантирует. На наших
    восьми «клинических» кейсах разных пациентов оказалось четыре.

ПЕРСОНАЛЬНЫЕ ДАННЫЕ. В DICOM лежат ФИО и дата рождения; в кейс они не
попадают (как и в подаче панорам). Соответствие «кейс -> пациент» пишется
ОТДЕЛЬНО, в `research_3d/patient_map.json`: клинике надо уметь найти пациента,
но эти данные не должны ездить вместе с набором для разметки.

СТОРОНЫ. Снимок с аппарата, для которого врач не подтвердил конвенцию сторон,
ввозится, но в разметку НЕ идёт: номера зубов на нём могут быть зеркальны.
Такие кейсы перечисляются в конце отдельным списком — их надо показать врачу
и подтвердить сторону один раз на аппарат.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import cbct_backend as cb  # noqa: E402

MIN_SLICES = 80           # меньше — обзорный снимок или отчёт, не объём
MAX_SPACING_MM = 0.8      # КЛКТ снимает мельче; крупнее — не наш случай
MIN_SIDE = 150            # вокселей по стороне
HASH_FILE = cb.R3D / "volume_hashes.json"
PATIENT_MAP = cb.R3D / "patient_map.json"
MODEL = "sp"


def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}


def save_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def volume_hash(img: sitk.Image) -> str:
    """Отпечаток по прореженным вокселям: разные пациенты расходятся и на
    каждом четвёртом, а полный том хэшировать — лишние секунды на снимок."""
    a = sitk.GetArrayFromImage(img)
    return hashlib.md5(np.ascontiguousarray(a[::4, ::4, ::4]).tobytes()).hexdigest()[:12]


def known_hashes() -> dict:
    """Хэши уже заведённых кейсов; считаем один раз и запоминаем."""
    cache = load_json(HASH_FILE)
    changed = False
    for cid, c in sorted(cb.cases.items()):
        v = c.get("volume")
        if not v or cid in cache or not Path(v).is_file():
            continue
        try:
            cache[cid] = volume_hash(sitk.ReadImage(v))
            changed = True
            print(f"  отпечаток {cid}: {cache[cid]}", flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"  {cid}: не прочитался ({e})")
    if changed:
        save_json(HASH_FILE, cache)
    return {h: cid for cid, h in cache.items()}


def find_series(src: Path, work: Path) -> list[tuple[Path, str, list[str]]]:
    """Все DICOM-серии под `src`. Архивы распаковываются во временную папку."""
    roots = [src]
    for z in sorted(src.rglob("*.zip")):
        out = work / f"zip_{z.stem}"
        if not out.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(z) as f:
                    f.extractall(out)
                print(f"  распакован {z.name}")
            except Exception as e:                      # noqa: BLE001
                print(f"  {z.name}: не распаковался ({e})")
                continue
        roots.append(out)

    found = []
    reader = sitk.ImageSeriesReader()
    seen_dirs = set()
    for root in roots:
        for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
            key = str(d.resolve())
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            try:
                ids = reader.GetGDCMSeriesIDs(str(d))
            except Exception:                           # noqa: BLE001
                continue
            for sid in ids:
                names = reader.GetGDCMSeriesFileNames(str(d), sid)
                if len(names) >= MIN_SLICES:
                    found.append((d, sid, list(names)))
    return found


def import_series(names: list[str], batch: str, n: int) -> dict | None:
    """Серия -> кейс. Возвращает запись кейса или None, если это не КЛКТ."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    img = reader.Execute()
    size = list(img.GetSize())          # (x, y, z)
    sp = list(img.GetSpacing())
    if min(size) < MIN_SIDE or max(sp) > MAX_SPACING_MM:
        print(f"    пропуск: {size} вокселей, шаг {[round(v, 2) for v in sp]} — не КЛКТ")
        return None

    cid = f"batch_{batch}_{n:03d}"
    cdir = cb.CASES_DIR / cid
    cdir.mkdir(parents=True, exist_ok=True)
    vpath = cdir / "volume.mha"
    sitk.WriteImage(img, str(vpath), useCompression=True)
    c = cb._new_case(cid, f"Партия {batch} · {n:03d}", vpath)
    c["shape"] = list(img.GetSize()[::-1])
    c["spacing"] = list(img.GetSpacing()[::-1])
    c["scanner"] = cb.read_scanner(names[0])
    cb._save_meta(c)
    return c


def patient_of(dcm_path: str) -> dict:
    """ФИО и дата рождения — ТОЛЬКО для отдельной таблицы соответствия."""
    try:
        import pydicom
        d = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return {"name": str(getattr(d, "PatientName", "") or ""),
                "id": str(getattr(d, "PatientID", "") or ""),
                "birth": str(getattr(d, "PatientBirthDate", "") or ""),
                "study_date": str(getattr(d, "StudyDate", "") or "")}
    except Exception:                                   # noqa: BLE001
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="папка с выгрузкой (флешка, распакованный телеграм)")
    ap.add_argument("--dry-run", action="store_true", help="только показать, что нашлось")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-segment", action="store_true", help="не запускать сегментацию")
    ap.add_argument("--no-prepare", action="store_true", help="не готовить зубы к разметке")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_dir():
        raise SystemExit(f"нет папки {src}")
    batch = time.strftime("%Y%m%d")
    work = cb.R3D / "batch_tmp" / batch
    work.mkdir(parents=True, exist_ok=True)

    print(f"ищу серии в {src}")
    series = find_series(src, work)
    print(f"серий с {MIN_SLICES}+ срезами: {len(series)}")
    if a.limit:
        series = series[:a.limit]

    print("отпечатки уже заведённых кейсов…")
    seen = known_hashes()
    imported, dupes, skipped = [], [], []
    pmap = load_json(PATIENT_MAP)

    for i, (d, sid, names) in enumerate(series, 1):
        print(f"[{i}/{len(series)}] {d} · {len(names)} срезов", flush=True)
        if a.dry_run:
            continue
        try:
            img = sitk.ReadImage(names[0])              # дешёвая проверка читаемости
            del img
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(names)
            vol = reader.Execute()
        except Exception as e:                          # noqa: BLE001
            skipped.append((str(d), f"не читается: {e}"))
            continue
        h = volume_hash(vol)
        if h in seen:
            print(f"    дубль тома {h}: уже есть как {seen[h]}")
            dupes.append((str(d), seen[h]))
            continue
        c = import_series(names, batch, len(imported) + 1)
        if c is None:
            skipped.append((str(d), "не похоже на КЛКТ"))
            continue
        seen[h] = c["case_id"]
        cache = load_json(HASH_FILE)
        cache[c["case_id"]] = h
        save_json(HASH_FILE, cache)
        # ФИО отдельно от набора: клинике надо найти пациента, разметке — нет
        pmap[c["case_id"]] = {**patient_of(names[0]), "source": str(d),
                              "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        save_json(PATIENT_MAP, pmap)
        imported.append(c)
        sc = c.get("scanner") or {}
        print(f"    -> {c['case_id']}  {c['shape']}  шаг {c['spacing'][0]:.2f} мм  "
              f"{sc.get('manufacturer','?')} {sc.get('model','?')}")

    if a.dry_run:
        print("\nпробный прогон, ничего не ввезено")
        return

    print(f"\nввезено {len(imported)}, дублей {len(dupes)}, пропущено {len(skipped)}")
    for d, why in skipped:
        print(f"  пропуск: {d} — {why}")

    unconfirmed = []
    for c in imported:
        st = cb.scanner_sides_state(c)
        if not st.get("confirmed"):
            unconfirmed.append((c["case_id"], st.get("scanner", "?")))

    if imported and not a.no_segment:
        print("\nсегментация…")
        for i, c in enumerate(imported, 1):
            t0 = time.time()
            try:
                c["masks"][MODEL] = str(cb._run_inference(c, MODEL))
                c["model"] = MODEL
                cb._save_meta(c)
                print(f"  [{i}/{len(imported)}] {c['case_id']} — {time.time() - t0:.0f} с", flush=True)
            except Exception as e:                      # noqa: BLE001
                print(f"  [{i}/{len(imported)}] {c['case_id']} — ошибка: {e}")

    if imported and not a.no_segment and not a.no_prepare:
        ok = [c["case_id"] for c in imported
              if c["masks"].get(MODEL) and cb.scanner_sides_state(c).get("confirmed")]
        if ok:
            print(f"\nподготовка зубов к разметке: {len(ok)} кейсов")
            import cbct_canal_prepare as prep
            net, device = cb.canal_net()
            rows = []
            for cid in ok:
                print(f"-> {cid}", flush=True)
                rows += prep.prepare_case(cid, MODEL, net, device, False)
            all_rows = prep.rows_on_disk()
            queue = prep.build_queue([r for r in all_rows
                                      if r["length_mm"] >= prep.MIN_TOOTH_LEN_MM])
            (prep.OUT / "queue.json").write_text(
                json.dumps(queue, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"в очереди разметки теперь {len(queue)} зубов")

    if unconfirmed:
        print("\nСТОРОНЫ НЕ ПОДТВЕРЖДЕНЫ — эти кейсы в разметку не пущены:")
        for cid, key in unconfirmed:
            print(f"  {cid} · аппарат {key}")
        print("  Откройте кейс в демо и подтвердите сторону — правило запомнится "
              "на аппарат, для всех его снимков сразу.")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
