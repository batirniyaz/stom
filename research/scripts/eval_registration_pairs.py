"""Настоящая точность совмещения скана с КЛКТ — на парах ОДНОГО пациента.

Зачем. В проекте записано: «пары "скан и КЛКТ одного пациента" у нас НЕТ,
поэтому настоящая точность не измерена и измерить её нечем». Проверка шла
самотестом на зашумлённой копии, а копия сама с собой сходится в ноль — то
есть проверялась математика, а не метод.

Датасет `3D multimodal dental dataset based on CBCT and oral scan` (figshare
26965903, CC BY 4.0) даёт 289 пар. Этого достаточно, чтобы измерить две вещи,
и вторая важнее первой:

  * ТОЧНОСТЬ на своих парах — сколько миллиметров расхождения по зубам;
  * РАЗДЕЛЕНИЕ своих и чужих — если чужого пациента метод «принимает», то
    любая метрика на своих парах бессмысленна. Замер на client4 + batir уже
    показывал, что общий остаток ICP тут обманывает: 1.02 мм на заведомо
    разных людях, потому что ICP отбрасывает 20% худших пар и промахи прячутся
    именно в них. Судим по МЕДИАНЕ ПО ЗУБАМ.

Кейсы кладутся рядом с обычными (web_cases), поэтому результат можно открыть
во вьюере и посмотреть глазами.

Запуск:
    .venv/Scripts/python.exe research/scripts/eval_registration_pairs.py --limit 3
    .venv/Scripts/python.exe research/scripts/eval_registration_pairs.py --limit 10 --skip-build
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402
import ios_cbct_register as reg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "research_3d/datasets/paired_cbct_ios"
IOS_CASES = ROOT / "research_ios/web_cases"
PY = ROOT / ".venv/Scripts/python.exe"
BUILD = ROOT / "research/scripts/ios_build_mesh.py"


def patient_dirs(limit: int) -> list[Path]:
    ds = sorted(p for p in PAIRS.iterdir()
                if p.is_dir() and p.name.isdigit())
    return ds[:limit]


def case_ids(pid: str) -> tuple[str, str]:
    return f"pair{pid}_cbct", f"pair{pid}_ios"


def build_cbct(pid: str, model: str) -> bool:
    """Завести КЛКТ как обычный кейс и посчитать маску."""
    cid, _ = case_ids(pid)
    vol = PAIRS / pid / f"{pid}_cbct" / f"{pid}_cbct.nii.gz"
    if not vol.is_file():
        print(f"  {pid}: нет тома {vol.name}")
        return False
    c = cb.get_case(cid)
    if c is None:
        c = cb._new_case(cid, f"Пара {pid}: КЛКТ", vol, {}, example=False)
        r = sitk.ImageFileReader()
        r.SetFileName(str(vol))
        r.ReadImageInformation()
        c["shape"] = list(r.GetSize()[::-1])
        c["spacing"] = list(r.GetSpacing()[::-1])
    if c["masks"].get(model):
        return True
    # Каталог кейса обычно создаёт загрузка через веб; здесь кейс заводится
    # напрямую, и без mkdir запись маски падает на «No such file or directory».
    (cb.CASES_DIR / cid).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        path = cb._run_inference(c, model)
    except Exception as exc:  # noqa: BLE001
        print(f"  {pid}: КЛКТ не сегментирован — {exc}")
        return False
    c["masks"][model] = str(path)
    c["model"] = model
    cb._save_meta(c)
    print(f"  {pid}: КЛКТ готов за {time.time()-t0:.0f} с")
    return True


def build_ios(pid: str) -> bool:
    """Собрать scan.glb из пары STL теми же скриптами, что и клинический кейс."""
    _, iid = case_ids(pid)
    out = IOS_CASES / iid / "scan.glb"
    if out.is_file():
        return True
    up = PAIRS / pid / f"{pid}_ios" / f"{pid}_UpperJawScan.stl"
    lo = PAIRS / pid / f"{pid}_ios" / f"{pid}_LowerJawScan.stl"
    if not up.is_file() or not lo.is_file():
        print(f"  {pid}: нет STL челюстей")
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run(
        [str(PY), str(BUILD), "--upper", str(up), "--lower", str(lo),
         "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT))
    if r.returncode != 0 or not out.is_file():
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        print(f"  {pid}: скан не собран — {' | '.join(tail)}")
        return False
    print(f"  {pid}: скан готов за {time.time()-t0:.0f} с")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--model", default="sp")
    ap.add_argument("--skip-build", action="store_true",
                    help="только совмещение по уже готовым кейсам")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    ds = patient_dirs(a.limit)
    print(f"пар к обработке: {len(ds)}\n")

    ready = []
    for d in ds:
        pid = d.name
        if a.skip_build:
            cid, iid = case_ids(pid)
            if cb.get_case(cid) and (IOS_CASES / iid / "scan.glb").is_file():
                ready.append(pid)
            continue
        print(f"--- пациент {pid} ---")
        if build_cbct(pid, a.model) and build_ios(pid):
            ready.append(pid)
    if len(ready) < 2:
        raise SystemExit(f"готовых пар {len(ready)}, для контроля нужно ≥2")

    print(f"\nготовых пар: {len(ready)} — {', '.join(ready)}")

    own, cross = [], []
    print("\n=== СВОИ пары (скан и КЛКТ одного пациента) ===")
    for pid in ready:
        cid, iid = case_ids(pid)
        try:
            r = reg.register(cid, iid, a.model)
        except SystemExit as exc:
            print(f"  {pid}: {exc}")
            continue
        if not r.get("ok"):
            print(f"  {pid}: {r.get('error')}")
            continue
        r["pair"] = pid
        own.append(r)
        print(f"  {pid}: общих зубов {len(r['common_teeth']):>2}, "
              f"RMS {r['rms_mm']:>5.2f} мм, по зубам {r['tooth_median_mm']:>6.2f} мм"
              + ("   принято" if r["good"] else ""))

    print("\n=== ЧУЖИЕ пары (контроль: скан от другого пациента) ===")
    for i, pid in enumerate(ready):
        other = ready[(i + 1) % len(ready)]
        if other == pid:
            continue
        cid, _ = case_ids(pid)
        _, iid = case_ids(other)
        try:
            r = reg.register(cid, iid, a.model)
        except SystemExit as exc:
            print(f"  {pid}+{other}: {exc}")
            continue
        if not r.get("ok"):
            print(f"  {pid}+{other}: {r.get('error')}")
            continue
        r["pair"] = f"{pid}+{other}"
        cross.append(r)
        print(f"  КЛКТ {pid} + скан {other}: RMS {r['rms_mm']:>5.2f} мм, "
              f"по зубам {r['tooth_median_mm']:>6.2f} мм"
              + ("   ПРИНЯТО ОШИБОЧНО" if r["good"] else ""))

    print(f"\n{'='*64}")
    print("ИТОГ")
    print(f"{'='*64}")
    for name, rows in (("свои", own), ("чужие", cross)):
        if not rows:
            continue
        tm = np.array([r["tooth_median_mm"] for r in rows])
        rm = np.array([r["rms_mm"] for r in rows])
        print(f"  {name:<7} n={len(rows):>2}  по зубам: медиана {np.median(tm):>7.2f} мм, "
              f"разброс {tm.min():.2f}..{tm.max():.2f}   общий RMS: медиана {np.median(rm):.2f}")
    if own and cross:
        a_own = np.array([r["tooth_median_mm"] for r in own])
        a_cross = np.array([r["tooth_median_mm"] for r in cross])
        gap = a_cross.min() - a_own.max()
        print(f"\n  ЗАЗОР между своими и чужими по зубам: {gap:+.2f} мм "
              + ("— группы не перекрываются" if gap > 0
                 else "— ГРУППЫ ПЕРЕКРЫВАЮТСЯ, порог так не поставить"))
        bad = sum(1 for r in cross if r["good"])
        print(f"  чужих пар, ошибочно принятых текущим правилом: {bad} из {len(cross)}")

    if a.json:
        a.json.write_text(json.dumps({"own": own, "cross": cross},
                                     ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"\n  подробности: {a.json}")


if __name__ == "__main__":
    main()
