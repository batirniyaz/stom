"""Скачивание Teeth3DS+ (интраоральные сканы) с OSF.

Данные лежат не в корне проекта osf.io/xctdy — там только файлы разбиения
train/test и лицензия, все папки по 0 байт. Сами сканы разложены по девяти
дочерним компонентам, в каждом ровно один zip. Итого ~8.9 ГБ.

Запуск:
    .venv/Scripts/python.exe research/scripts/download_teeth3ds.py
    .venv/Scripts/python.exe research/scripts/download_teeth3ds.py --check

Докачка: повторный запуск пропускает файлы, у которых размер уже совпал
с заявленным на OSF, и дотягивает оборванные (HTTP Range).
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "research_ios" / "datasets" / "teeth3ds"
API = "https://api.osf.io/v2/nodes/{}/files/osfstorage/"

COMPONENTS = [
    ("5cmg3", "data_part_1"), ("xfcn9", "data_part_2"), ("hw7bj", "data_part_3"),
    ("n9bd7", "data_part_4"), ("7az58", "data_part_5"), ("2ybr4", "data_part_6"),
    ("gvjah", "data_part_7"), ("um96h", "3DTeethLand_landmarks"),
    ("e53qr", "Teeth3DS_sample"),
]
UA = {"User-Agent": "stomadent-research/1.0"}


def api_files(node_id: str) -> list[tuple[str, int, str]]:
    """-> [(имя, размер, ссылка на скачивание)]"""
    req = urllib.request.Request(API.format(node_id), headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    out = []
    for it in d["data"]:
        a = it["attributes"]
        if a["kind"] == "file":
            out.append((a["name"], a.get("size") or 0, it["links"]["download"]))
    return out


def fetch(url: str, dst: Path, size: int) -> bool:
    have = dst.stat().st_size if dst.exists() else 0
    if size and have == size:
        print(f"   уже скачан: {dst.name}")
        return True
    headers = dict(UA)
    mode = "wb"
    if have and size and have < size:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"   докачка с {have/1e6:.0f} МБ")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, mode) as f:
            done = have
            step = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                step += len(chunk)
                if step >= (1 << 28):           # отчёт каждые ~256 МБ
                    step = 0
                    pct = f"{done/size*100:5.1f}%" if size else ""
                    print(f"   {dst.name}: {done/1e9:.2f} ГБ {pct}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"   ОШИБКА {dst.name}: {exc}")
        return False
    got = dst.stat().st_size
    ok = (not size) or got == size
    print(f"   {'готово' if ok else 'РАЗМЕР НЕ СОВПАЛ'}: {dst.name} "
          f"{got/1e9:.2f} ГБ" + ("" if ok else f" (ждали {size/1e9:.2f})"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="только показать, что есть и чего не хватает")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    total = done_bytes = 0
    missing = []
    for node_id, title in COMPONENTS:
        try:
            files = api_files(node_id)
        except Exception as exc:  # noqa: BLE001
            print(f"{title}: не удалось получить список ({exc})")
            continue
        for name, size, url in files:
            dst = OUT / name
            total += size
            have = dst.stat().st_size if dst.exists() else 0
            done_bytes += min(have, size) if size else have
            if size and have == size:
                continue
            missing.append((title, name, size, url, dst))

    print(f"всего в датасете: {total/1e9:.2f} ГБ, уже на диске: "
          f"{done_bytes/1e9:.2f} ГБ, осталось файлов: {len(missing)}")
    if args.check:
        for title, name, size, _, _ in missing:
            print(f"   нет: {name} ({size/1e9:.2f} ГБ) из {title}")
        return

    bad = []
    for title, name, size, url, dst in missing:
        print(f"\n{title} -> {name} ({size/1e9:.2f} ГБ)")
        if not fetch(url, dst, size):
            bad.append(name)
    print(f"\nкаталог: {OUT}")
    if bad:
        print(f"НЕ ДОКАЧАНЫ: {bad} — запустите скрипт ещё раз, он продолжит")
        sys.exit(1)
    print("все части на месте")


if __name__ == "__main__":
    main()
