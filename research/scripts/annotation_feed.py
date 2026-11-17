"""Панорамы из карточек пациентов -> набор для разметки врачами.

Запуск (досыпать снимки, загруженные до появления подачи):
    .venv/Scripts/python.exe research/scripts/annotation_feed.py --backfill

ЗАЧЕМ. Разметку по новым классам (пульпа, эндо, убыль кости) ИИ не делает —
её делают врачи, и упирается она в число снимков. Публичных панорам с такой
разметкой нет, а клиника снимает их каждый день. Если набор для разметки
пополнять руками (выгрузить, загрузить пачкой), он не пополняется вовсе:
в системе за месяц накопилось 13 снимков, и все — одной ручной пачкой.
Поэтому каждая панорама, прошедшая обработку в карточке пациента, сама
попадает в набор `clinic-patients` и уходит на ИИ-преразметку.

ИМЯ ФАЙЛА — номер исследования, без имени пациента. Врачу-разметчику имя не
нужно, а набор потом уходит в обучение и может покидать машину клиники.
Шапка отчёта Dentium с фамилией срезается при преразметке
(`report_crop`).

ДУБЛИ. Один и тот же снимок мог попасть и ручной пачкой, и через карточку
(так уже случилось с Report_20260812_184236). Разметка одного снимка дважды —
потерянное время врача и утечка между обучением и проверкой, если копии
разойдутся по разные стороны разбиения. Сверяем MD5 ИСХОДНОГО файла: в наборе
разметки лежит обрезанная копия, её хэш с исходником не совпадёт, поэтому
реестр исходных хэшей ведётся отдельно (`feed.json`), а ручная пачка
сверяется по оригиналам в `research/example/2d_images`.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # research/
SLUG = "clinic-patients"
DS = ROOT / "datasets/annotation_sets" / SLUG
NAME = "Клиника — снимки из карточек пациентов"
# Оригиналы ручной пачки Dentium (в наборе разметки лежат их обрезанные копии).
MANUAL_ORIGINALS = ROOT / "example/2d_images"
ANNOTATE_URL = "http://127.0.0.1:7862"
KINDS = ("opg", "periapical")
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")

_manual_hashes: set[str] | None = None


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _known_manual() -> set[str]:
    global _manual_hashes
    if _manual_hashes is None:
        _manual_hashes = {_md5(p) for p in MANUAL_ORIGINALS.glob("*")
                          if p.suffix.lower() in IMG_SUFFIXES}
    return _manual_hashes


def known_hashes() -> dict[str, str]:
    """MD5 исходника -> где этот снимок уже лежит, по ВСЕМ наборам разметки.

    Реестры два: `feed.json` у подачи из карточек и `images.json` у наборов,
    куда врачи добавляют панорамы сами. Плюс оригиналы ручной пачки Dentium,
    у которой реестра нет. Один снимок у двух врачей — двойная работа и
    утечка между обучением и проверкой, поэтому проверка общая.
    """
    out = {h: "ручная пачка Dentium" for h in _known_manual()}
    for ds in (ROOT / "datasets/annotation_sets").iterdir():
        for reg_name in ("images.json", "feed.json"):
            f = ds / reg_name
            if not f.is_file():
                continue
            try:
                reg = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for key, r in reg.items():
                if isinstance(r, dict) and r.get("md5") and r.get("file", key):
                    out.setdefault(r["md5"], f"{ds.name}/{r.get('file') or key}")
    return out


def _ensure_dataset() -> dict:
    """Создаёт набор при первом снимке; возвращает реестр подачи."""
    for sub in ("images", "labels", "labels_human"):
        (DS / sub).mkdir(parents=True, exist_ok=True)
    meta = DS / "meta.json"
    if not meta.is_file():
        meta.write_text(json.dumps({
            "name": NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False), encoding="utf-8")
    reg = DS / "feed.json"
    if reg.is_file():
        try:
            return json.loads(reg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _save_registry(reg: dict) -> None:
    (DS / "feed.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def request_analysis() -> bool:
    """Просит сервер разметки преразметить новые снимки.

    Не ждём и не падаем: если сервер разметки лежит, снимок остаётся без
    ИИ-разметки и будет преразмечен при его следующем старте (он сам
    досчитывает снимки без `labels/`).
    """
    try:
        req = urllib.request.Request(
            f"{ANNOTATE_URL}/api/datasets/{SLUG}/analyze", method="POST")
        urllib.request.urlopen(req, timeout=3).read()
        return True
    except OSError:
        return False


def add_study(sid: str, image: Path, loader=None, analyze: bool = True) -> str:
    """Кладёт снимок исследования в набор. Возвращает, что произошло.

    `loader` — функция чтения DICOM в RGB (`demo_server.load_image`): модели
    разметки DICOM не читают, а окно и инверсию надо взять ТЕ ЖЕ, что при
    анализе, иначе врач размечает не ту картинку, что видела модель.
    """
    image = Path(image)
    reg = _ensure_dataset()
    if sid in reg:
        return "уже в наборе"
    digest = _md5(image)
    if digest in known_hashes():
        reg[sid] = {"md5": digest, "file": None, "skipped": "дубль",
                    "added_at": datetime.now().isoformat(timespec="seconds")}
        _save_registry(reg)
        return "дубль — снимок уже размечается"

    suffix = image.suffix.lower()
    if suffix in IMG_SUFFIXES:
        name = f"{sid}{'.jpg' if suffix == '.jpeg' else suffix}"
        (DS / "images" / name).write_bytes(image.read_bytes())
    elif suffix == ".dcm":
        if loader is None:
            raise ValueError("для DICOM нужен loader (demo_server.load_image)")
        name = f"{sid}.png"
        loader(str(image)).save(DS / "images" / name, "PNG")
    else:
        raise ValueError(f"не снимок: {image.name}")

    reg[sid] = {"md5": digest, "file": name,
                "added_at": datetime.now().isoformat(timespec="seconds")}
    _save_registry(reg)
    if analyze:
        request_analysis()
    return f"добавлен как {name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backfill", action="store_true",
                        help="добавить все уже загруженные 2D-исследования")
    args = parser.parse_args()
    if not args.backfill:
        parser.print_help()
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import patients as pat

    loader = None
    added = 0
    for p in pat.list_patients():
        for st in pat.list_studies(p["id"]):
            if st["kind"] not in KINDS or st["status"] != "ready":
                continue
            src = sorted(x for x in (pat.study_dir(st["id"]) / "source").glob("*")
                         if x.suffix.lower() in IMG_SUFFIXES + (".dcm",))
            if not src:
                continue
            if src[0].suffix.lower() == ".dcm" and loader is None:
                # Тяжёлый импорт (веса моделей) — только если DICOM и правда есть.
                import demo_server
                loader = demo_server.load_image
            res = add_study(st["id"], src[0], loader=loader, analyze=False)
            added += res.startswith("добавлен")
            print(f"{st['id']}  {src[0].name}: {res}")
    if added:
        ok = request_analysis()
        print(f"добавлено {added}; преразметка "
              + ("запрошена" if ok else "не запрошена — сервер разметки не отвечает"))
    else:
        print("новых снимков нет")


if __name__ == "__main__":
    main()
