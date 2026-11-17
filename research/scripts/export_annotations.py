"""Разметка врачей -> вырезы зубов с метками для второго этапа 2D.

    .venv\\Scripts\\python.exe research\\scripts\\export_annotations.py
    .venv\\Scripts\\python.exe research\\scripts\\export_annotations.py --sets clinic-patients

Результат — `research/datasets/clinic_tooth_crops/` в формате
`tooth_crops2` (`labels.json` + `train/` и `val/`), поэтому
`train_tooth_cls.py --data tooth_crops2 clinic_tooth_crops` учит на обоих
наборах сразу. Плюс `report.json`: что не удалось сопоставить и где рамки
нарисованы не по правилу — это обратная связь врачам.

ЧТО СЧИТАЕТСЯ ПРИМЕРОМ. Только снимки, сохранённые врачом (`labels_human`).
ИИ-преразметка без проверки примером не является. Каждый зуб проверенного
снимка даёт один вырез, и зуб без отметок — ЗДОРОВЫЙ пример. Поэтому врачам
и сказано сохранять снимок только целиком размеченным.

НО ЗДОРОВ ОН ТОЛЬКО ПО ТЕМ КЛАССАМ, КОТОРЫЕ ВРАЧ ВИДЕЛ. Снимки 13.08
проверены, когда классов было четыре: о пломбах и эндо на них не сказано
ничего. Сервер разметки пишет в сопроводительный файл `reviewed_classes`;
у старых файлов этой отметки нет, и они считаются проверенными по четырём.
Метка непроверенного класса — `null`, а не 0, и обучение её пропускает.

НОМЕР ЗУБА И ВЫРЕЗ — ОТ ДЕТЕКТОРА (`teeth_detect`), как при работе. Вырез
по рамке врача научил бы сеть на другой геометрии, чем та, что придёт от
детектора на приёме. Рамка врача служит только для того, чтобы понять, к
какому зубу относится метка:

    рамка «весь зуб»   перекрытие с зубом IoU >= 0.5 (как в tooth_crops2);
                       если рамка целиком внутри зуба и сильно меньше его —
                       метка засчитывается, но врачу уходит замечание:
                       скорее всего обведён очаг, а не зуб
    пломба, фрагмент,  зуб, в котором лежит большая часть объекта
    имплант
    коронка / мост     каждый зуб, который она перекрывает по ширине:
                       мост лежит на нескольких зубах сразу

Зуб, которого детектор не нашёл, но врач обвёл, всё равно становится
примером — вырез по рамке врача, номер 0. Выбрасывать его нельзя: пропуски
детектора не случайны (восьмёрки, коронки), и именно такие зубы нужны.

РАЗБИЕНИЕ — ПО ПАЦИЕНТУ, если он известен (снимки из карточек), иначе по
снимку. Зубы одного человека похожи, и разнеси их по обе стороны — проверка
завысится. Сторона считается хэшем ключа, а не случайной перетасовкой:
новые снимки не перекладывают старые между обучением и проверкой, и метрики
разных выгрузок сравнимы. Ключ снимка — имя набора и файла, а НЕ хэш
содержимого: 17.09 снимки перерезали, хэш сменился, и один из трёх
проверенных снимков молча переехал из проверки в обучение.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from annotation_classes import CLASSES  # noqa: E402

ROOT = HERE.parents[0]                          # research/
SETS = ROOT / "datasets/annotation_sets"
DST = ROOT / "datasets/clinic_tooth_crops"
FEED_SET = "clinic-patients"                    # имена файлов = номер исследования

VAL_PERCENT = 15
LEGACY_REVIEWED = 4        # проверки до 17.09 шли по четырём классам
PAD = 0.25                 # как в tooth_crops2 и в работе (`teeth_detect.crop`)

IOU_SAME = 0.5             # как в make_tooth_crops2
# «Обведён очаг, а не зуб»: рамка почти целиком внутри зуба и заметно меньше
# его. Засчитываем (зуб определён однозначно), но сообщаем врачу.
INSIDE_FRAC = 0.7
SMALL_FRAC = 0.6
OBJECT_INSIDE = 0.5        # доля объекта внутри зуба
BRIDGE_WIDTH = 0.5         # доля ширины зуба под коронкой или мостом


def _box_px(b: dict, w: int, h: int) -> tuple[float, float, float, float]:
    return ((b["cx"] - b["w"] / 2) * w, (b["cy"] - b["h"] / 2) * h,
            (b["cx"] + b["w"] / 2) * w, (b["cy"] + b["h"] / 2) * h)


def _area(a) -> float:
    return max(a[2] - a[0], 0) * max(a[3] - a[1], 0)


def _inter(a, b) -> float:
    return (max(min(a[2], b[2]) - max(a[0], b[0]), 0)
            * max(min(a[3], b[3]) - max(a[1], b[1]), 0))


def _iou(a, b) -> float:
    i = _inter(a, b)
    return i / (_area(a) + _area(b) - i) if i else 0.0


def read_human(ds: Path, stem: str) -> tuple[list[dict], int, list | None]:
    boxes = []
    for line in (ds / "labels_human" / f"{stem}.txt").read_text(
            encoding="utf-8").splitlines():
        v = line.split()
        if len(v) >= 5:
            boxes.append({"cls": int(float(v[0])), "cx": float(v[1]),
                          "cy": float(v[2]), "w": float(v[3]), "h": float(v[4])})
    reviewed, visible = LEGACY_REVIEWED, None
    side = ds / "labels_human" / f"{stem}.json"
    if side.is_file():
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
            reviewed = int(meta.get("reviewed_classes", LEGACY_REVIEWED))
            # Полоса кадра (доли высоты), которую врач видел при проверке.
            # Ставится, когда снимок перерезали после проверки: до 17.09
            # отчёты резались по корням нижних зубов (см. report_crop).
            visible = meta.get("visible_y")
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return boxes, reviewed, visible


def split_key(set_name: str, img: Path) -> str:
    """Пациент, если известен, иначе содержимое снимка."""
    if set_name == FEED_SET:
        try:
            import patients as pat
            st = pat.get_study(img.stem)
            if st and st.get("patient_id"):
                return "patient:" + st["patient_id"]
        except Exception:  # noqa: BLE001 — без базы пациентов режем по снимку
            pass
    return f"image:{set_name}/{img.stem}"


def split_of(key: str, val_percent: int) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return "val" if h % 100 < val_percent else "train"


def assign(teeth: list[dict], boxes: list[dict], w: int, h: int,
           notes: list[str], band: tuple[float, float] | None = None) -> None:
    """Раскладывает рамки врача по зубам (дописывает `tags` у зубов).

    `band` — полоса по высоте (в пикселях), которую врач видел. Рамки он
    рисовал в её пределах, и рамка нижнего зуба обрывается на краю кадра;
    сравнивать её надо с ВИДИМОЙ частью зуба, иначе перекрытие падает ниже
    порога и зуб получает второй, «врачебный» вырез.
    """
    def vis(t):
        if band is None:
            return t["box"]
        x1, y1, x2, y2 = t["box"]
        return (x1, max(y1, band[0]), x2, min(y2, band[1]))

    for b in boxes:
        if not 0 <= b["cls"] < len(CLASSES):
            notes.append(f"неизвестный класс {b['cls']} — пропущен")
            continue
        c = CLASSES[b["cls"]]
        box = _box_px(b, w, h)
        name = c["name"]

        if c["box"] == "tooth":
            best = max(teeth, key=lambda t: _iou(vis(t), box), default=None)
            if best is not None and _iou(vis(best), box) >= IOU_SAME:
                best["tags"].add(c["key"])
                continue
            inside = max(teeth, key=lambda t: _inter(vis(t), box), default=None)
            if (inside is not None and _area(box)
                    and _inter(vis(inside), box) / _area(box) >= INSIDE_FRAC
                    and _area(box) < SMALL_FRAC * _area(vis(inside))):
                inside["tags"].add(c["key"])
                notes.append(f"«{name}» у зуба {inside['fdi'] or '?'}: рамка "
                             "меньше зуба — обведён очаг, а не весь зуб")
                continue
            # Детектор зуба не нашёл — вырез по рамке врача. Две рамки
            # врача на одном таком зубе сливаются в один пример.
            own = next((t for t in teeth if t["source"] == "doctor"
                        and _iou(t["box"], box) >= IOU_SAME), None)
            if own is None:
                own = {"fdi": 0, "box": box, "source": "doctor", "tags": set()}
                teeth.append(own)
                notes.append(f"«{name}»: зуб не найден детектором — "
                             "вырез по рамке врача, номер неизвестен")
            own["tags"].add(c["key"])
            continue

        # Рамка вокруг объекта.
        if c["key"] == "crown":
            hit = [t for t in teeth if t["source"] == "detector"
                   and _inter(t["box"], box) > 0
                   and (min(t["box"][2], box[2]) - max(t["box"][0], box[0]))
                   >= BRIDGE_WIDTH * (t["box"][2] - t["box"][0])]
        else:
            best = max(teeth, key=lambda t: _inter(t["box"], box), default=None)
            hit = ([best] if best is not None and _area(box)
                   and _inter(best["box"], box) / _area(box) >= OBJECT_INSIDE
                   else [])
        for t in hit:
            t["tags"].add(c["key"])
        if not hit:
            notes.append(f"«{name}»: не лёг ни на один найденный зуб — "
                         "в выгрузку не попал")


def labels_for(tags: set[str], reviewed: int, seen_whole: bool = True) -> dict:
    # Зуб, который врач видел не целиком, «здоровым» не считается: что не
    # отмечено, то неизвестно. Поставленные отметки остаются.
    lab = {c["key"]: (1 if c["key"] in tags
                      else 0 if c["id"] < reviewed and seen_whole else None)
           for c in CLASSES}

    def known(*keys):
        return all(lab[k] is not None for k in keys)

    row = {"labels": lab}
    # Совместимые с tooth_crops2 поля: 0 нет · 1 кариес · 2 глубокий.
    # Кариес до пульпы — тоже «глубокий»: в DENTEX отдельного класса нет.
    if known("caries", "deep_caries"):
        row["caries"] = (2 if lab["deep_caries"] or lab["pulp_exposure"]
                         else 1 if lab["caries"] else 0)
    else:
        row["caries"] = None
    row["apical"] = lab["periapical"]
    # Полная шкала: 3 — кариес дошёл до пульпы.
    if known("caries", "deep_caries", "pulp_exposure"):
        row["caries_level"] = (3 if lab["pulp_exposure"] else row["caries"])
    else:
        row["caries_level"] = None
    bl = ("bone_loss_1", "bone_loss_2", "bone_loss_3")
    row["bone_loss"] = (max((i + 1 for i, k in enumerate(bl) if lab[k]), default=0)
                        if known(*bl) else None)
    return row


def export(sets: list[str] | None, out: Path, val_percent: int,
           conf: float) -> dict:
    import teeth_detect as td

    names = sets or sorted(p.name for p in SETS.iterdir()
                           if (p / "labels_human").is_dir())
    for sub in ("train", "val"):
        if (out / sub).is_dir():
            shutil.rmtree(out / sub)          # выгрузка пересобирается целиком
    out.mkdir(parents=True, exist_ok=True)

    model = td._load()
    rows, report = [], {"images": [], "sets": names}
    for set_name in names:
        ds = SETS / set_name
        if not (ds / "labels_human").is_dir():
            print(f"нет набора: {set_name}")
            continue
        for lab in sorted((ds / "labels_human").glob("*.txt")):
            img = next((p for p in (ds / "images").glob(lab.stem + ".*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
            if img is None:
                report["images"].append({"set": set_name, "image": lab.stem,
                                         "notes": ["снимок не найден"]})
                continue
            boxes, reviewed, visible = read_human(ds, lab.stem)
            dets, _ = td.resolve(td.detect(str(img), model=model, conf=conf))
            with Image.open(img) as im:
                im = im.convert("L")
                w, h = im.size
                teeth = [{"fdi": d["fdi"], "source": "detector", "tags": set(),
                          "box": (d["xywh"][0] - d["xywh"][2] / 2,
                                  d["xywh"][1] - d["xywh"][3] / 2,
                                  d["xywh"][0] + d["xywh"][2] / 2,
                                  d["xywh"][1] + d["xywh"][3] / 2)}
                         for d in dets]
                notes: list[str] = []
                assign(teeth, boxes, w, h, notes,
                       band=(visible[0] * h, visible[1] * h) if visible else None)
                key = split_key(set_name, img)
                split = split_of(key, val_percent)
                if reviewed < len(CLASSES):
                    notes.append(f"проверен по {reviewed} классам из "
                                 f"{len(CLASSES)} — остальные в выгрузке пустые")
                if visible:
                    notes.append("снимок перерезан после проверки: зубы вне "
                                 "видимой врачу полосы без отметок — неизвестны; "
                                 "снимок надо открыть и сохранить заново")
                hidden = 0
                for n, t in enumerate(teeth):
                    seen = (not visible
                            or (t["box"][1] >= visible[0] * h - 2
                                and t["box"][3] <= visible[1] * h + 2))
                    hidden += not seen
                    row = labels_for(t["tags"], reviewed, seen)
                    if t["tags"] >= {"caries", "deep_caries"}:
                        notes.append(f"зуб {t['fdi'] or '?'}: кариес и глубокий "
                                     "кариес одновременно — взята большая степень")
                    if sum(k in t["tags"] for k in
                           ("bone_loss_1", "bone_loss_2", "bone_loss_3")) > 1:
                        notes.append(f"зуб {t['fdi'] or '?'}: несколько степеней "
                                     "убыли кости — взята большая")
                    x1, y1, x2, y2 = t["box"]
                    crop = td.crop(im, {"xywh": [(x1 + x2) / 2, (y1 + y2) / 2,
                                                 x2 - x1, y2 - y1]}, pad=PAD)
                    tag = t["fdi"] if t["fdi"] else f"d{n}"
                    fname = f"{set_name}__{img.stem}__{tag}.png"
                    (out / split).mkdir(parents=True, exist_ok=True)
                    crop.save(out / split / fname)
                    rows.append({"file": f"{split}/{fname}", "split": split,
                                 "fdi": t["fdi"], "set": set_name,
                                 "image": img.name, "source": t["source"],
                                 "reviewed_classes": reviewed,
                                 "seen_whole": seen, **row})
            report["images"].append({
                "set": set_name, "image": img.name, "split": split,
                "teeth": len(teeth), "doctor_boxes": len(boxes),
                "reviewed_classes": reviewed, "partly_hidden": hidden,
                "notes": notes})
            print(f"  {set_name}/{img.name}: зубов {len(teeth)}, рамок врача "
                  f"{len(boxes)}, {split}" + (f", замечаний {len(notes)}"
                                             if notes else ""), flush=True)

    # Сводка по классам: сколько зубов с меткой и сколько с известным
    # значением — второе показывает, сколько у класса вообще есть примеров.
    summary = {}
    for c in CLASSES:
        s = {}
        for split in ("train", "val"):
            vals = [r["labels"][c["key"]] for r in rows if r["split"] == split]
            s[split] = {"positive": sum(v == 1 for v in vals),
                        "known": sum(v is not None for v in vals)}
        summary[c["key"]] = s
    report["summary"] = summary
    report["crops"] = Counter(r["split"] for r in rows)
    (out / "labels.json").write_text(json.dumps(rows, ensure_ascii=False),
                                     encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                indent=1), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="",
                    help="наборы через запятую (по умолчанию все)")
    ap.add_argument("--out", default=str(DST))
    ap.add_argument("--val-percent", type=int, default=VAL_PERCENT)
    ap.add_argument("--conf", type=float, default=0.30,
                    help="порог детектора зубов, как в make_tooth_crops2")
    args = ap.parse_args()

    sets = [s.strip() for s in args.sets.split(",") if s.strip()] or None
    rep = export(sets, Path(args.out), args.val_percent, args.conf)

    n_img = sum(1 for r in rep["images"] if "teeth" in r)
    print(f"\nпроверенных снимков: {n_img}; вырезов: "
          f"обучение {rep['crops'].get('train', 0)}, "
          f"проверка {rep['crops'].get('val', 0)}")
    print(f"{'класс':52s} {'обучение':>14s} {'проверка':>14s}")
    for c in CLASSES:
        s = rep["summary"][c["key"]]
        if not (s["train"]["known"] or s["val"]["known"]):
            continue
        print(f"  {c['name'][:50]:50s} "
              f"{s['train']['positive']:>5d} из {s['train']['known']:<5d} "
              f"{s['val']['positive']:>5d} из {s['val']['known']:<5d}")
    never = [c["name"] for c in CLASSES
             if not (rep["summary"][c["key"]]["train"]["known"]
                     or rep["summary"][c["key"]]["val"]["known"])]
    if never:
        print("не проверен ни на одном снимке: " + "; ".join(never))
    notes = defaultdict(int)
    for r in rep["images"]:
        for n in r.get("notes", []):
            # Номера зубов убираем: группируется вид замечания, а не зуб.
            notes[re.sub(r"(зуба?) (\d+|\?)", r"\1 N", n)] += 1
    if notes:
        print("\nзамечания (подробно — report.json):")
        for n, k in sorted(notes.items(), key=lambda x: -x[1]):
            print(f"  {k:3d} × {n}")
    print(f"\nметки: {Path(args.out) / 'labels.json'}")


if __name__ == "__main__":
    main()
