"""Stomadent — менеджер датасетов и проверка AI-разметки врачом (FastAPI + фронт).

Запуск:
    .venv/Scripts/python.exe research/scripts/annotate_server.py --port 7862

Открыть: http://localhost:7862

Датасеты живут в research/datasets/annotation_sets/<slug>/:
    images/        — снимки; добавленные врачом называются
                     ГГГГММДД-ЧЧММСС-<6 знаков md5>.jpg — по имени видно,
                     когда снимок пришёл, и список сортируется по времени
    images.json    — реестр добавленных: время, исходное имя файла, md5
    labels/        — AI-преразметка, 6 полей: cls cx cy w h score
    labels_human/  — разметка врача, стандартный YOLO: cls cx cy w h;
                     рядом <stem>.json: по скольким классам проверен,
                     когда проверен впервые и в последний раз
    trash/         — удалённые снимки вместе с разметкой (не насовсем)
    meta.json      — {"name": ..., "created_at": ...}

Работа врача: свой набор (обычно по фамилии), каждый день в него
добавляются панорамы и размечаются — норма `DAILY_GOAL` в день.

Эндпоинты:
    GET    /api/classes                         — справочник классов
    GET    /api/progress                        — сколько размечено врачами по классам
    GET    /api/datasets                        — список датасетов со статистикой
    POST   /api/datasets                        — {"name"} -> пустой набор {slug}
    POST   /api/datasets/upload                 — multipart: name + files[] -> {slug}
    POST   /api/datasets/{slug}/images          — multipart files[] -> добавить в набор
    DELETE /api/datasets/{slug}/images/{name}   — снимок с разметкой в корзину набора
    POST   /api/datasets/{slug}/analyze         — фоновая AI-преразметка (?force=1 — все)
    GET    /api/datasets/{slug}/analyze_status  — {running, done, total, last_error}
    DELETE /api/datasets/{slug}                 — набор целиком в корзину
    GET    /api/datasets/{slug}/images          — [{name, has_ai, has_human}]
    GET    /api/datasets/{slug}/image/{name}    — файл снимка
    GET    /api/datasets/{slug}/labels/{name}   — {"ai": [...], "human": [...] | null}
    POST   /api/datasets/{slug}/labels/{name}   — {"boxes": [...]} -> labels_human
    DELETE /api/datasets/{slug}/labels/{name}   — удалить разметку врача
"""

import argparse
import json
import re
import shutil
import threading
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO

import annotation_feed
from annotation_classes import CLASSES, ENSEMBLE_MIN_CONF, NUM_CLASSES
from report_crop import crop_report

ROOT = Path(__file__).resolve().parents[1]  # research/
DATASETS_DIR = ROOT / "datasets/annotation_sets"
STATIC_DIR = Path(__file__).resolve().parent / "annotate_static"

IMG_EXTS = (".jpg", ".jpeg", ".png")
# Сколько панорам врач размечает в день — решение клиники (17.09).
DAILY_GOAL = 2
# Удалённые наборы целиком: при ежедневной работе случайное удаление набора
# стёрло бы недели разметки врача.
TRASH_DIR = ROOT / "datasets/annotation_trash"
# Справочник классов и правила рисования рамок — в `annotation_classes.py`.
# Сколько классов предсказывает ансамбль: WBF и его пороги относятся только к
# ним. Копии `wbf` в demo_server и eval_ensemble перебирают те же четыре.
ENSEMBLE_CLASSES = 4

# --- ансамбль: параметры как в demo_server.py ---
ENSEMBLE_CONF = 0.05   # низкий порог инференса, фильтрация после WBF
FINAL_CONF = 0.25
WBF_IOU = 0.55
MODEL_WEIGHTS = [
    ROOT / "output/runs/yolo12x_imgsz960/weights/best.pt",
    ROOT / "output/runs/yolo26x_tuned/weights/best.pt",
]

# Модель на 14 состояний (Zenodo 15487430, CC BY 4.0) предразмечает то, чего
# ансамбль не знает. Берутся только классы, где она сильна (mAP50 на её
# валидации): END 0.80, IMP 0.69, ROT 0.64, PRR 0.53, OBT 0.47.
# OBT в этом наборе — ПЛОМБА НА КОРОНКЕ (румынское obturatie), а не канал:
# проверено геометрией рамок, см. opg_teeth_findings.py.
STATES_WEIGHTS = ROOT / "output/runs/panoramic14/weights/best.pt"
STATES_NAMES = ["IMP", "PRR", "OBT", "END", "CAR", "BON", "IMT",
                "API", "ROT", "FUR", "APS", "ROR", "ORD", "SRD"]
STATES_TO_CLASS = {"END": 5, "ROT": 10, "IMP": 16, "PRR": 17, "OBT": 18}
STATES_CONF = 0.25

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Stomadent — Разметка данных", docs_url=None, redoc_url=None)


@app.middleware("http")
async def no_store_api(request, call_next):
    """API отдаётся через Cloudflare Tunnel — CF кэширует GET по расширению
    (.jpg в /api/.../labels/<name>.jpg). no-store от origin CF уважает."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response

# модели грузим один раз при старте
print("Загрузка моделей ансамбля…")
models = [YOLO(str(w)) for w in MODEL_WEIGHTS]
states_model = YOLO(str(STATES_WEIGHTS)) if STATES_WEIGHTS.is_file() else None
print("Модели загружены." if states_model is not None else
      "Модели загружены; модели состояний нет — предразметка только 4 классов.")

# статусы фонового анализа: slug -> {"running", "done", "total", "last_error"}
analyze_status: dict[str, dict] = {}
analyze_lock = threading.Lock()  # инференс строго последовательный (GPU)


# ================= инференс (логика demo_server.py / preannotate_clinic.py) =================

def predict_raw(model: YOLO, img: np.ndarray, conf: float) -> list:
    """[x1, y1, x2, y2, score, cls] в абсолютных координатах."""
    res = model.predict(img, imgsz=960, conf=conf, max_det=100, verbose=False)[0]
    out = []
    for xyxy, cf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy(),
                             res.boxes.cls.cpu().numpy().astype(int)):
        out.append([*map(float, xyxy), float(cf), int(cls)])
    return out


def wbf(dets: list, n_models: int, iou_thr: float = WBF_IOU) -> list:
    """Weighted Boxes Fusion отдельно по каждому классу."""
    fused = []
    for cls in range(ENSEMBLE_CLASSES):
        d = sorted([x for x in dets if x[5] == cls], key=lambda x: -x[4])
        clusters = []
        for det in d:
            placed = False
            for cl in clusters:
                rep = cl[0]
                xx1, yy1 = max(det[0], rep[0]), max(det[1], rep[1])
                xx2, yy2 = min(det[2], rep[2]), min(det[3], rep[3])
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                a = (det[2] - det[0]) * (det[3] - det[1])
                b = (rep[2] - rep[0]) * (rep[3] - rep[1])
                if inter / (a + b - inter + 1e-9) >= iou_thr:
                    cl.append(det)
                    placed = True
                    break
            if not placed:
                clusters.append([det])
        for cl in clusters:
            wsum = sum(x[4] for x in cl)
            x1 = sum(x[0] * x[4] for x in cl) / wsum
            y1 = sum(x[1] * x[4] for x in cl) / wsum
            x2 = sum(x[2] * x[4] for x in cl) / wsum
            y2 = sum(x[3] * x[4] for x in cl) / wsum
            score = min(1.0, (wsum / len(cl)) * (len(cl) / n_models + 0.5))
            fused.append([x1, y1, x2, y2, score, cls])
    return fused


def analyze_image(img: np.ndarray, do_crop: bool = True) -> tuple[np.ndarray, list]:
    """Кроп -> ансамбль -> WBF -> фильтр conf.
    Возвращает (кропнутый снимок, [x1,y1,x2,y2,score,cls]).

    `do_crop=False` — снимок уже обрезан при первом анализе (см.
    `_analyze_worker`)."""
    crop = crop_report(img) if do_crop else img
    all_dets = []
    for m in models:
        all_dets += predict_raw(m, crop, ENSEMBLE_CONF)
    out = [d for d in wbf(all_dets, len(models))
           if d[4] >= max(FINAL_CONF, ENSEMBLE_MIN_CONF.get(d[5], 0.0))]
    # Состояния идут отдельно и без WBF: модель одна, сливать не с чем.
    if states_model is not None:
        for x1, y1, x2, y2, sc, c in predict_raw(states_model, crop, STATES_CONF):
            name = STATES_NAMES[c] if c < len(STATES_NAMES) else ""
            if name in STATES_TO_CLASS:
                out.append([x1, y1, x2, y2, sc, STATES_TO_CLASS[name]])
    return crop, out


def _analyze_worker(slug: str, force: bool) -> None:
    ds = DATASETS_DIR / slug
    st = analyze_status[slug]
    with analyze_lock:  # один инференс за раз
        try:
            files = sorted(p for p in (ds / "images").iterdir()
                           if p.suffix.lower() in IMG_EXTS)
            if not force:
                files = [f for f in files
                         if not (ds / "labels" / f"{f.stem}.txt").is_file()]
            st["total"] = len(files)
            for f in files:
                if analyze_status.get(slug) is not st:  # статус сброшен (re-run)
                    break
                img = cv2.imread(str(f))
                if img is None:
                    st["last_error"] = f"Не удалось прочитать {f.name}"
                    st["done"] += 1
                    continue
                # Обрезаем ТОЛЬКО при первом анализе. Обрезка не идемпотентна:
                # у прежнего правила на уже обрезанном снимке иногда снова
                # находилась «правдоподобная» рамка и срезалась ещё полоса.
                # Так 17.09 при повторной преразметке Report_20260812_184613
                # потерял треть высоты, а рамки врача, сохранённые в долях
                # старого кадра, на таком снимке съехали бы молча.
                # Признак первого анализа — отсутствие ИИ-разметки.
                first = not (ds / "labels" / f"{f.stem}.txt").is_file()
                crop, dets = analyze_image(img, do_crop=first)
                if crop is not img:
                    cv2.imwrite(str(f), crop)
                H, W = crop.shape[:2]
                lines = []
                for bx1, by1, bx2, by2, score, cls in dets:
                    cx, cy = (bx1 + bx2) / 2 / W, (by1 + by2) / 2 / H
                    bw, bh = (bx2 - bx1) / W, (by2 - by1) / H
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.3f}")
                (ds / "labels" / f"{f.stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                st["done"] += 1
        except Exception as exc:  # noqa: BLE001 — показываем причину на фронте
            st["last_error"] = str(exc)
        finally:
            st["running"] = False


# ================= утилиты датасетов =================

TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify(name: str) -> str:
    s = "".join(TRANSLIT.get(c, c) for c in name.lower())
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "dataset"


def unique_slug(name: str) -> str:
    base = slugify(name)[:60]
    slug = base
    n = 2
    while (DATASETS_DIR / slug).exists():
        slug = f"{base}-{n}"
        n += 1
    return slug


def dataset_dir(slug: str) -> Path | None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", slug):
        return None
    p = (DATASETS_DIR / slug).resolve()
    if p.parent != DATASETS_DIR.resolve() or not p.is_dir():
        return None
    return p


def image_files(ds: Path) -> list[Path]:
    return sorted(p for p in (ds / "images").iterdir()
                  if p.suffix.lower() in IMG_EXTS)


def find_image(ds: Path, name: str) -> Path | None:
    if not re.fullmatch(r"[\w.\-]+", name):
        return None
    p = (ds / "images" / name).resolve()
    if p.parent != (ds / "images").resolve() or not p.is_file():
        return None
    return p


def read_labels(path: Path, with_score: bool) -> list[dict] | None:
    """Читает YOLO-файл; 6-е поле (score) опционально. None, если файла нет."""
    if not path.is_file():
        return None
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        box = {
            "cls": int(float(parts[0])),
            "cx": float(parts[1]),
            "cy": float(parts[2]),
            "w": float(parts[3]),
            "h": float(parts[4]),
        }
        if with_score:
            box["score"] = float(parts[5]) if len(parts) >= 6 else 1.0
        boxes.append(box)
    return boxes


def read_human_labels(ds: Path, stem: str) -> list[dict] | None:
    """Human-разметка + авторство боксов из сайдкар-файла <stem>.json
    ({"origins": ["ai"|"human", ...], "scores": [float|null, ...]}).
    Без сайдкара (или при рассинхроне длин) — все боксы "human"."""
    boxes = read_labels(ds / "labels_human" / f"{stem}.txt", with_score=False)
    if boxes is None:
        return None
    origins = scores = None
    try:
        side = json.loads(
            (ds / "labels_human" / f"{stem}.json").read_text(encoding="utf-8"))
        if isinstance(side.get("origins"), list) \
                and len(side["origins"]) == len(boxes):
            origins = side["origins"]
        if isinstance(side.get("scores"), list) \
                and len(side["scores"]) == len(boxes):
            scores = side["scores"]
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    for i, b in enumerate(boxes):
        o = origins[i] if origins and origins[i] in ("ai", "human") else "human"
        b["origin"] = o
        # % уверенности ИИ показываем только у нетронутых врачом ИИ-боксов
        if o == "ai" and scores and isinstance(scores[i], (int, float)):
            b["score"] = float(scores[i])
    return boxes


def read_meta(ds: Path) -> dict:
    try:
        return json.loads((ds / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"name": ds.name, "created_at": ""}


# ================= API: датасеты =================

@app.get("/api/classes")
def api_classes():
    """Справочник классов разметки — единственный источник для фронта."""
    return {"classes": CLASSES}


# Ориентир «сколько размечать» — из нашего же замера на DENTEX, а не с потолка.
# Голова классификатора выреза на 126 обучающих зубах с очагом дала F1 0.31,
# на 460 с глубоким кариесом — 0.60. Отсюда цель 500 зубов на класс; первую
# пробную модель есть смысл учить от 150.
TARGET_PER_CLASS = 500
TRIAL_PER_CLASS = 150


@app.get("/api/progress")
def api_progress():
    """Размеченное врачами по всем наборам: рамок и снимков на класс.

    Считается только `labels_human` — ИИ-преразметка обучающим примером не
    является, пока врач её не подтвердил. Снимок, открытый и сохранённый
    врачом без правок, подтверждает всё, что на нём осталось.
    """
    boxes = [0] * NUM_CLASSES
    images = [0] * NUM_CLASSES
    reviewed = total = 0
    per_ds = []
    for ds in sorted(DATASETS_DIR.iterdir()):
        if not ds.is_dir() or not (ds / "images").is_dir():
            continue
        n_img = len(image_files(ds))
        n_rev = 0
        for f in sorted((ds / "labels_human").glob("*.txt"))                 if (ds / "labels_human").is_dir() else []:
            n_rev += 1
            seen = set()
            for b in read_labels(f, with_score=False) or []:
                if 0 <= b["cls"] < NUM_CLASSES:
                    boxes[b["cls"]] += 1
                    seen.add(b["cls"])
            for c in seen:
                images[c] += 1
        reviewed += n_rev
        total += n_img
        per_ds.append({"slug": ds.name, "name": read_meta(ds).get("name", ds.name),
                       "images": n_img, "reviewed": n_rev})
    return {
        "classes": [{**c, "boxes": boxes[c["id"]], "images": images[c["id"]]}
                    for c in CLASSES],
        "reviewed": reviewed, "total": total, "datasets": per_ds,
        "target": TARGET_PER_CLASS, "trial": TRIAL_PER_CLASS,
    }


def _analyze_pending() -> None:
    """Преразметить снимки, пришедшие, пока сервер лежал.

    Подача из карточек пациентов (`annotation_feed`) не ждёт этот сервер: если
    он не отвечал, снимок лежит в наборе без `labels/`. Без досчёта врач
    открыл бы такой снимок пустым и размечал бы с нуля то, что ИИ делает сам.
    """
    for ds in sorted(DATASETS_DIR.iterdir()):
        if not (ds / "images").is_dir():
            continue
        pending = [f for f in image_files(ds)
                   if not (ds / "labels" / f"{f.stem}.txt").is_file()]
        if pending:
            analyze(ds.name, force=False)


def read_registry(ds: Path) -> dict:
    try:
        return json.loads((ds / "images.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_registry(ds: Path, reg: dict) -> None:
    (ds / "images.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def review_info(ds: Path, stem: str) -> dict:
    try:
        return json.loads((ds / "labels_human" / f"{stem}.json")
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _day(iso: str | None) -> str:
    return (iso or "")[:10]


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    out = []
    today = datetime.now().date().isoformat()
    for ds in sorted(DATASETS_DIR.iterdir()):
        if not ds.is_dir() or not (ds / "images").is_dir():
            continue
        imgs = image_files(ds)
        n_ai = len(list((ds / "labels").glob("*.txt"))) if (ds / "labels").is_dir() else 0
        n_hu = len(list((ds / "labels_human").glob("*.txt"))) \
            if (ds / "labels_human").is_dir() else 0
        size = sum(p.stat().st_size for p in imgs)
        meta = read_meta(ds)
        reg = read_registry(ds)
        reviewed_days = [_day(review_info(ds, p.stem).get("first_reviewed_at"))
                         for p in imgs
                         if (ds / "labels_human" / f"{p.stem}.txt").is_file()]
        added_days = [_day(r.get("added_at")) for r in reg.values()]
        activity = [d for d in reviewed_days + added_days if d]
        out.append({
            "slug": ds.name,
            "name": meta.get("name", ds.name),
            "created_at": meta.get("created_at", ""),
            "images_count": len(imgs),
            "ai_count": n_ai,
            "human_count": n_hu,
            "size_mb": round(size / 1024 / 1024, 1),
            "today_added": added_days.count(today),
            "today_reviewed": reviewed_days.count(today),
            "daily_goal": DAILY_GOAL,
            "last_activity": max(activity) if activity else "",
        })
    out.sort(key=lambda d: d["created_at"], reverse=True)
    return out


class NewDataset(BaseModel):
    name: str


def _create_dataset(name: str) -> Path:
    slug = unique_slug(name)
    ds = DATASETS_DIR / slug
    for sub in ("images", "labels", "labels_human"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    (ds / "meta.json").write_text(json.dumps({
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False), encoding="utf-8")
    return ds


@app.post("/api/datasets")
def create_dataset(req: NewDataset):
    """Пустой набор: врач заводит его один раз и потом добавляет снимки."""
    name = req.name.strip()
    if not name:
        return JSONResponse({"error": "Укажите название набора"}, status_code=400)
    if len(name) > 120:
        return JSONResponse({"error": "Название слишком длинное"}, status_code=400)
    return {"slug": _create_dataset(name).name}


def add_image(ds: Path, filename: str, data: bytes, known: dict) -> dict:
    """Кладёт один снимок в набор. Возвращает {"file"} или {"skipped"}.

    Имя — время добавления и начало md5: так список сортируется по времени,
    а имя пациента из имени файла (бывает и такое) в набор не попадает.
    Исходное имя остаётся в реестре — врачу найти свой снимок.
    """
    import hashlib
    suffix = Path(filename).suffix.lower()
    if suffix not in IMG_EXTS:
        return {"original": filename, "skipped": "не JPG и не PNG"}
    if len(data) < 1024:
        return {"original": filename, "skipped": "пустой или повреждённый файл"}
    digest = hashlib.md5(data).hexdigest()
    if digest in known:
        return {"original": filename,
                "skipped": f"этот снимок уже есть: {known[digest]}"}
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"original": filename, "skipped": "файл не читается как снимок"}
    now = datetime.now()
    name = f"{now:%Y%m%d-%H%M%S}-{digest[:6]}{'.jpg' if suffix == '.jpeg' else suffix}"
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "images" / name).write_bytes(data)
    reg = read_registry(ds)
    reg[name] = {"added_at": now.isoformat(timespec="seconds"),
                 "original_name": Path(filename).name, "md5": digest,
                 "bytes": len(data), "file": name}
    write_registry(ds, reg)
    known[digest] = f"{ds.name}/{name}"
    return {"original": filename, "file": name}


@app.post("/api/datasets/{slug}/images")
async def add_images(slug: str, files: list[UploadFile] = File(...)):
    """Добавить панорамы в существующий набор и сразу отдать их ИИ."""
    ds = dataset_dir(slug)
    if ds is None:
        return JSONResponse({"error": "Датасет не найден"}, status_code=404)
    known = annotation_feed.known_hashes()
    results = [add_image(ds, f.filename or "", await f.read(), known)
               for f in files]
    added = [r for r in results if "file" in r]
    if added:
        analyze(slug, force=False)
    return {"added": added, "skipped": [r for r in results if "skipped" in r]}


@app.delete("/api/datasets/{slug}/images/{name}")
def delete_image(slug: str, name: str):
    """Снимок и его разметка — в корзину набора, не насовсем."""
    ds = dataset_dir(slug)
    p = find_image(ds, name) if ds else None
    if p is None:
        return JSONResponse({"error": "Снимок не найден"}, status_code=404)
    st = analyze_status.get(slug)
    if st and st["running"]:
        return JSONResponse({"error": "Идёт анализ ИИ — дождитесь окончания"},
                            status_code=409)
    bin_ = ds / "trash" / f"{datetime.now():%Y%m%d-%H%M%S}-{p.stem}"
    bin_.mkdir(parents=True, exist_ok=True)
    for src in (p, ds / "labels" / f"{p.stem}.txt",
                ds / "labels_human" / f"{p.stem}.txt",
                ds / "labels_human" / f"{p.stem}.json"):
        if src.is_file():
            shutil.move(str(src), str(bin_ / f"{src.parent.name}__{src.name}"))
    reg = read_registry(ds)
    if p.name in reg:
        # Из реестра убираем: иначе тот же снимок нельзя будет добавить снова.
        (bin_ / "registry.json").write_text(
            json.dumps(reg.pop(p.name), ensure_ascii=False), encoding="utf-8")
        write_registry(ds, reg)
    return {"ok": True}


@app.post("/api/datasets/upload")
async def upload_dataset(name: str = Form(...),
                         files: list[UploadFile] = File(...)):
    name = name.strip()
    if not name:
        return JSONResponse({"error": "Укажите название датасета"}, status_code=400)
    good = [f for f in files
            if f.filename and Path(f.filename).suffix.lower() in IMG_EXTS]
    if not good:
        return JSONResponse(
            {"error": "Нужен хотя бы один файл JPG или PNG"}, status_code=400)

    ds = _create_dataset(name)
    known = annotation_feed.known_hashes()
    results = [add_image(ds, Path(f.filename).name, await f.read(), known)
               for f in good]
    return {"slug": ds.name,
            "added": [r for r in results if "file" in r],
            "skipped": [r for r in results if "skipped" in r]}


@app.post("/api/datasets/{slug}/analyze")
def analyze(slug: str, force: bool = Query(False)):
    ds = dataset_dir(slug)
    if ds is None:
        return JSONResponse({"error": "Датасет не найден"}, status_code=404)
    st = analyze_status.get(slug)
    if st and st["running"]:
        return {"ok": True, "already_running": True}
    analyze_status[slug] = {"running": True, "done": 0, "total": 0,
                            "last_error": None}
    t = threading.Thread(target=_analyze_worker, args=(slug, force),
                         daemon=True)
    t.start()
    return {"ok": True}


@app.get("/api/datasets/{slug}/analyze_status")
def get_analyze_status(slug: str):
    if dataset_dir(slug) is None:
        return JSONResponse({"error": "Датасет не найден"}, status_code=404)
    st = analyze_status.get(slug)
    if st is None:
        return {"running": False, "done": 0, "total": 0, "last_error": None}
    return st


@app.delete("/api/datasets/{slug}")
def delete_dataset(slug: str):
    ds = dataset_dir(slug)
    if ds is None:
        return JSONResponse({"error": "Датасет не найден"}, status_code=404)
    st = analyze_status.get(slug)
    if st and st["running"]:
        return JSONResponse(
            {"error": "Идёт анализ — дождитесь окончания"}, status_code=409)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ds), str(TRASH_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"))
    analyze_status.pop(slug, None)
    return {"ok": True}


# ================= API: снимки и разметка внутри датасета =================

@app.get("/api/datasets/{slug}/images")
def list_images(slug: str) -> list[dict]:
    ds = dataset_dir(slug)
    if ds is None:
        return JSONResponse({"error": "Датасет не найден"}, status_code=404)
    reg = read_registry(ds)
    out = []
    for p in image_files(ds):
        rv = review_info(ds, p.stem)
        r = reg.get(p.name, {})
        out.append({
            "name": p.name,
            "has_ai": (ds / "labels" / f"{p.stem}.txt").is_file(),
            "has_human": (ds / "labels_human" / f"{p.stem}.txt").is_file(),
            "recheck": rv.get("recheck") or None,
            "added_at": r.get("added_at"),
            "original_name": r.get("original_name"),
            "reviewed_at": rv.get("first_reviewed_at"),
        })
    return out


@app.get("/api/datasets/{slug}/image/{name}")
def get_image(slug: str, name: str):
    ds = dataset_dir(slug)
    p = find_image(ds, name) if ds else None
    if p is None:
        return JSONResponse({"error": "Снимок не найден"}, status_code=404)
    return FileResponse(p)


@app.get("/api/datasets/{slug}/labels/{name}")
def get_labels(slug: str, name: str):
    ds = dataset_dir(slug)
    p = find_image(ds, name) if ds else None
    if p is None:
        return JSONResponse({"error": "Снимок не найден"}, status_code=404)
    stem = p.stem
    return {
        "ai": read_labels(ds / "labels" / f"{stem}.txt", with_score=True) or [],
        "human": read_human_labels(ds, stem),
    }


class BoxesPayload(BaseModel):
    boxes: list[dict] = Field(default_factory=list)
    origins: list[str] | None = None            # "ai" | "human", параллельно boxes
    scores: list[float | None] | None = None    # % уверенности ИИ, параллельно boxes


@app.post("/api/datasets/{slug}/labels/{name}")
def save_labels(slug: str, name: str, payload: BoxesPayload):
    ds = dataset_dir(slug)
    p = find_image(ds, name) if ds else None
    if p is None:
        return JSONResponse({"error": "Снимок не найден"}, status_code=404)
    lines = []
    for b in payload.boxes:
        try:
            cls = int(b["cls"])
            cx, cy, w, h = (float(b[k]) for k in ("cx", "cy", "w", "h"))
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"error": "Некорректный бокс"}, status_code=400)
        if not (0 <= cls < NUM_CLASSES):
            return JSONResponse({"error": "Неизвестный класс"}, status_code=400)
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 1e-4), 1.0)
        h = min(max(h, 1e-4), 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    (ds / "labels_human").mkdir(exist_ok=True)
    (ds / "labels_human" / f"{p.stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    # Сайдкар пишется ВСЕГДА: в нём лежит, по какому справочнику врач
    # проверял снимок. Неотмеченный зуб на проверенном снимке экспорт
    # считает здоровым — но только по классам, которые врач видел при
    # проверке. Снимок, проверенный при четырёх классах, ничего не говорит
    # о пломбах и эндо, и без этой отметки ушёл бы в обучение как «пломб
    # нет». Авторство и уверенность — только если длины сошлись.
    now = datetime.now().isoformat(timespec="seconds")
    # Норма «2 снимка в день» считается по ПЕРВОЙ проверке: пересохранение
    # вчерашнего снимка сегодняшнюю норму не закрывает.
    first = review_info(ds, p.stem).get("first_reviewed_at") or now
    side = {"reviewed_classes": NUM_CLASSES,
            "first_reviewed_at": first, "reviewed_at": now}
    if payload.origins is not None \
            and len(payload.origins) == len(payload.boxes) \
            and all(o in ("ai", "human") for o in payload.origins):
        scores = payload.scores
        if scores is None or len(scores) != len(payload.boxes):
            scores = [None] * len(payload.boxes)
        side["origins"] = payload.origins
        side["scores"] = scores
    (ds / "labels_human" / f"{p.stem}.json").write_text(
        json.dumps(side), encoding="utf-8")
    return {"ok": True}


@app.delete("/api/datasets/{slug}/labels/{name}")
def delete_labels(slug: str, name: str):
    ds = dataset_dir(slug)
    p = find_image(ds, name) if ds else None
    if p is None:
        return JSONResponse({"error": "Снимок не найден"}, status_code=404)
    lbl = ds / "labels_human" / f"{p.stem}.txt"
    if lbl.is_file():
        lbl.unlink()
    sidecar = ds / "labels_human" / f"{p.stem}.json"
    if sidecar.is_file():
        sidecar.unlink()
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()

    import uvicorn
    _analyze_pending()  # фоновые потоки; сервер поднимается, не дожидаясь их
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
