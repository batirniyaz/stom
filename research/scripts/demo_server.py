"""Stomadent — демо-сервер для «вау»-презентации (FastAPI + статический фронт).

Запуск:
    .venv/Scripts/python.exe research/scripts/demo_server.py --port 7861

Открыть: http://localhost:7861
Эндпоинты:
    GET  /                      — single-page фронт (demo_static/index.html)
    GET  /api/examples          — список имён примеров из валидации DENTEX
    GET  /api/examples/{name}   — картинка примера (для превью)
    POST /api/predict           — multipart file (PNG/JPG/DICOM) + ?conf=0.25&model=ensemble
    POST /api/predict_example   — {"name": "val_15.png"} + ?conf=0.25&model=ensemble

Параметр model: ensemble (yolo12x + yolo26x_tuned, Weighted Boxes Fusion),
yolo12x, yolo26x, yolo26x_tuned.
"""

import json
import argparse
import base64
import io
import os
import re
import secrets
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from ultralytics import YOLO

import cbct_backend as cbct  # КЛКТ: кейсы, nnU-Net, меши, срезы
from annotation_classes import ENSEMBLE_MIN_CONF  # порог кариеса — один на демо и разметку
import patients as pat      # пациенты, исследования, анализы
import jobs                # очередь обработки: одна задача на GPU
import eventlog           # журнал запросов, задач и ошибок
import study_pipeline      # конвейеры КЛКТ, скана и 2D

ROOT = Path(__file__).resolve().parents[1]          # research/
RUNS = ROOT / "output/runs"
EXAMPLES_DIR = ROOT / "datasets/dentex_yolo/images/val"
CLINIC_DIR = ROOT / "datasets/annotation_sets/dentium-2026-08-12/images"
STATIC_DIR = Path(__file__).resolve().parent / "demo_static"

MODELS_INFO = {
    "ensemble": {
        "label": "Ансамбль (12x + 26x-tuned)",
        "members": ("yolo12x", "yolo26x_tuned"),
        "map50": 0.665,
    },
    "yolo12x": {
        "label": "YOLO12x",
        "weights": RUNS / "yolo12x_imgsz960/weights/best.pt",
        "map50": 0.647,
    },
    "yolo26x": {
        "label": "YOLO26x",
        "weights": RUNS / "yolo26x_imgsz960/weights/best.pt",
        "map50": 0.596,
    },
    "yolo26x_tuned": {
        "label": "YOLO26x (tuned)",
        "weights": RUNS / "yolo26x_tuned/weights/best.pt",
        "map50": 0.592,
    },
}
DEFAULT_MODEL = "ensemble"

ENSEMBLE_CONF = 0.05   # низкий порог инференса для ансамбля, фильтрация после WBF
WBF_IOU = 0.55

CLASS_RU = {
    "Impacted": "Ретенированный зуб",
    "Caries": "Кариес",
    "Periapical Lesion": "Периапикальное поражение",
    "Deep Caries": "Глубокий кариес",
}

CLASS_COLORS = {
    "Impacted": "#3b82f6",           # синий
    "Caries": "#eab308",             # жёлтый
    "Deep Caries": "#f97316",        # оранжевый
    "Periapical Lesion": "#ef4444",  # красный
}

MAX_WIDTH = 1600  # аннотации рисуем поверх уменьшенной копии

# Все веса грузим один раз при старте (GPU хватает на три модели).
#
# БЕЗ ВЕСОВ СЕРВЕР ОБЯЗАН ПОДНИМАТЬСЯ. Веса в репозиторий не кладутся: четыре
# файла по 119 МБ превышают жёсткий лимит GitHub в 100 МБ. Значит у
# разработчика, взявшего проект из git, моделей нет — и если падать на
# импорте, ему закрыт весь фронт: карточки пациентов, 3D, находки, план и
# PDF. Ничего из этого моделей не требует, всё считается по уже сохранённым
# кейсам. Поэтому берём те веса, что есть, а отсутствие отдаём честной
# ошибкой в момент запроса инференса (`require_models`).
yolo_models = {
    mid: YOLO(str(info["weights"]))
    for mid, info in MODELS_INFO.items()
    if "weights" in info and Path(info["weights"]).is_file()
}
MODELS_READY = "yolo12x" in yolo_models

# Порядок классов задан разметкой DENTEX и совпадает с порядком CLASS_RU.
# При живых весах имена берём у модели, иначе из справочника: без этого не
# собрался бы даже ответ со списком классов.
CLASS_NAMES = (yolo_models["yolo12x"].names if MODELS_READY
               else dict(enumerate(CLASS_RU)))


def require_models() -> None:
    """Инференс без весов невозможен — говорим об этом прямо, а не 500-й.

    Сообщение называет и причину, и то, что всё-таки работает: иначе
    разработчик без моделей видит трассировку и думает, что сломал код.
    """
    if not MODELS_READY:
        raise HTTPException(
            503,
            "Веса моделей не найдены в research/output/runs/. В репозитории "
            "их нет: файлы больше лимита GitHub в 100 МБ. Разделы, которым "
            "модели не нужны, работают — кейсы КЛКТ, находки, план, PDF, "
            "карточки пациентов.",
        )


app = FastAPI(title="Stomadent Demo", docs_url=None, redoc_url=None)


# Пути, где содержимое меняется только при пересборке кейса: их кэширует
# браузер, а свежесть проверяется по ETag.
CACHED_PREFIXES = ("/api/cbct/mesh/", "/api/ios/mesh/", "/api/ios/raw/",
                   "/api/cbct/panorama/")


def _cached_file(path: Path, media_type: str, request) -> Response:
    """Файл с проверкой по ETag: повторный запрос стоит 304, а не мегабайтов.

    ETag берём из размера и времени правки файла — этого достаточно, чтобы
    пересборка кейса сразу отдала новую версию, и не нужно считать хэш от
    шести мегабайт на каждый запрос.
    """
    st = path.stat()
    etag = f'"{st.st_size:x}-{int(st.st_mtime):x}"'
    headers = {"ETag": etag,
               # no-cache — это не «не кэшируй», а «кэшируй, но каждый раз
               # спроси». Ровно то, что нужно: врач никогда не увидит
               # устаревший меш, но и качать его повторно не будет.
               "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)

    # Сжатие. Замер на настоящих мешах: 6.51 МБ -> 3.12 МБ (48%) у КЛКТ и
    # 2.63 -> 1.39 у скана. Жмём ОДИН раз и держим рядом: на лету это 264 мс
    # процессорного времени при каждом открытии кейса, а меняется файл только
    # при пересборке.
    if "gzip" in request.headers.get("accept-encoding", ""):
        gz = path.with_suffix(path.suffix + ".gz")
        if not gz.is_file() or gz.stat().st_mtime < st.st_mtime:
            import gzip as _gzip
            tmp = gz.with_suffix(gz.suffix + ".tmp")
            tmp.write_bytes(_gzip.compress(path.read_bytes(), 6))
            tmp.replace(gz)          # подменяем целиком: половинчатый файл
                                     # браузер не должен увидеть никогда
        return FileResponse(gz, media_type=media_type, headers={
            **headers, "Content-Encoding": "gzip"})
    return FileResponse(path, media_type=media_type, headers=headers)


@app.middleware("http")
async def log_requests(request, call_next):
    """Каждый запрос в журнал: без этого нельзя отличить «задача считается»
    от «запрос не дошёл». Ровно на этом вопросе врача — «скан загружается,
    не зависло ли?» — и выяснилось, что ответить нечем."""
    t0 = time.perf_counter()
    path = request.url.path
    try:
        resp = await call_next(request)
    except Exception as exc:                       # noqa: BLE001
        eventlog.add("ERR", f"{request.method} {path} — {type(exc).__name__}: {exc}",
                     path=path, method=request.method)
        raise
    ms = round((time.perf_counter() - t0) * 1000)
    # Статику и опрос статуса не пишем: они идут пачками и утопят журнал,
    # в котором врач ищет свою загрузку.
    # Сам журнал и опрос статуса в журнал не пишем: экран журнала опрашивает
    # сервер каждые две секунды и за минуту утопил бы в себе ту самую
    # загрузку, ради которой врач его открыл.
    noisy = (path.startswith(("/fonts/", "/vue", "/app.css", "/api/logs"))
             or path.endswith((".png", ".jpg", ".ico"))
             or "/status" in path)
    if not noisy and (path.startswith("/api/") or resp.status_code >= 400):
        eventlog.add("ERR" if resp.status_code >= 400 else "REQ",
                     f"{request.method} {path} → {resp.status_code} за {ms} мс",
                     path=path, method=request.method,
                     code=resp.status_code, ms=ms)
    return resp


@app.middleware("http")
async def no_store_api(request, call_next):
    """Ничего не кэшируем: ни API, ни саму страницу.

    Раньше `no-store` стоял только на `/api/`, а страница и её скрипт шли
    через StaticFiles с обычными заголовками — браузер держал их у себя.
    Врач жаловался, что стороны «не поменялись», хотя на диске уже лежала
    исправленная сборка: он смотрел закэшированную страницу. Данные тут
    меняются между версиями обработки постоянно, кэшировать нечего.
    """
    resp = await call_next(request)
    path = request.url.path
    # Меши и панорамы из-под запрета выведены: это мегабайты, которые не
    # меняются, пока кейс не пересобран, а `no-store` заставлял браузер
    # качать их заново при каждом открытии. Им проставляется ETag, и повторный
    # заход стоит одного ответа 304 вместо шести мегабайт (см. _cached_file).
    if any(path.startswith(pref) for pref in CACHED_PREFIXES):
        return resp
    if path.startswith("/api/") or path.endswith((".html", ".js", ".css"))             or path == "/" or "." not in path.rsplit("/", 1)[-1]:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


def load_image(path: str) -> Image.Image:
    """Читает PNG/JPG или DICOM -> 8-bit RGB."""
    if path.lower().endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        arr = arr * slope + intercept
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            lo, hi = float(wc) - float(ww) / 2, float(wc) + float(ww) / 2
            arr = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
        else:
            arr = (arr - arr.min()) / max(arr.ptp(), 1)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 1.0 - arr
        img = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        img = Image.open(path)
    return img.convert("RGB")


def localize(cx: float, cy: float, w: int, h: int) -> str:
    """Приблизительная локализация: на ОПТГ право пациента — слева на снимке."""
    side = "правая" if cx < w / 2 else "левая"
    jaw = "верхняя челюсть" if cy < h / 2 else "нижняя челюсть"
    return f"{jaw}, {side} сторона"


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_on(color_hex: str) -> str:
    """Цвет текста на цветном чипе: тёмный на светлом, белый на тёмном."""
    r, g, b = (int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111827" if lum > 150 else "#ffffff"


def annotate(img: Image.Image, detections: list) -> tuple:
    """Рисует тонкие скруглённые рамки и чипы-подписи поверх (уменьшенной) копии.

    Возвращает (image, width, height, scale) — scale переводит координаты
    исходного снимка в координаты возвращаемой картинки.
    """
    w, h = img.size
    scale = min(1.0, MAX_WIDTH / w)
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    w2, h2 = img.size
    draw = ImageDraw.Draw(img)
    font = get_font(max(15, w2 // 62))
    lw = max(2, w2 // 640)
    radius = max(6, w2 // 200)
    chip_rects = []  # уже нарисованные чипы — чтобы не наезжать друг на друга

    def overlaps(rect) -> bool:
        rx1, ry1, rx2, ry2 = rect
        for a1, b1, a2, b2 in chip_rects:
            if rx1 < a2 and rx2 > a1 and ry1 < b2 and ry2 > b1:
                return True
        return False

    for det in detections:
        x1, y1, x2, y2 = (c * scale for c in det["box"])
        color = det["color"]
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius,
                               outline=color, width=lw)

        label = f'{det["class_ru"]} {det["confidence"]:.0%}'
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = max(8, lw * 3), max(4, lw * 2)
        ch_w, ch_h = tw + pad_x * 2, th + pad_y * 2
        cx1 = min(max(x1, 2), max(w2 - ch_w - 2, 2))
        # кандидаты: над рамкой, внутри сверху, под рамкой, внутри снизу,
        # затем — свободные ряды выше/ниже с шагом в высоту чипа
        candidates = [y1 - ch_h - lw - 3, y1 + lw + 3, y2 + lw + 3,
                      y2 - ch_h - lw - 3]
        candidates += [y1 - ch_h - lw - 3 - k * (ch_h + 3) for k in (1, 2, 3)]
        candidates += [y2 + lw + 3 + k * (ch_h + 3) for k in (1, 2, 3)]
        cy1 = candidates[0]
        for cand in candidates:
            if 2 <= cand <= h2 - ch_h - 2 and \
                    not overlaps((cx1, cand, cx1 + ch_w, cand + ch_h)):
                cy1 = cand
                break
        chip_rects.append((cx1, cy1, cx1 + ch_w, cy1 + ch_h))
        draw.rounded_rectangle([cx1, cy1, cx1 + ch_w, cy1 + ch_h],
                               radius=ch_h // 2, fill=color)
        draw.text((cx1 + pad_x - tb[0], cy1 + pad_y - tb[1]), label,
                  font=font, fill=_text_on(color))
    return img, w2, h2, scale


def predict_raw(model: YOLO, img: Image.Image, conf: float,
                max_det: int = 100) -> list:
    """[x1, y1, x2, y2, score, cls] в абсолютных координатах."""
    res = model.predict(img, imgsz=960, conf=conf, max_det=max_det,
                        verbose=False)[0]
    out = []
    for xyxy, cf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy(),
                             res.boxes.cls.cpu().numpy().astype(int)):
        out.append([*map(float, xyxy), float(cf), int(cls)])
    return out


def wbf(dets: list, n_models: int, iou_thr: float = WBF_IOU) -> list:
    """Weighted Boxes Fusion отдельно по каждому классу."""
    fused = []
    for cls in range(len(CLASS_NAMES)):
        d = sorted([x for x in dets if x[5] == cls], key=lambda x: -x[4])
        clusters = []  # каждый: [boxes...]
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
            # стандартный WBF score: средний вес с бонусом за согласие моделей
            score = min(1.0, (wsum / len(cl)) * (len(cl) / n_models + 0.5))
            fused.append([x1, y1, x2, y2, score, cls])
    return fused


def run_inference(img_path: str, conf: float, model_id: str = DEFAULT_MODEL) -> dict:
    require_models()
    conf = float(min(max(conf, 0.05), 0.9))
    if model_id not in MODELS_INFO:
        model_id = DEFAULT_MODEL
    info = MODELS_INFO[model_id]
    img = load_image(img_path)
    w, h = img.size

    t0 = time.perf_counter()
    if "members" in info:  # ансамбль: обе модели -> WBF -> фильтр по conf
        all_dets = []
        for member in info["members"]:
            all_dets += predict_raw(yolo_models[member], img, ENSEMBLE_CONF)
        raw = wbf(all_dets, len(info["members"]))
    else:
        raw = predict_raw(yolo_models[model_id], img, conf, max_det=30)
    # Кариес ниже своего минимума не показываем даже при низком ползунке
    # (замер и причина — в annotation_classes.ENSEMBLE_MIN_CONF).
    raw = [d for d in raw if d[4] >= max(conf, ENSEMBLE_MIN_CONF.get(d[5], 0.0))]
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    detections = []
    for x1, y1, x2, y2, cf, cls in raw:
        name = CLASS_NAMES[cls]
        ru = CLASS_RU.get(name, name)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append({
            "class_en": name,
            "class_ru": ru,
            "confidence": round(float(cf), 4),
            "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            "location": localize(cx, cy, w, h),
            "color": CLASS_COLORS.get(name, "#3b82f6"),
        })
    detections.sort(key=lambda d: -d["confidence"])

    annotated, w2, h2, scale = annotate(img.copy(), detections)
    buf = io.BytesIO()
    annotated.save(buf, "PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    # координаты боксов — в системе возвращаемой картинки (для оверлея на фронте)
    for det in detections:
        det["box"] = [round(v * scale, 1) for v in det["box"]]

    summary = {}
    for det in detections:
        summary[det["class_ru"]] = summary.get(det["class_ru"], 0) + 1

    return {
        "image_b64": image_b64,
        "width": w2,
        "height": h2,
        "findings": detections,
        "summary": summary,
        "inference_ms": inference_ms,
        "model": model_id,
        "model_label": info["label"],
    }


class ExampleRequest(BaseModel):
    name: str


def _example_path(name: str) -> Path | None:
    if not re.fullmatch(r"[\w.\-]+", name):
        return None
    p = (EXAMPLES_DIR / name).resolve()
    if p.parent != EXAMPLES_DIR.resolve() or not p.is_file():
        return None
    return p


@app.get("/api/examples")
def list_examples() -> dict:
    def key(p: Path):
        m = re.search(r"\d+", p.stem)
        return int(m.group()) if m else 0
    files = sorted(EXAMPLES_DIR.glob("*.png"), key=key)[:6]
    return {"examples": [f.name for f in files]}


@app.get("/api/examples/{name}")
def example_image(name: str):
    p = _example_path(name)
    if p is None:
        return JSONResponse({"error": "Пример не найден"}, status_code=404)
    return FileResponse(p)


@app.post("/api/predict_example")
def predict_example(req: ExampleRequest, conf: float = Query(0.25),
                    model: str = Query(DEFAULT_MODEL)):
    p = _example_path(req.name)
    if p is None:
        return JSONResponse({"error": "Пример не найден"}, status_code=404)
    return run_inference(str(p), conf, model)


# ---- примеры из клиники (кропнутые панорамы Dentium) ----

def _clinic_path(name: str) -> Path | None:
    if not re.fullmatch(r"[\w.\-]+", name):
        return None
    p = (CLINIC_DIR / name).resolve()
    if p.parent != CLINIC_DIR.resolve() or not p.is_file():
        return None
    return p


@app.get("/api/examples_clinic")
def list_examples_clinic() -> dict:
    files = sorted(CLINIC_DIR.glob("*.jpg"))[:5]
    return {"examples": [f.name for f in files]}


@app.get("/api/examples_clinic/{name}")
def example_clinic_image(name: str):
    p = _clinic_path(name)
    if p is None:
        return JSONResponse({"error": "Пример не найден"}, status_code=404)
    return FileResponse(p)


@app.post("/api/predict_example_clinic")
def predict_example_clinic(req: ExampleRequest, conf: float = Query(0.25),
                           model: str = Query(DEFAULT_MODEL)):
    p = _clinic_path(req.name)
    if p is None:
        return JSONResponse({"error": "Пример не найден"}, status_code=404)
    return run_inference(str(p), conf, model)


OPG_NAMES = {
    "impacted": "ретенция", "endo_treated": "канал запломбирован",
    "restoration": "пломба", "root_fragment": "фрагмент корня",
    "implant": "имплант", "prosthesis": "коронка или протез",
}


def _opg_teeth(path: str, conf: float) -> dict:
    """Находки на зуб для вьюера 2D, в координатах ПОКАЗАННОЙ картинки.

    Вьюер получает снимок уменьшенным до MAX_WIDTH (см. `annotate`), поэтому
    рамки пересчитываются тем же масштабом — иначе номера зубов легли бы не
    на свои зубы. Снимок при этом не обрезается, только масштабируется.

    DICOM модели зубов не читают, поэтому сначала он переводится в PNG тем же
    `load_image`, которым пользуется основной анализ, — окно и инверсия
    MONOCHROME1 должны совпадать.
    """
    import opg_teeth_findings as otf
    tmp = None
    try:
        src = path
        if path.lower().endswith(".dcm"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t:
                tmp = t.name
            load_image(path).save(tmp, "PNG")
            src = tmp
        res = otf.analyze(src, conf=conf)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    scale = min(1.0, MAX_WIDTH / max(res["width"], 1))
    for t in res["teeth"]:
        t["box"] = [round(v * scale, 1) for v in t["box"]]
    # Размеры — в тех же координатах, что и рамки: вьюер карточки пересчитывает
    # рамки в доли по этим числам. Исходные оставлены отдельно.
    res["orig_width"], res["orig_height"] = res["width"], res["height"]
    res["width"] = round(res["width"] * scale)
    res["height"] = round(res["height"] * scale)
    for f in res["findings"]:
        f["states_ru"] = [OPG_NAMES.get(x, x) for x in f.get("states", [])]
    res["scale"] = scale
    res["draft"] = True
    res["note"] = ("Черновик: пороги врачом не подтверждены. Отрицательного "
                   "заключения по панораме нет — часть кариеса на этот тип "
                   "снимка не проецируется.")
    return res


@app.post("/api/opg/teeth")
async def opg_teeth(file: UploadFile = File(...), conf: float = Query(0.25)):
    """Находки панорамы на ЗУБ: номер, кариес двумя схемами, состояния.

    В отличие от `/api/predict`, который отдаёт рамки патологий, здесь ответ
    привязан к номеру зуба и разделён на «обе схемы согласны» и «сказала
    одна». Замер на официальной валидации DENTEX: так пропускается 29
    кариозных зубов вместо 54 у одной схемы, а ложные тревоги уходят в
    отдельный список, а не подаются наравне с уверенными.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".dcm"):
        return JSONResponse({"error": "Поддерживаются PNG, JPG и DICOM"},
                            status_code=400)
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Пустой файл"}, status_code=400)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        return _opg_teeth(tmp_path, conf)
    except Exception as exc:  # noqa: BLE001 — демо показывает причину фронту
        log.exception("2D-анализ зубов не удался")
        return JSONResponse({"error": f"Не удалось обработать: {exc}"},
                            status_code=500)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class OpgExampleRequest(BaseModel):
    name: str
    kind: str = "dentex"


@app.post("/api/opg/teeth_example")
def opg_teeth_example(req: OpgExampleRequest, conf: float = Query(0.25)):
    """То же для примеров: DENTEX (`kind=dentex`) или снимков клиники."""
    p = _clinic_path(req.name) if req.kind == "clinic" else _example_path(req.name)
    if p is None:
        return JSONResponse({"error": "Пример не найден"}, status_code=404)
    try:
        return _opg_teeth(str(p), conf)
    except Exception as exc:  # noqa: BLE001
        log.exception("2D-анализ зубов не удался")
        return JSONResponse({"error": f"Не удалось обработать: {exc}"},
                            status_code=500)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), conf: float = Query(0.25),
                  model: str = Query(DEFAULT_MODEL)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".dcm"):
        return JSONResponse(
            {"error": "Поддерживаются форматы PNG, JPG и DICOM (.dcm)"},
            status_code=400)
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Пустой файл"}, status_code=400)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        return run_inference(tmp_path, conf, model)
    except Exception as exc:  # noqa: BLE001 — демо: показываем причину фронту
        return JSONResponse({"error": f"Не удалось обработать снимок: {exc}"},
                            status_code=500)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ================= КЛКТ (3D CBCT) =================

class CbctSegmentRequest(BaseModel):
    case_id: str
    model: str = "ours"  # ours | pretrained


@app.get("/api/cbct/examples")
def cbct_examples() -> dict:
    return {"examples": cbct.list_examples()}


@app.post("/api/cbct/upload")
async def cbct_upload(files: list[UploadFile] = File(...)):
    """ZIP с DICOM-серией или набор .dcm -> volume.mha, {case_id}."""
    from fastapi.concurrency import run_in_threadpool
    payload = [(f.filename, await f.read()) for f in files if f.filename]
    if not payload:
        return JSONResponse({"error": "Файлы не получены"}, status_code=400)
    try:
        c = await run_in_threadpool(cbct.import_upload, payload)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Не удалось прочитать серию: {exc}"},
                            status_code=500)
    return {"case_id": c["case_id"], "shape": c["shape"],
            "spacing": c["spacing"]}


@app.post("/api/cbct/segment")
def cbct_segment(req: CbctSegmentRequest):
    ok, msg = cbct.start_segment(req.case_id, req.model)
    if not ok:
        return JSONResponse({"error": msg}, status_code=404)
    return {"ok": True, "status": msg}


@app.get("/api/cbct/status/{case_id}")
def cbct_status(case_id: str):
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    return {"stage": c["stage"], "detail": c["detail"],
            "model": c["model"]}


@app.get("/api/cbct/info/{case_id}")
def cbct_info(case_id: str):
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    if not c.get("info"):
        return JSONResponse({"error": "Сегментация ещё не выполнена"},
                            status_code=409)
    # Аппарат и статус сторон идут вместе с информацией о кейсе: определить
    # левое и правое по файлу нельзя (аппараты клиники пишут позицию
    # пациента заглушкой), поэтому врач должен видеть, подтверждена ли
    # конвенция для ЭТОГО аппарата, а не узнавать об этом по перепутанным
    # номерам зубов в плане лечения.
    out = dict(c["info"])
    out["scanner"] = c.get("scanner") or {}
    out["sides"] = cbct.scanner_sides_state(c)
    return out


@app.get("/api/cbct/findings/{case_id}")
async def cbct_findings(case_id: str):
    """Слой находок: список фактов на зуб с черновой маршрутизацией."""
    from fastapi.concurrency import run_in_threadpool
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    model = cbct.mask_model_for(c)
    if model is None:
        return JSONResponse({"error": "Сегментация ещё не выполнена"},
                            status_code=409)
    res = await run_in_threadpool(cbct.findings_for, c, model)
    if res is None:
        return JSONResponse({"error": "Не удалось посчитать находки"},
                            status_code=500)
    # Отметки врача накладываем на выдаче, а не при подсчёте: находки
    # кэшируются в meta.json, и решение врача не должно в этот кэш попадать.
    # В ленте показываем и отклонённое — врач должен видеть, что отсмотрел.
    return cbct.apply_review(res, cbct.review_for(c), drop_rejected=False)


@app.post("/api/cbct/review/{case_id}")
async def cbct_review(case_id: str, body: dict = Body(...)):
    """Отметка врача по одной находке: подтверждена / отклонена / снята.

    Отклонённая находка не попадает в план — ради этого проверка и нужна.
    """
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    key = str(body.get("key") or "")
    state = body.get("state")
    if not key:
        return JSONResponse({"error": "нужен key находки"}, status_code=400)
    if state is not None and state not in cbct.REVIEW_STATES:
        return JSONResponse({"error": "state: confirmed | rejected | null"},
                            status_code=400)
    rev = cbct.set_review(c, key, state, str(body.get("by") or ""))
    return {"case_id": case_id, "review": rev}


@app.post("/api/cbct/chat/{case_id}")
async def cbct_chat(case_id: str, body: dict = Body(...)):
    """Вопрос по кейсу: отвечает ТОЛЬКО по находкам этого пациента.

    Вся защита — в `case_chat`: в контекст уходят только находки этого кейса,
    ФИО и снимки не уходят вовсе, а ответ без опоры на существующую находку
    не показывается. Здесь остаётся граница HTTP.

    Ответ несёт `used` — ключи находок, те же, что у отметок врача
    (`finding_key`). Поэтому ссылку под ответом можно сделать кликабельной и
    вести в ту же строку ленты.
    """
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    question = str(body.get("question") or "")
    if not question.strip():
        return JSONResponse({"error": "нужен question"}, status_code=400)
    import case_chat
    try:
        return case_chat.chat(case_id, question, body.get("model"),
                              body.get("complaints"))
    except RuntimeError as exc:
        # Нет ключа, провайдер не ответил, обезличивание не прошло — всё это
        # причины, которые должен видеть разработчик, а не «500».
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/api/cbct/summary/{case_id}")
async def cbct_summary(case_id: str, body: dict = Body(None)):
    """Первое сообщение чата: сводка находок с учётом жалоб пациента.

    POST, а не GET, именно из-за жалоб: это сведения о здоровье, и в строке
    адреса им быть нельзя — она попадает в журналы сервера и в историю
    браузера.

    `counts` в ответе посчитаны кодом, а не моделью: число проблем в сводке
    обязано совпадать с лентой находок.
    """
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    body = body or {}
    import case_chat
    try:
        return case_chat.summary(case_id, body.get("model"),
                                 body.get("complaints"))
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/api/cbct/sign/{case_id}")
async def cbct_sign(case_id: str, body: dict = Body(...)):
    """Подпись врача под текущим набором находок.

    Это ОТМЕТКА, а не электронная подпись: аутентификации в демо нет, имя
    врач вводит сам. Так и подписано в интерфейсе — выдавать её за
    юридически значимую нельзя.
    """
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    by = str(body.get("by") or "").strip()
    if not by:
        return JSONResponse({"error": "нужно имя врача"}, status_code=400)
    rev = cbct.sign_case(c, by, str(body.get("note") or "").strip())
    return {"case_id": case_id, "review": rev}


@app.get("/api/cbct/slice/{case_id}")
async def cbct_slice(case_id: str, axis: str = Query("z"),
                     idx: int = Query(0),
                     overlay: str = Query("contour")):
    from fastapi.concurrency import run_in_threadpool
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    model = cbct.mask_model_for(c)
    if model is None:
        return JSONResponse({"error": "Сегментация ещё не выполнена"},
                            status_code=409)
    if axis not in ("x", "y", "z"):
        axis = "z"
    if overlay not in ("contour", "fill", "none"):
        overlay = "contour"
    png = await run_in_threadpool(cbct.render_slice_png, c, model, axis, idx,
                                  overlay)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/cbct/teeth/{case_id}")
async def cbct_teeth(case_id: str):
    """Зубы кейса, для которых карточка строится."""
    from fastapi.concurrency import run_in_threadpool
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    model = cbct.mask_model_for(c)
    if model is None:
        return JSONResponse({"error": "Сегментация ещё не выполнена"},
                            status_code=409)
    teeth = await run_in_threadpool(cbct.teeth_present, c, model)
    return {"case_id": case_id, "model": model, "teeth": teeth}


@app.get("/api/cbct/tooth/{case_id}/{fdi}")
async def cbct_tooth_card(case_id: str, fdi: int):
    """Карточка зуба: геометрия и список плоскостей для срезов."""
    from fastapi.concurrency import run_in_threadpool
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    model = cbct.mask_model_for(c)
    if model is None:
        return JSONResponse({"error": "Сегментация ещё не выполнена"},
                            status_code=409)
    info = await run_in_threadpool(cbct.tooth_card, c, model, fdi)
    if info is None:
        return JSONResponse({"error": f"Зуб {fdi} не найден в маске"},
                            status_code=404)
    info["case_id"] = case_id
    info["model"] = model
    return info


@app.get("/api/cbct/tooth/{case_id}/{fdi}/{plane}.png")
async def cbct_tooth_slice(case_id: str, fdi: int, plane: str,
                           overlay: str = Query("contour"),
                           offset: float = Query(0.0)):
    """Срез зуба вдоль его собственной оси (мезио-дистальный и т.д.)."""
    from fastapi.concurrency import run_in_threadpool
    import cbct_tooth
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    if plane not in cbct_tooth.PLANES:
        return JSONResponse({"error": "Неизвестная плоскость"}, status_code=400)
    model = cbct.mask_model_for(c)
    if model is None:
        return JSONResponse({"error": "Сегментация ещё не выполнена"},
                            status_code=409)
    # Сдвиг ограничиваем: дальше 15 мм от зуба уже чужая анатомия, а поле
    # кадра осталось бы прежним — врач смотрел бы на соседний зуб, думая,
    # что смотрит на этот.
    offset = max(min(float(offset), 15.0), -15.0)
    png, meta = await run_in_threadpool(cbct.tooth_slice_png, c, model, fdi,
                                        plane, overlay != "fill", offset)
    if png is None:
        return JSONResponse({"error": f"Зуб {fdi} не найден в маске"},
                            status_code=404)
    return Response(content=png, media_type="image/png", headers={
        "Cache-Control": "no-store",
        # Размеры в мм — фронту, чтобы подписать масштаб под картинкой.
        "X-Tooth-Mm": f'{meta["w_mm"]}x{meta["h_mm"]}',
    })


@app.get("/api/cbct/panorama/{case_id}")
async def cbct_panorama_png(case_id: str):
    """Панорамная реконструкция (ОПТГ), построенная из томограммы."""
    from fastapi.concurrency import run_in_threadpool
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    model = cbct.mask_model_for(c)
    if model is None:
        return JSONResponse({"error": "Сегментация ещё не выполнена"},
                            status_code=409)
    try:
        png, _meta = await run_in_threadpool(cbct.panorama_for, c, model)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/cbct/panorama/{case_id}/meta")
async def cbct_panorama_meta(case_id: str):
    """Где на панораме какой зуб — чтобы ставить на неё находки."""
    from fastapi.concurrency import run_in_threadpool
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    model = cbct.mask_model_for(c)
    if model is None:
        return JSONResponse({"error": "Сегментация ещё не выполнена"},
                            status_code=409)
    try:
        _png, meta = await run_in_threadpool(cbct.panorama_for, c, model)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
    return meta


@app.get("/api/cbct/mesh/{case_id}")
def cbct_mesh(case_id: str, request: Request):
    c = cbct.get_case(case_id)
    if c is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    model = cbct.mask_model_for(c)
    gpath = cbct.case_dir(c) / f"mesh_{model}.glb" if model else None
    if gpath is None or not gpath.is_file():
        return JSONResponse({"error": "3D-модель ещё не построена"},
                            status_code=409)
    return _cached_file(gpath, "model/gltf-binary", request)


# ---- интраоральные сканы (IOS): отдельная страница, свой вьюер ----
IOS_CASES = Path(__file__).resolve().parents[2] / "research_ios" / "web_cases"
IOS_STATIC = Path(__file__).resolve().parent / "ios_static"


def _ios_case(name: str) -> Path | None:
    if not re.fullmatch(r"[\w\-]+", name):
        return None
    d = (IOS_CASES / name).resolve()
    if d.parent != IOS_CASES.resolve() or not d.is_dir():
        return None
    return d


@app.get("/api/ios/cases")
def ios_cases() -> dict:
    if not IOS_CASES.is_dir():
        return {"cases": []}
    return {"cases": sorted(d.name for d in IOS_CASES.iterdir()
                            if (d / "scan.glb").is_file())}


@app.get("/api/ios/mesh/{name}")
def ios_mesh(name: str, request: Request):
    d = _ios_case(name)
    if d is None or not (d / "scan.glb").is_file():
        return JSONResponse({"error": "Скан не найден"}, status_code=404)
    return _cached_file(d / "scan.glb", "model/gltf-binary", request)


@app.get("/api/cbct/superimpose/{case_id}")
def cbct_superimpose(case_id: str):
    """Какие сканы совмещены с этой томограммой и с каким качеством.

    Матрица считается офлайн (`ios_cbct_register.py`) и лежит у кейса скана:
    на запрос ICP не вешаем, он занимает десятки секунд. Отдаём ВСЕ найденные
    совмещения вместе с их RMS — включая непринятые, чтобы врач видел, что
    попытка была и чем кончилась, а не пустой список.
    """
    if cbct.get_case(case_id) is None:
        return JSONResponse({"error": "Кейс не найден"}, status_code=404)
    out = []
    if IOS_CASES.is_dir():
        for d in sorted(IOS_CASES.iterdir()):
            f = d / f"register_{case_id}.json"
            if not f.is_file():
                continue
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({k: r.get(k) for k in
                        ("ios_case", "matrix", "rms_mm", "coarse_rms_mm",
                         "tooth_median_mm", "common_teeth", "good", "note",
                         "draft")})
    return {"case_id": case_id, "registrations": out}


@app.get("/api/ios/raw/{name}")
def ios_raw(name: str, request: Request):
    """Исходный скан с родным цветом, без сегментации."""
    d = _ios_case(name)
    if d is None or not (d / "raw.glb").is_file():
        return JSONResponse({"error": "Исходный скан не собран"},
                            status_code=404)
    return _cached_file(d / "raw.glb", "model/gltf-binary", request)


@app.get("/api/ios/info/{name}")
def ios_info(name: str):
    d = _ios_case(name)
    if d is None or not (d / "scan.json").is_file():
        return JSONResponse({"error": "Скан не найден"}, status_code=404)
    return FileResponse(d / "scan.json", media_type="application/json")


@app.get("/api/ios/occlusion/{name}")
def ios_occlusion(name: str):
    """Карта окклюзионных контактов: зазоры на вершину + находки по прикусу.

    Строится офлайн (ios_occlusion_map.py) и лежит в кейсе: расчёт занимает
    десятки секунд, на запрос его не повесить.
    """
    d = _ios_case(name)
    if d is None or not (d / "occlusion.json").is_file():
        return JSONResponse({"error": "Карта прикуса не построена"},
                            status_code=404)
    return FileResponse(d / "occlusion.json", media_type="application/json")


@app.get("/api/plan")
async def treatment_plan(cbct_case: str = Query(None),
                        ios_case: str = Query(None)):
    """План лечения из находок КЛКТ и/или интраорального скана.

    Источники независимы: у клиники может быть только томограмма, только скан
    или и то и другое. Пациентской сущности, связывающей кейсы, пока нет,
    поэтому оба идентификатора приходят параметрами.
    """
    from fastapi.concurrency import run_in_threadpool
    import cbct_plan

    if not cbct_case and not ios_case:
        return JSONResponse({"error": "нужен хотя бы один источник: "
                                      "cbct_case или ios_case"}, status_code=400)
    res = None
    if cbct_case:
        c = cbct.get_case(cbct_case)
        if c is None:
            return JSONResponse({"error": "Кейс КЛКТ не найден"}, status_code=404)
        model = cbct.mask_model_for(c)
        if model is None:
            return JSONResponse({"error": "Сегментация ещё не выполнена"},
                                status_code=409)
        res = await run_in_threadpool(cbct.findings_for, c, model)
        if res is None:
            return JSONResponse({"error": "Не удалось посчитать находки"},
                                status_code=500)
        # Отклонённые врачом находки в план не идут.
        review = cbct.review_for(c)
        res = cbct.apply_review(res, review)
    ios = ios_teeth = None
    if ios_case:
        if _ios_case(ios_case) is None:
            return JSONResponse({"error": "Кейс скана не найден"}, status_code=404)
        ios, ios_teeth = cbct_plan.load_ios(ios_case)
    plan = cbct_plan.build_plan(res, ios, ios_teeth)
    if res is not None:
        # Подпись и счётчик проверки едут вместе с планом: врач должен видеть,
        # на КАКОМ наборе находок этот план собран.
        plan["review"] = res.get("review")
    return plan


@app.get("/api/ios/findings/{name}")
def ios_findings(name: str):
    """Находки по скану для всех специальностей (ios_findings.py)."""
    d = _ios_case(name)
    if d is None or not (d / "findings.json").is_file():
        return JSONResponse({"error": "Находки не посчитаны"}, status_code=404)
    return FileResponse(d / "findings.json", media_type="application/json")


# ================= пациенты и исследования =================
#
# Продукт был устроен вокруг файла, а врач работает вокруг пациента: у одного
# человека бывает КЛКТ, скан и панорама, и смотреть их надо вместе. Слой
# пациентов живёт в `patients.py` (SQLite + файлы), здесь только HTTP.
#
# Демо-кейсы перенесены как пациенты-примеры БЕЗ копирования данных: у их
# исследований проставлен `legacy_case`, и просмотр идёт по прежним адресам
# (/cbct/<кейс>, /ios/<кейс>). Иначе миграция утащила бы десятки гигабайт.


class PatientIn(BaseModel):
    last_name: str
    first_name: str = ""
    middle_name: str = ""
    birth_year: int | None = None
    sex: str | None = None
    card_no: str = ""
    note: str = ""


@app.get("/api/patients")
def api_patients(q: str = Query("", description="поиск по фамилии или карте"),
                 limit: int = 200) -> dict:
    return {"patients": pat.list_patients(q, limit)}


@app.post("/api/patients")
def api_patient_create(body: PatientIn) -> dict:
    try:
        pid = pat.create_patient(
            body.last_name, body.first_name, middle_name=body.middle_name,
            birth_year=body.birth_year, sex=body.sex, card_no=body.card_no,
            note=body.note)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"id": pid, "patient": pat.get_patient(pid)}


@app.get("/api/patients/{pid}")
def api_patient(pid: str) -> dict:
    p = pat.get_patient(pid)
    if p is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    studies = pat.list_studies(pid)
    # Для исследований из демо-кейсов сразу отдаём адрес просмотра: врачу
    # нужна ссылка, а не идентификатор.
    for st in studies:
        st["view_url"] = _study_view_url(st)
    return {"patient": p, "studies": studies,
            "registration": _registration_state(studies)}


def _registration_state(studies: list) -> dict:
    """Совмещено ли КЛКТ со сканом У ЭТОГО пациента.

    Наличие обоих исследований и наличие совмещения — РАЗНЫЕ вещи: матрица
    считается отдельным скриптом (`ios_cbct_register.py`), и без него в
    карточке горела зелёная отметка там, где ничего не выполнялось. Это тот
    же класс ошибки, что «находок не выявлено» на непосчитанном скане:
    интерфейс сообщает как факт то, чего не делали.
    """
    cbct_cases = [s["legacy_case"] for s in studies
                  if s["kind"] == "cbct" and s["legacy_case"]]
    ios_cases = [s["legacy_case"] for s in studies
                 if s["kind"] == "ios" and s["legacy_case"]]
    if not (cbct_cases and ios_cases):
        return {"possible": False, "done": False, "accepted": False}
    for ios in ios_cases:
        d = IOS_CASES / ios
        for cid in cbct_cases:
            f = d / f"register_{cid}.json"
            if not f.is_file():
                continue
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            return {"possible": True, "done": True,
                    "accepted": bool(r.get("good")),
                    "tooth_median_mm": r.get("tooth_median_mm"),
                    "rms_mm": r.get("rms_mm")}
    return {"possible": True, "done": False, "accepted": False}


@app.put("/api/patients/{pid}")
def api_patient_update(pid: str, body: PatientIn) -> dict:
    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    pat.update_patient(pid, **body.model_dump())
    return {"patient": pat.get_patient(pid)}


def _study_view_url(st: dict) -> str:
    """Куда ведёт исследование.

    У КЛКТ модель указывается ЯВНО. Без неё вьюер открывает модель по
    умолчанию (`ours`), а считали мы `sp` — врач попадал на пустой кейс и
    ждал, пока тот пересчитается другой сетью. Модель — часть адреса, а не
    подразумеваемое состояние.
    """
    case = st.get("legacy_case") or ""
    if st["kind"] == "cbct" and case:
        return f"/cbct/{case}/{st.get('model') or 'sp'}"
    if st["kind"] == "ios" and case:
        return f"/ios/{case}"
    return f"/studies/{st['id']}"


@app.post("/api/patients/{pid}/studies")
async def api_study_upload(pid: str, files: list[UploadFile] = File(...),
                           kind: str = Query("", description="тип; пусто — угадать")):
    """Загрузка исследования: файлы -> source/ -> очередь обработки."""
    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    names = [f.filename for f in files if f.filename]
    if not names:
        return JSONResponse({"error": "Файлы не получены"}, status_code=400)
    kind = kind or study_pipeline.guess_kind(names)
    if kind not in pat.STUDY_KINDS:
        return JSONResponse({"error": f"неизвестный тип: {kind}"}, status_code=400)

    sid = pat.create_study(pid, kind)
    src = pat.study_dir(sid) / "source"
    src.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not f.filename:
            continue
        # Имя из браузера в файловую систему не пускаем: путь приходит извне.
        safe = re.sub(r"[^\w.\-]+", "_", Path(f.filename).name)[:120] or "file"
        with open(src / safe, "wb") as out:
            while chunk := await f.read(1 << 20):
                out.write(chunk)
    try:
        jobs.enqueue(sid)
    except ValueError as exc:
        pat.set_study_status(sid, "error", str(exc))
        return JSONResponse({"error": str(exc), "study_id": sid}, status_code=400)
    return {"study_id": sid, "kind": kind, "kind_label": pat.STUDY_KINDS[kind],
            "state": jobs.state(sid)}


# ---- фотопротокол ----
#
# Фотографии не проходят инференс: клиника снимает их для контроля лечения и
# для показа пациенту. Поэтому отдельный набор ручек, а не очередь обработки.
# Единственная обработка — уменьшение до разумного размера: телефон отдаёт
# кадр на 4-8 МБ, а в карточке он показывается плиткой 320 px, и без сжатия
# страница с девятью снимками грузила бы полсотни мегабайт.

PHOTO_MAX_SIDE = 2000      # хватает, чтобы разглядеть край реставрации
PHOTO_QUALITY = 88
PHOTO_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
              ".webp": "image/webp", ".heic": "image/jpeg", ".heif": "image/jpeg"}


@app.get("/api/photo-slots")
def api_photo_slots():
    """Позиции стандартного протокола. Отдаём с сервера, чтобы подписи не
    разъехались между карточкой и будущим отчётом."""
    return {"slots": pat.PHOTO_SLOTS}


@app.get("/api/patients/{pid}/photos")
def api_photos(pid: str):
    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    out = []
    for ph in pat.list_photos(pid):
        out.append({
            "id": ph["id"], "slot": ph["slot"], "note": ph["note"],
            "taken_at": ph["taken_at"], "uploaded_at": ph["uploaded_at"],
            "width": ph["width"], "height": ph["height"], "bytes": ph["bytes"],
            "url": f"/api/photos/{ph['id']}/file",
        })
    return {"photos": out, "slots": pat.PHOTO_SLOTS}


@app.post("/api/patients/{pid}/photos")
async def api_photo_upload(pid: str, files: list[UploadFile] = File(...),
                           slot: str = Query(""),
                           taken_at: str = Query("")):
    """Загрузка пачкой. Слот приходит один на запрос: браузер отправляет
    девять отдельных запросов, по одному на позицию протокола, — так проще
    и показывать прогресс, и переснять одну позицию, не трогая остальные."""
    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return JSONResponse({"error": "Pillow не установлен"}, status_code=500)

    dst = pat.photos_dir(pid)
    dst.mkdir(parents=True, exist_ok=True)
    added = []
    for f in files:
        if not f.filename:
            continue
        raw = await f.read()
        ext = Path(f.filename).suffix.lower()
        if ext not in PHOTO_MIME:
            continue
        try:
            img = Image.open(io.BytesIO(raw))
            # Телефон пишет поворот в EXIF, а не в пикселях: без этого
            # половина протокола лежит на боку.
            img = ImageOps.exif_transpose(img)
            shot = _photo_taken_at(img) or (taken_at or None)
            img = img.convert("RGB")
        except Exception as exc:
            log.warning("фото %s не читается: %s", f.filename, exc)
            continue
        if max(img.size) > PHOTO_MAX_SIDE:
            k = PHOTO_MAX_SIDE / max(img.size)
            img = img.resize((max(1, round(img.width * k)),
                              max(1, round(img.height * k))), Image.LANCZOS)
        name = f"{secrets.token_hex(6)}.jpg"
        img.save(dst / name, "JPEG", quality=PHOTO_QUALITY, optimize=True)
        size = (dst / name).stat().st_size
        phid = pat.add_photo(pid, name, slot=slot, mime="image/jpeg",
                             width=img.width, height=img.height, size=size,
                             taken_at=shot)
        added.append({"id": phid, "slot": slot,
                      "url": f"/api/photos/{phid}/file"})
    if not added:
        return JSONResponse({"error": "Ни один файл не распознан как снимок"},
                            status_code=400)
    eventlog.add("photo", f"загружено снимков: {len(added)}", patient=pid,
                 slot=slot or "-")
    return {"added": added}


def _photo_taken_at(img) -> str | None:
    """Дата съёмки из EXIF. Врач сравнивает «до и после», и дата загрузки для
    этого не годится: архив за прошлый год загружается сегодня."""
    try:
        exif = img.getexif()
        if not exif:
            return None
        # 36867 DateTimeOriginal, 306 DateTime — в таком порядке приоритет.
        for tag in (36867, 306):
            v = exif.get(tag)
            if not v:
                continue
            t = datetime.strptime(str(v).strip(), "%Y:%m:%d %H:%M:%S")
            return t.isoformat(timespec="seconds")
    except Exception:
        return None
    return None


@app.get("/api/photos/{photo_id}/file")
def api_photo_file(photo_id: str):
    ph = pat.get_photo(photo_id)
    if not ph:
        return JSONResponse({"error": "Снимок не найден"}, status_code=404)
    f = pat.photos_dir(ph["patient_id"]) / ph["filename"]
    if not f.exists():
        return JSONResponse({"error": "Файл потерян"}, status_code=404)
    # Снимок не меняется после загрузки, поэтому в отличие от остального API
    # его можно кэшировать надолго: карточка с девятью плитками иначе
    # перекачивает их при каждом переключении вкладки.
    return FileResponse(f, media_type=ph["mime"],
                        headers={"Cache-Control": "private, max-age=86400"})


@app.patch("/api/photos/{photo_id}")
def api_photo_update(photo_id: str, body: dict = Body(...)):
    if not pat.get_photo(photo_id):
        return JSONResponse({"error": "Снимок не найден"}, status_code=404)
    pat.update_photo(photo_id, **{k: v for k, v in body.items()
                                  if k in ("slot", "note", "taken_at")})
    return {"ok": True}


@app.delete("/api/photos/{photo_id}")
def api_photo_delete(photo_id: str):
    if not pat.delete_photo(photo_id):
        return JSONResponse({"error": "Снимок не найден"}, status_code=404)
    return {"ok": True}


# ---- загрузка большими файлами: по частям ----
#
# Cloudflare Tunnel режет запрос на 100 МБ, а папка интраорального скана
# весит под двести: врач получал «Ошибка 413» и ничего не мог сделать.
# Обойти лимит прокси нельзя, но можно не превышать его: файл режется на
# части в браузере и собирается здесь. Побочно появляется честный прогресс —
# при загрузке в один запрос его взять негде.

CHUNK_DIR_NAME = "_chunks"


@app.post("/api/patients/{pid}/studies/chunk")
async def api_study_chunk(pid: str, request: Request,
                          upload_id: str = Query(...),
                          name: str = Query(...),
                          index: int = Query(...),
                          total: int = Query(...)):
    """Принять одну часть файла. Части собираются в файл при `index == total-1`."""
    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    if not re.fullmatch(r"[\w-]{6,64}", upload_id):
        return JSONResponse({"error": "неверный upload_id"}, status_code=400)
    safe = re.sub(r"[^\w.\-]+", "_", Path(name).name)[:120] or "file"
    tmp = pat.DATA_DIR / CHUNK_DIR_NAME / upload_id
    tmp.mkdir(parents=True, exist_ok=True)
    part = tmp / f"{safe}.part{index:04d}"
    with open(part, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    return {"received": index + 1, "of": total}


@app.post("/api/patients/{pid}/studies/finish")
def api_study_finish(pid: str, upload_id: str = Query(...),
                     kind: str = Query("")):
    """Собрать части в файлы, создать исследование и поставить в очередь."""
    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    if not re.fullmatch(r"[\w-]{6,64}", upload_id):
        return JSONResponse({"error": "неверный upload_id"}, status_code=400)
    tmp = pat.DATA_DIR / CHUNK_DIR_NAME / upload_id
    parts = sorted(tmp.glob("*.part*")) if tmp.is_dir() else []
    if not parts:
        return JSONResponse({"error": "части не получены"}, status_code=400)

    names = sorted({p.name.rsplit(".part", 1)[0] for p in parts})
    kind = kind or study_pipeline.guess_kind(names)
    if kind not in pat.STUDY_KINDS:
        return JSONResponse({"error": f"неизвестный тип: {kind}"}, status_code=400)

    sid = pat.create_study(pid, kind)
    src = pat.study_dir(sid) / "source"
    src.mkdir(parents=True, exist_ok=True)
    for name in names:
        chunks = sorted(p for p in parts if p.name.startswith(name + ".part"))
        with open(src / name, "wb") as out:
            for c in chunks:
                out.write(c.read_bytes())
    shutil.rmtree(tmp, ignore_errors=True)

    eventlog.add("JOB", f"собрано из частей: {kind}, файлов {len(names)}",
                 study=sid, study_kind=kind)
    try:
        jobs.enqueue(sid)
    except ValueError as exc:
        pat.set_study_status(sid, "error", str(exc))
        return JSONResponse({"error": str(exc), "study_id": sid}, status_code=400)
    return {"study_id": sid, "kind": kind, "kind_label": pat.STUDY_KINDS[kind],
            "state": jobs.state(sid)}


@app.get("/api/studies/{sid}/status")
def api_study_status(sid: str) -> dict:
    st = pat.get_study(sid)
    if st is None:
        return JSONResponse({"error": "Исследование не найдено"}, status_code=404)
    out = jobs.state(sid)
    out["view_url"] = _study_view_url(st)
    out["kind"] = st["kind"]
    return out


@app.post("/api/studies/{sid}/retry")
def api_study_retry(sid: str) -> dict:
    if pat.get_study(sid) is None:
        return JSONResponse({"error": "Исследование не найдено"}, status_code=404)
    try:
        jobs.enqueue(sid)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return jobs.state(sid)


@app.get("/api/studies/{sid}/image")
def api_study_image(sid: str, request: Request):
    """Исходный снимок исследования. Раньше 2D нигде не хранился — файл
    удалялся сразу после предсказания, и вернуться к нему было нельзя."""
    st = pat.get_study(sid)
    if st is None:
        return JSONResponse({"error": "Исследование не найдено"}, status_code=404)
    src = pat.study_dir(sid) / "source"
    files = [f for f in sorted(src.glob("*")) if f.is_file()] if src.is_dir() else []
    if not files:
        return JSONResponse({"error": "Файл снимка не найден"}, status_code=404)
    f = files[0]
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
             ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}
    mime = types.get(f.suffix.lower())
    if mime is None:                       # DICOM: отдаём как картинку
        img = load_image(str(f))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "no-cache"})
    return _cached_file(f, mime, request)


@app.get("/api/studies/{sid}/result")
def api_study_result(sid: str) -> dict:
    """Находки по 2D-снимку: что нашёл ансамбль."""
    st = pat.get_study(sid)
    if st is None:
        return JSONResponse({"error": "Исследование не найдено"}, status_code=404)
    f = pat.study_dir(sid) / "derived" / "findings2d.json"
    if not f.is_file():
        return JSONResponse({"error": "Анализ ещё не выполнен"}, status_code=409)
    data = json.loads(f.read_text(encoding="utf-8"))
    # Снимки, обработанные до порога кариеса, хранят находки от 0.25. Порог
    # применяем при выдаче, а не переписываем файл: правило может меняться,
    # а исходный ответ модели пусть лежит как был.
    min_by_name = {CLASS_NAMES[k]: v for k, v in ENSEMBLE_MIN_CONF.items()}
    data["findings"] = [x for x in data.get("findings", [])
                        if x.get("confidence", 1.0) >= min_by_name.get(x.get("class_en"), 0.0)]
    summary = {}
    for x in data["findings"]:
        summary[x["class_ru"]] = summary.get(x["class_ru"], 0) + 1
    data["summary"] = summary
    owner = pat.get_patient(st["patient_id"])
    return {"study": st, "patient": owner, "result": data}


def teeth2d_for(sid: str, compute: bool = True) -> dict | None:
    """Зубы и находки на зуб для 2D-исследования, с кэшем в derived/.

    Новые снимки считаются при обработке (`study_pipeline.handle_2d`), а для
    загруженных раньше — при первом открытии: пересчитывать всю базу заранее
    незачем, а врачу старый снимок нужен в том же виде, что и новый.
    """
    import opg_teeth_findings as otf
    d = pat.study_dir(sid) / "derived"
    f = d / "teeth2d.json"
    if f.is_file():
        try:
            cached = json.loads(f.read_text(encoding="utf-8"))
            # Кэш по прошлым правилам молча доживал бы до врача — та же
            # ловушка, что с находками КЛКТ без FINDINGS_VERSION.
            if cached.get("rules") == otf.RULES_VERSION:
                return cached
        except (OSError, ValueError):
            pass
    if not compute:
        return None
    src = pat.study_dir(sid) / "source"
    imgs = sorted(x for x in src.glob("*")
                  if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".dcm"))
    if not imgs:
        return None
    res = _opg_teeth(str(imgs[0]), 0.25)
    d.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


@app.get("/api/studies/{sid}/teeth")
def api_study_teeth(sid: str):
    """Зубная карта 2D-снимка: номера, кариес двумя схемами, очаги, состояния."""
    st = pat.get_study(sid)
    if st is None:
        return JSONResponse({"error": "Исследование не найдено"}, status_code=404)
    if st["kind"] not in ("opg", "periapical"):
        return JSONResponse({"error": "Зубная карта есть только у 2D-снимков"},
                            status_code=400)
    try:
        res = teeth2d_for(sid)
    except Exception as exc:  # noqa: BLE001
        log.exception("зубы 2D не посчитаны")
        return JSONResponse({"error": f"Не удалось обработать: {exc}"},
                            status_code=500)
    if res is None:
        return JSONResponse({"error": "Файл снимка не найден"}, status_code=404)
    return res


OPG_STATE_RU = {
    "endo_treated": "Канал запломбирован", "restoration": "Пломба",
    "prosthesis": "Коронка или протез", "implant": "Имплант",
    "impacted": "Ретенированный зуб", "root_fragment": "Фрагмент корня",
}


def teeth2d_findings(d: dict, study_id: str) -> list[dict]:
    """Находки 2D на зуб — в общем виде сводки пациента.

    Строка на группу («Кариес: зубы 36, 46»), а не на зуб: так читается
    лента, и у каждой строки есть список `teeth` для зубной карты.
    Уровень доверия переходит в severity: «обе схемы согласны» — warning и
    отметка на карте, «под вопросом» и состояния — info, на карте не
    отмечаются (иначе карта снова пометит почти всё).
    """
    groups: dict[tuple, list] = {}

    def add(key, tooth):
        groups.setdefault(key, []).append(tooth)

    for f in d.get("findings") or []:
        t = f.get("tooth")
        if f.get("code") == "caries":
            if f.get("level") == "confident":
                add(("deep" if f.get("deep") else "caries", "warning"), t)
            else:
                add(("caries_q", "info"), t)
        if f.get("apical_level") == "confident":
            add(("apical", "warning"), t)
        elif f.get("apical_level") == "possible":
            add(("apical_q", "info"), t)
        for x in f.get("states") or []:
            add(("state:" + x, "info"), t)

    names = {
        "caries": ("Кариес (обе схемы согласны)", "therapy"),
        "deep": ("Глубокий кариес (обе схемы согласны)", "therapy"),
        "caries_q": ("Кариес под вопросом — проверить на снимке", "therapy"),
        "apical": ("Периапикальный очаг (обе схемы согласны)", "therapy"),
        "apical_q": ("Периапикальный очаг под вопросом", "therapy"),
    }
    out = []
    for (key, sev), teeth in groups.items():
        teeth = sorted(set(teeth))
        if key.startswith("state:"):
            name = OPG_STATE_RU.get(key[6:], key[6:])
            route = "surgery" if key[6:] in ("impacted", "root_fragment") else "prostho"
        else:
            name, route = names[key]
        out.append({
            "code": "xray2d_" + key.replace("state:", ""),
            "teeth": teeth,
            "text": f"{name}: " + ", ".join(map(str, teeth)),
            "route": route, "severity": sev,
            "source": "opg", "source_label": "ОПТГ", "study_id": study_id,
        })
    return out


@app.get("/api/logs")
def api_logs(after: int = 0, limit: int = 300, kind: str = "") -> dict:
    """Журнал для экрана. `after` — только события новее этого id."""
    return {"events": eventlog.tail(limit, after, kind),
            "last_id": eventlog.last_id(), "stats": eventlog.stats(),
            "queue": jobs.queue_state()}


@app.get("/api/queue")
def api_queue() -> dict:
    return jobs.queue_state()


@app.get("/api/cases/{case_id}/owner")
def api_case_owner(case_id: str) -> dict:
    """Чей это кейс и что ещё есть у этого пациента.

    Вьюер показывает не только имя, но и переключатель исследований: врач
    сравнивает КЛКТ со сканом одного человека десятки раз за приём, и
    возвращаться ради этого в список — лишний шаг на каждое сравнение.
    """
    found = pat.find_by_case(case_id)
    if not found:
        return {"patient": None, "studies": []}
    others = pat.list_studies(found["study"]["patient_id"])
    for st in others:
        st["view_url"] = _study_view_url(st)
        st["current"] = st["id"] == found["study"]["id"]
    return {"patient": found["patient"], "study": found["study"],
            "studies": others}


@app.get("/api/patients/{pid}/summary")
def api_patient_summary(pid: str) -> dict:
    """Всё, что нашли у пациента, — в одном месте.

    Врач думает о человеке, а не о файлах: находки по КЛКТ, скану и панораме
    он держит в голове вместе. Раньше они лежали каждая внутри своего
    исследования, и карточка пациента показывала только список файлов —
    главное (что нашли и что с этим делать) приходилось собирать самому,
    открывая исследования по очереди.

    Формат находок у трёх источников разный, поэтому здесь они приводятся к
    одному виду с пометкой, откуда взялись: без этого нельзя ни отсортировать
    по важности, ни показать одной лентой.
    """
    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    studies = pat.list_studies(pid)
    out, teeth_present, missing = [], set(), set()

    # Берём ПОСЛЕДНЕЕ готовое исследование каждого типа. Иначе два скана
    # одного пациента дают одинаковые находки дважды, и лента читается как
    # «нашли два раза». Прежние снимки остаются в списке исследований — это
    # история, а не текущее состояние.
    latest = {}
    for st in studies:                       # list_studies уже сортирует по дате
        if st["status"] == "ready" and st["kind"] not in latest:
            latest[st["kind"]] = st

    for st in latest.values():
        case = st.get("legacy_case") or ""
        src = st["kind"]

        if src == "cbct" and case:
            c = cbct.get_case(case)
            if c:
                f = (c.get("findings") or {}).get("findings") or []
                for x in f:
                    out.append({**x, "source": "cbct",
                                "source_label": "КЛКТ", "study_id": st["id"]})
                info = c.get("info") or {}
                teeth_present |= set(info.get("teeth_present") or [])

        elif src == "ios" and case:
            # ТА ЖЕ функция, что у плана (`cbct_plan.load_ios`): она читает и
            # findings.json, и occlusion.json. Своё чтение здесь уже развело
            # источники — план ссылался на неполное смыкание и распределение
            # контактов, а в сводке этих находок не было, и агент честно
            # сообщил о противоречии, которого у пациента нет.
            import cbct_plan
            try:
                items, teeth = cbct_plan.load_ios(case)
            except Exception:                      # noqa: BLE001
                items, teeth = [], []
            for x in items:
                out.append({**x, "source": "ios",
                            "source_label": "Скан", "study_id": st["id"]})
            teeth_present |= set(teeth)

        elif src in ("opg", "periapical"):
            fp = pat.study_dir(st["id"]) / "derived" / "findings2d.json"
            # Зубы посчитаны — отдаём находки с номерами. Старый разрез «по
            # классу без номеров» остаётся только для снимков, где зубов ещё
            # нет (их посчитает первое открытие снимка).
            t2d = teeth2d_for(st["id"], compute=False)
            if t2d is not None:
                out.extend(teeth2d_findings(t2d, st["id"]))
                teeth_present |= {t["fdi"] for t in t2d.get("teeth") or []}
            elif fp.is_file():
                try:
                    d = json.loads(fp.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    d = {}
                # У 2D формат свой: класс, уверенность и место на снимке.
                # Приводим к общему виду, номера зубов там не определяются.
                # Группируем по классу патологии. На панораме их бывает
                # полтора десятка, и каждая рамка отдельной строкой
                # превращает ленту в простыню из одинаковых фраз: номера
                # зуба 2D-модель не даёт, различать строки нечем.
                by_class = {}
                for x in d.get("findings") or []:
                    name = x.get("class_ru") or "находка"
                    g = by_class.setdefault(name, {"n": 0, "places": []})
                    g["n"] += 1
                    if x.get("location"):
                        g["places"].append(x["location"])
                for name, g in by_class.items():
                    places = sorted(set(g["places"]))
                    tail = ("; ".join(places[:3])
                            + (" и др." if len(places) > 3 else "")) if places else ""
                    out.append({
                        "code": "xray2d", "teeth": [],
                        "text": (f"{name}: {g['n']} "
                                 + ("участок" if g["n"] == 1 else "участков")
                                 + (f" — {tail}" if tail else "")),
                        "route": "therapy", "severity": "warning",
                        "value": g["n"], "unit": "шт",
                        "source": "opg", "source_label": "ОПТГ",
                        "study_id": st["id"]})

    order = {"alarm": 0, "warning": 1, "info": 2}
    out.sort(key=lambda f: order.get(f.get("severity"), 3))
    attention = [f for f in out if f.get("severity") in ("alarm", "warning")]

    # Зубы с находками — для зубной карты: врач ищет глазами «где отмечено».
    # Описательные находки на ВЕСЬ ряд (симметрия дуги, индекс Болтона,
    # неровность ряда) в карту не идут: они относятся к трём десяткам зубов
    # разом, и карта, где помечено всё, ничего не выделяет.
    WHOLE_ROW = {"arch_asymmetry", "arch_irregularity", "arch_fit_poor",
                 "bolton", "bolton_skipped", "occlusion", "occlusion_summary",
                 "midline"}
    marked = {}
    for f in out:
        if f.get("code") in WHOLE_ROW or len(f.get("teeth") or []) > 8:
            continue
        if f.get("severity") not in ("alarm", "warning"):
            continue
        for t in f.get("teeth") or []:
            marked.setdefault(int(t), []).append(f.get("code"))

    return {
        "findings": out,
        "attention": attention[:12],
        "counts": {"total": len(out), "attention": len(attention)},
        "teeth_present": sorted(teeth_present),
        "teeth_marked": marked,
        "registration": _registration_state(studies),
    }


@app.get("/api/patients/{pid}/plan")
async def api_patient_plan(pid: str) -> dict:
    """Единый план лечения ПО ПАЦИЕНТУ, из всех его исследований.

    Раньше план считался по одному кейсу, и у человека с КЛКТ и сканом
    получалось два несвязанных плана — сшивал их врач в голове. Здесь берётся
    свежее исследование каждого типа и собирается один список этапов.
    """
    from fastapi.concurrency import run_in_threadpool
    import cbct_plan

    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    studies = pat.list_studies(pid)
    latest = {}
    for st in studies:
        if st["status"] == "ready" and st["kind"] not in latest:
            latest[st["kind"]] = st

    res = ios = ios_teeth = None
    sources = []
    cb_st = latest.get("cbct")
    if cb_st and cb_st.get("legacy_case"):
        c = cbct.get_case(cb_st["legacy_case"])
        model = cbct.mask_model_for(c) if c else None
        if c and model:
            res = await run_in_threadpool(cbct.findings_for, c, model)
            if res is not None:
                res = cbct.apply_review(res, cbct.review_for(c))
                sources.append("КЛКТ")

    ios_st = latest.get("ios")
    if ios_st and ios_st.get("legacy_case") and _ios_case(ios_st["legacy_case"]):
        ios, ios_teeth = cbct_plan.load_ios(ios_st["legacy_case"])
        sources.append("скан")

    plan = cbct_plan.build_plan(res, ios, ios_teeth)

    # Панорама: 2D-модель номеров зубов не даёт, поэтому позиции получаются
    # без списка зубов — «лечение кариеса, 14 участков». Это честнее, чем
    # придумывать номера: врач уточнит их на осмотре и по снимку.
    two_d = latest.get("opg") or latest.get("periapical")
    t2d = teeth2d_for(two_d["id"], compute=False) if two_d else None
    if t2d is not None:
        # Зубы посчитаны — позиции плана получают номера. В план идут ТОЛЬКО
        # находки, где согласны обе схемы: «под вопросом» — повод посмотреть
        # снимок, а не основание для лечения.
        by_key = {}
        for x in teeth2d_findings(t2d, two_d["id"]):
            by_key[x["code"]] = x["teeth"]
        rows = [
            ("xray2d_caries", "Лечение кариеса", "therapy"),
            ("xray2d_deep", "Лечение глубокого кариеса", "therapy"),
            ("xray2d_apical", "Оценка периапикального очага", "therapy"),
            ("xray2d_impacted", "Консультация хирурга: ретенция", "surgery"),
            ("xray2d_root_fragment", "Удаление фрагмента корня", "surgery"),
        ]
        added = False
        for code, name, stage in rows:
            teeth = by_key.get(code) or []
            if not teeth:
                continue
            for st_row in plan.get("stages", []):
                if st_row.get("key") == stage:
                    st_row.setdefault("procedures", []).append({
                        "name": name, "teeth": teeth, "count": len(teeth),
                        "stage": stage,
                        "why": "панорамный снимок: " + ", ".join(map(str, teeth)),
                        "confidence": "medium",
                        "note": "по панораме; подтвердить на осмотре",
                        "source": "ОПТГ"})
                    added = True
        if added:
            sources.append("ОПТГ")
    elif two_d:
        f = pat.study_dir(two_d["id"]) / "derived" / "findings2d.json"
        if f.is_file():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                d = {}
            counts = {}
            for x in d.get("findings") or []:
                counts[x.get("class_ru") or "находка"] =                     counts.get(x.get("class_ru") or "находка", 0) + 1
            names = {"Кариес": "Лечение кариеса",
                     "Глубокий кариес": "Лечение глубокого кариеса",
                     "Периапикальное поражение": "Эндодонтическое лечение",
                     "Ретенированный зуб": "Консультация хирурга: ретенция"}
            for cls, n in counts.items():
                stage = "surgery" if cls == "Ретенированный зуб" else "therapy"
                for st_row in plan.get("stages", []):
                    if st_row.get("key") == stage:
                        st_row.setdefault("procedures", []).append({
                            "name": names.get(cls, cls), "teeth": [], "count": n,
                            "stage": stage,
                            "why": f"панорамный снимок: {cls.lower()}, {n} шт",
                            "confidence": "medium",
                            "note": "номера зубов по 2D не определяются — "
                                    "уточнить на осмотре",
                            "source": "ОПТГ"})
            if counts:
                sources.append("ОПТГ")

    plan["sources"] = sources
    plan["patient_id"] = pid
    return plan


@app.post("/api/patients/{pid}/plan/compose")
async def api_plan_compose(pid: str) -> dict:
    """Сводный план, изложенный языковой моделью.

    Модель НЕ решает, что лечить: решения приняты правилами (`cbct_plan`), а
    она их излагает, ищет противоречия между исследованиями и говорит, чего
    не хватает. Позиция без основания во входных данных отбрасывается кодом —
    см. `plan_agent._validate`.

    Наружу уходят только обезличенные строки: номера зубов и измерения, без
    имени, даты рождения и номера карты.
    """
    from fastapi.concurrency import run_in_threadpool
    import plan_agent

    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    try:
        res = await run_in_threadpool(plan_agent.build, pid)
    except SystemExit as exc:              # нет ключа, модель не ответила
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:               # noqa: BLE001
        eventlog.add("ERR", f"агент плана: {type(exc).__name__}: {exc}",
                     patient=pid)
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                            status_code=500)
    eventlog.add("JOB", f"план собран моделью: этапов "
                        f"{len(res.get('stages') or [])}", patient=pid)
    return res


@app.get("/api/patients/{pid}/studies")
def api_patient_studies(pid: str) -> dict:
    if pat.get_patient(pid) is None:
        return JSONResponse({"error": "Пациент не найден"}, status_code=404)
    studies = pat.list_studies(pid)
    for st in studies:
        st["view_url"] = _study_view_url(st)
    return {"studies": studies, "kinds": pat.STUDY_KINDS}


@app.get("/api/studies/{sid}")
def api_study(sid: str) -> dict:
    st = pat.get_study(sid)
    if st is None:
        return JSONResponse({"error": "Исследование не найдено"}, status_code=404)
    st["view_url"] = _study_view_url(st)
    return {"study": st, "patient": pat.get_patient(st["patient_id"])}


# ================= адресация экранов по URL =================
#
# Врачу нужно давать ССЫЛКУ, а не инструкцию «нажмите сюда, потом сюда».
# Поэтому каждый экран адресуется путём, а состояние внутри экрана — тем же
# путём или query-параметром:
#
#     /2d                     двумерные снимки (ОПТГ)
#     /2d/<файл>              сразу открыть пример
#     /cbct                   КЛКТ: загрузка и примеры
#     /cbct/<кейс>            кейс с моделью по умолчанию
#     /cbct/<кейс>/<модель>   ours | pretrained | tf3
#     /ios                    интраоральные сканы
#     /ios/<кейс>             сегментация
#     /ios/<кейс>/raw         исходный скан
#
# Плюс ?tab=model|findings|plan — какая вкладка панели открыта.
#
# Маршруты объявлены ДО mount("/"), иначе статика перехватит всё. Страницы
# отдаются без кэша: иначе браузер держит старую разметку после правок.

PAGE_HEADERS = {"Cache-Control": "no-store"}


def _page(static_dir: Path, name: str = "index.html"):
    return FileResponse(static_dir / name, media_type="text/html",
                        headers=PAGE_HEADERS)


@app.get("/2d")
@app.get("/2d/{name}")
def page_2d(name: str = ""):
    return _page(STATIC_DIR)


@app.get("/cbct")
@app.get("/cbct/{case_id}")
@app.get("/cbct/{case_id}/{model}")
def page_cbct(case_id: str = "", model: str = ""):
    return _page(STATIC_DIR)


@app.get("/studies/{sid}")
def page_study(sid: str):
    """Просмотр исследования. Сейчас это 2D-снимок: у КЛКТ и скана свои
    вьюеры, и ссылка на них ведёт напрямую."""
    return _page(STATIC_DIR, "study.html")


@app.get("/logs")
def page_logs():
    return _page(STATIC_DIR, "logs.html")


@app.get("/design")
def page_design():
    """Прототип интерфейса. Лежит рядом с продуктом, чтобы врач смотрел его
    в том же браузере и на том же экране, что и рабочие вьюеры."""
    return _page(STATIC_DIR, "design.html")


@app.get("/patients")
@app.get("/patients/{pid}")
def page_patients(pid: str = ""):
    return _page(STATIC_DIR, "patients.html")


@app.get("/ios")
@app.get("/ios/")
@app.get("/ios/{case_id}")
@app.get("/ios/{case_id}/{mode}")
def page_ios(case_id: str = "", mode: str = ""):
    if not IOS_STATIC.is_dir():
        return JSONResponse({"error": "Вьюер сканов не установлен"},
                            status_code=404)
    return _page(IOS_STATIC)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    cbct.warmup()  # фоновая загрузка nnU-Net моделей КЛКТ

    study_pipeline.register_all()
    # Очередь живёт в памяти процесса. Всё, что осталось в работе с прошлого
    # запуска, результата уже не даст: помечаем ошибкой, иначе врач ждёт
    # исследование, которое никто не считает.
    stuck = jobs.reset_stuck()
    if stuck:
        print(f"прервано перезапуском: {stuck} исследований — нужен повтор")
    print(f"Пациенты: {len(pat.list_patients(limit=10000))} "
          f"(хранилище {pat.DATA_DIR})")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
