"""Разметка корневых каналов врачом на КЛКТ (FastAPI + фронт).

Запуск:
    .venv/Scripts/python.exe research/scripts/canal_annotate_server.py --port 7863

Открыть: http://localhost:7863 (инструкция врачам — /guide.html)

ЗАЧЕМ. Модель канала доводит его до верхушки на эталонных данных, но на нашем
Dentium прибавка частичная, и проверить её нечем: разметки каналов на нашем
аппарате не существует НИ ОДНОЙ. Здесь врач её создаёт, а мы получаем разом и
эталон для измерения, и данные для дообучения.

ЧТО ВИДИТ ВРАЧ. Один зуб за раз на РОДНЫХ срезах тома — том же виде, что во
вьюере аппарата: канал там тёмная точка в кольце дентина. Наш выпрямленный
вырез для врача не годился (его прямые слова: «фарш») — он пересчитан с шагом
0.16 мм из вокселей 0.31 мм, и каждый пиксель размазан по восьми соседям.
Рядом — продольный вид вдоль корня, собранный из тех же родных вокселей.
Линии каналов уже нарисованы моделью: работа врача — подтвердить или
дотянуть, а не рисовать с нуля.

ПОЧЕМУ ТОЧКИ ЖИВУТ В ВОКСЕЛЯХ ТОМА. Врач размечает на РОДНЫХ срезах тома —
том же виде, что во вьюере аппарата, — поэтому его точки сразу в вокселях
тома. Выпрямленная сетка нужна только модели, и в неё точки переводятся
здесь же (`to_grid`). Обратный порядок был бы хрупким: вырез строится по
маске зуба текущей модели, и при смене сегментации разметка, записанная в
сетке, молча уехала бы вместе с рамкой.

ЧТО СЧИТАЕТСЯ СДЕЛАННЫМ. Зуб засчитан, только если врач отметил `complete`:
все каналы этого зуба разобраны. Частично размеченный зуб при экспорте дал бы
«канала нет» там, где врач просто не дошёл, и мы выучили бы ровно ту болезнь,
которую лечим. То же правило, что в разметке панорам.

Эндпоинты:
    GET  /api/doctors                  — список врачей
    POST /api/doctors                  — {"name"} -> {slug}
    GET  /api/queue?doctor=<slug>      — очередь зубов этого врача
    GET  /api/tooth/{id}?doctor=<slug> — описание зуба, черновики, своя разметка
    GET  /api/tooth/{id}/slices.png    — лист поперечных срезов
    GET  /api/tooth/{id}/body.png      — маска тела зуба тем же листом
    POST /api/tooth/{id}/mark          — сохранить разметку врача
    GET  /api/progress                 — сколько размечено, кем и за сегодня
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, date
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "research_3d/canal_annotation"
TEETH = DATA / "teeth"
MARKS = DATA / "marks"
STATIC = HERE / "canal_static"

DAILY_GOAL = 20          # зубов в день на врача; зуб с подсказкой — 10-20 секунд
DOUBLE_EVERY = 10        # каждый десятый зуб размечают ДВОЕ — мера согласия

app = FastAPI(title="Stomadent — разметка каналов")


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


# Фамилии врачи пишут кириллицей, а slug уходит и в путь на диске, и в адрес
# запроса — поэтому транслитерируем, а не оставляем кириллицу как есть.
TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e",
    "ю": "yu", "я": "ya", "ў": "u", "қ": "q", "ғ": "g", "ҳ": "h"})


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKC", name.strip().lower()).translate(TRANSLIT)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "doctor"


def tooth_dir(tooth_id: str) -> Path | None:
    if not re.fullmatch(r"[\w\-]+", tooth_id):
        return None
    d = TEETH / tooth_id
    return d if (d / "tooth.json").is_file() else None


def load_queue() -> list[dict]:
    p = DATA / "queue.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []


def doctors() -> dict:
    p = DATA / "doctors.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_doctors(d: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "doctors.json").write_text(json.dumps(d, ensure_ascii=False, indent=1),
                                       encoding="utf-8")


def marks_of(doctor: str) -> dict[str, dict]:
    d = MARKS / doctor
    out = {}
    if d.is_dir():
        for f in d.glob("*.json"):
            try:
                out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return out


def all_marked() -> dict[str, list[str]]:
    """Зуб -> кто его уже разметил."""
    out: dict[str, list[str]] = {}
    if MARKS.is_dir():
        for doc in MARKS.iterdir():
            if doc.is_dir():
                for f in doc.glob("*.json"):
                    out.setdefault(f.stem, []).append(doc.name)
    return out


def to_volume(points: list[list[float]], rec: dict) -> list[list[float]]:
    """Точки сетки выреза -> воксели ИСХОДНОГО тома.

    Без этого перевода разметка привязана к маске текущей модели и тихо
    сдвинется, как только сегментация станет лучше.

    Нет рамки (зуб готовили старой версией `cbct_canal_prepare`) — возвращаем
    пусто, а разметку всё равно сохраняем с пометкой `reproject`. Терять
    работу врача из-за нашей недоделки нельзя; пересчёт сделает
    `reproject_marks()` при следующем старте, когда рамка появится.
    """
    if not rec.get("frame"):
        return []
    f = rec["frame"]
    ctr = np.array(f["ctr"], float)
    ax, u, w = (np.array(f[k], float) for k in ("ax", "u", "w"))
    sp = np.array(rec["spacing"], float)
    off = np.array(rec["box_offset"], float)
    out = []
    for l, r, c in points:
        mm = (ctr + (f["a0"] + f["da"] * l) * ax
              + (f["b0"] + f["db"] * r) * u + (f["c0"] + f["dc"] * c) * w)
        out.append([round(v, 3) for v in (mm / sp + off)])
    return out


def to_grid(points: list[list[float]], rec: dict) -> list[list[float]]:
    """Воксели тома -> сетка выреза, в которой работает модель."""
    f = rec.get("frame")
    if not f:
        return []
    ctr = np.array(f["ctr"], float)
    ax, u, w = (np.array(f[k], float) for k in ("ax", "u", "w"))
    sp = np.array(rec["spacing"], float)
    off = np.array(rec["box_offset"], float)
    out = []
    for p in points:
        mm = (np.array(p, float) - off) * sp - ctr
        a, b, c = float(mm @ ax), float(mm @ u), float(mm @ w)
        out.append([round((a - f["a0"]) / f["da"], 3),
                    round((b - f["b0"]) / f["db"], 3),
                    round((c - f["c0"]) / f["dc"], 3)])
    return out


class Canal(BaseModel):
    # Точки приходят в вокселях тома (родные срезы). Старое поле `points`
    # (сетка выреза) оставлено для совместимости с первыми разметками.
    points_vox: list[list[float]] = Field(default_factory=list)
    points: list[list[float]] = Field(default_factory=list)   # [level, row, col] в сетке
    filled: bool = False        # канал запломбирован (яркий, не тёмный)
    unclear: bool = False       # канал не читается: облитерация, металл, артефакт
    name: str = ""


class Mark(BaseModel):
    doctor: str
    canals: list[Canal] = Field(default_factory=list)
    no_canals: bool = False     # каналов не видно вовсе
    complete: bool = True       # разобраны ВСЕ каналы зуба
    note: str = ""
    seconds: float = 0.0


@app.get("/api/doctors")
def api_doctors():
    return {"doctors": [{"slug": k, **v} for k, v in sorted(doctors().items())]}


class NewDoctor(BaseModel):
    name: str


@app.post("/api/doctors")
def api_new_doctor(body: NewDoctor):
    d = doctors()
    slug = slugify(body.name)
    if slug not in d:
        d[slug] = {"name": body.name.strip(),
                   "created_at": datetime.now().isoformat(timespec="seconds")}
        save_doctors(d)
    return {"slug": slug}


@app.get("/api/queue")
def api_queue(doctor: str = Query(...), limit: int = 200):
    """Очередь: сначала спорные зубы (там сигнала больше), свои уже размеченные
    не показываем. Каждый десятый зуб намеренно отдаётся ВТОРОМУ врачу — без
    меры согласия между людьми нельзя сказать, мала ли ошибка модели."""
    marked = all_marked()
    mine = set(marks_of(doctor))
    out = []
    for i, q in enumerate(load_queue()):
        others = [d for d in marked.get(q["id"], []) if d != doctor]
        if q["id"] in mine:
            continue
        if others and i % DOUBLE_EVERY != 0:
            continue
        out.append({**q, "second_opinion": bool(others)})
        if len(out) >= limit:
            break
    today = sum(1 for m in marks_of(doctor).values()
                if m.get("at", "").startswith(date.today().isoformat()))
    return {"queue": out, "done_today": today, "goal": DAILY_GOAL,
            "done_total": len(mine)}


@app.get("/api/tooth/{tooth_id}")
def api_tooth(tooth_id: str, doctor: str = ""):
    d = tooth_dir(tooth_id)
    if d is None:
        return JSONResponse({"error": "нет такого зуба"}, status_code=404)
    rec = json.loads((d / "tooth.json").read_text(encoding="utf-8"))
    rec["id"] = tooth_id
    mark = marks_of(doctor).get(tooth_id) if doctor else None
    return {"tooth": rec, "mark": mark}


@app.get("/api/tooth/{tooth_id}/{name}")
def api_tooth_image(tooth_id: str, name: str):
    d = tooth_dir(tooth_id)
    if d is None or name not in ("slices.png", "body.png",
                                 "axial.png", "axial_tooth.png"):
        return JSONResponse({"error": "нет файла"}, status_code=404)
    return FileResponse(d / name)


@app.post("/api/tooth/{tooth_id}/mark")
def api_mark(tooth_id: str, body: Mark):
    d = tooth_dir(tooth_id)
    if d is None:
        return JSONResponse({"error": "нет такого зуба"}, status_code=404)
    if not body.doctor or slugify(body.doctor) != body.doctor:
        return JSONResponse({"error": "не указан врач"}, status_code=400)
    rec = json.loads((d / "tooth.json").read_text(encoding="utf-8"))
    canals, need_reproject = [], not rec.get("frame")
    for cn in body.canals:
        if cn.points_vox:
            vox = [[float(a) for a in p] for p in cn.points_vox]
            grid = to_grid(vox, rec)
        else:                       # разметка первых версий — в сетке выреза
            grid = [[float(a) for a in p] for p in cn.points]
            vox = to_volume(grid, rec)
        canals.append({"points_grid": grid, "points_vox": vox,
                       "filled": cn.filled, "unclear": cn.unclear, "name": cn.name,
                       "end_level": max((p[0] for p in grid), default=None)})
    out = {"tooth": tooth_id, "case": rec["case"], "fdi": rec["fdi"],
           "doctor": body.doctor, "canals": canals, "no_canals": body.no_canals,
           "complete": body.complete, "note": body.note,
           "seconds": round(body.seconds, 1),
           "at": datetime.now().isoformat(timespec="seconds"),
           "reproject": need_reproject,
           "model_reach": {"base": rec["reach_base"], "net": rec["reach_net"],
                           "den": rec["reach_den"]}}
    p = MARKS / body.doctor / f"{tooth_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file():          # первая отметка времени не затирается
        old = json.loads(p.read_text(encoding="utf-8"))
        out["first_at"] = old.get("first_at", old.get("at"))
    else:
        out["first_at"] = out["at"]
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "canals": len(canals)}


@app.get("/api/progress")
def api_progress():
    rows, total_canals, seconds = [], 0, 0.0
    for doc, info in sorted(doctors().items()):
        m = marks_of(doc)
        today = sum(1 for v in m.values() if v.get("at", "").startswith(date.today().isoformat()))
        cn = sum(len(v.get("canals", [])) for v in m.values())
        sec = sum(v.get("seconds", 0) for v in m.values())
        total_canals += cn
        seconds += sec
        rows.append({"slug": doc, "name": info.get("name", doc), "teeth": len(m),
                     "today": today, "canals": cn,
                     "median_seconds": round(np.median([v.get("seconds", 0)
                                                        for v in m.values()]), 1) if m else 0})
    marked = all_marked()
    return {"doctors": rows, "teeth_total": len(load_queue()),
            "teeth_marked": len(marked), "canals": total_canals,
            "double_marked": sum(1 for v in marked.values() if len(v) > 1),
            "hours": round(seconds / 3600, 1)}


def reproject_marks() -> int:
    """Досчитать воксели тома у разметок, сохранённых без рамки выреза."""
    fixed = 0
    if not MARKS.is_dir():
        return 0
    for doc in MARKS.iterdir():
        for f in doc.glob("*.json") if doc.is_dir() else []:
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not m.get("reproject"):
                continue
            d = tooth_dir(m["tooth"])
            if d is None:
                continue
            rec = json.loads((d / "tooth.json").read_text(encoding="utf-8"))
            if not rec.get("frame"):
                continue
            for cn in m["canals"]:
                cn["points_vox"] = to_volume(cn["points_grid"], rec)
            m["reproject"] = False
            f.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
            fixed += 1
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    import uvicorn
    STATIC.mkdir(parents=True, exist_ok=True)
    n = reproject_marks()
    if n:
        print(f"пересчитаны координаты у {n} разметок")
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
    print(f"разметка каналов: http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
