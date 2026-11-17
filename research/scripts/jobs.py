"""Очередь обработки исследований: одна задача на видеокарту за раз.

Зачем очередь вообще. Инференс КЛКТ и сегментация скана считаются на
видеокарте, и два одновременных запуска её переполняют. Windows при этом не
падает с ошибкой, а молча подменяет видеопамять оперативной: в одну из
сессий пересборка кейса заняла 94 ГБ RAM и встала намертво, а в логе не было
ни строчки. Единственный симптом — скорость. Поэтому задачи выполняются
СТРОГО по одной, а не «сколько прислали».

Что видит врач. Статус исследования различает «считается сейчас» и «третий в
очереди»: на пустом экране это неразличимо, и ожидание читается как поломка.
Стадия внутри задачи тоже показывается — «сегментация нейросетью» честнее,
чем крутящийся кружок без объяснений.

Что НЕ переживает перезапуск сервера. Очередь живёт в памяти процесса. При
старте демо все исследования, застрявшие в `processing` или `queued`,
помечаются ошибкой: считать их «готовыми» нельзя (результата нет), а тихо
перезапускать — тоже, врач не поймёт, почему видеокарта занята сама по себе.
Он нажимает «Повторить», и это осознанное действие.

Обработчики регистрируются снаружи (`register`), чтобы этот модуль не тянул
за собой ни nnU-Net, ни torch: демо импортирует его на старте, а веса грузятся
только когда работа действительно пришла.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import eventlog
import patients as pat

# Обработчик: (study_id, progress_callback) -> None. Исключение = ошибка задачи.
Handler = Callable[[str, Callable[[str], None]], None]

_handlers: dict[str, Handler] = {}
_lock = threading.Lock()
_q: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None


@dataclass
class Job:
    study_id: str
    kind: str
    status: str = "queued"          # queued | processing | ready | error
    stage: str = ""                 # что именно считается сейчас
    error: str = ""
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


_jobs: dict[str, Job] = {}          # study_id -> Job


def register(kind: str, handler: Handler) -> None:
    """Кто умеет считать исследование этого типа."""
    _handlers[kind] = handler


def enqueue(study_id: str) -> Job:
    """Поставить исследование в очередь. Повторная постановка не дублирует."""
    st = pat.get_study(study_id)
    if st is None:
        raise ValueError(f"исследования {study_id} нет")
    if st["kind"] not in _handlers:
        raise ValueError(f"нечем считать исследование типа {st['kind']}")
    with _lock:
        cur = _jobs.get(study_id)
        if cur and cur.status in ("queued", "processing"):
            return cur                      # уже в работе, второй раз не ставим
        job = Job(study_id=study_id, kind=st["kind"])
        _jobs[study_id] = job
    pat.set_study_status(study_id, "queued")
    eventlog.add("JOB", f"поставлено в очередь: {st['kind']} {study_id}",
                 study=study_id, study_kind=st["kind"], stage="queued")
    _q.put(study_id)
    _ensure_worker()
    return job


def job_of(study_id: str) -> Job | None:
    with _lock:
        return _jobs.get(study_id)


def position(study_id: str) -> int:
    """Место в очереди, начиная с 1. Ноль — считается сейчас или уже готово."""
    with _lock:
        waiting = [sid for sid, j in _jobs.items() if j.status == "queued"]
    waiting.sort(key=lambda s: _jobs[s].queued_at)
    return waiting.index(study_id) + 1 if study_id in waiting else 0


def state(study_id: str) -> dict:
    """Состояние для интерфейса: статус, стадия, место в очереди."""
    job = job_of(study_id)
    st = pat.get_study(study_id)
    if job is None:
        return {"status": (st or {}).get("status", "unknown"),
                "stage": "", "position": 0, "error": (st or {}).get("error", "")}
    return {"status": job.status, "stage": job.stage,
            "position": position(study_id), "error": job.error,
            "seconds": round((job.finished_at or time.time())
                             - (job.started_at or job.queued_at))}


def queue_state() -> dict:
    with _lock:
        jobs = list(_jobs.values())
    return {
        "running": [j.study_id for j in jobs if j.status == "processing"],
        "queued": len([j for j in jobs if j.status == "queued"]),
        "done": len([j for j in jobs if j.status == "ready"]),
        "failed": len([j for j in jobs if j.status == "error"]),
    }


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker and _worker.is_alive():
            return
        _worker = threading.Thread(target=_run, name="jobs", daemon=True)
        _worker.start()


def _run() -> None:
    while True:
        study_id = _q.get()
        job = job_of(study_id)
        if job is None:
            continue
        handler = _handlers.get(job.kind)
        if handler is None:
            _fail(job, f"нет обработчика для {job.kind}")
            continue
        job.status = "processing"
        job.started_at = time.time()
        pat.set_study_status(study_id, "processing")

        def progress(stage: str, _j=job) -> None:
            if stage != _j.stage:
                eventlog.add("JOB", f"{_j.kind} {_j.study_id}: {stage}",
                             study=_j.study_id, study_kind=_j.kind, stage=stage)
            _j.stage = stage

        try:
            handler(study_id, progress)
        except Exception as exc:                      # noqa: BLE001
            # Печатаем стек: одна строка «не найден файл» уже однажды увела
            # разбор в сторону — падать могло на любом шаге после сборки.
            traceback.print_exc()
            _fail(job, f"{type(exc).__name__}: {exc}")
            continue
        job.status = "ready"
        job.stage = ""
        job.finished_at = time.time()
        secs = round(job.finished_at - (job.started_at or job.finished_at))
        eventlog.add("JOB", f"готово: {job.kind} {study_id} за {secs} с",
                     study=study_id, study_kind=job.kind, stage="ready", seconds=secs)
        pat.set_study_status(study_id, "ready")


def _fail(job: Job, msg: str) -> None:
    eventlog.add("ERR", f"ошибка: {job.kind} {job.study_id} — {msg}",
                 study=job.study_id, study_kind=job.kind, stage="error")
    job.status = "error"
    job.error = msg
    job.finished_at = time.time()
    pat.set_study_status(job.study_id, "error", msg)


def reset_stuck() -> int:
    """Пометить ошибкой всё, что осталось в работе с прошлого запуска.

    Очередь живёт в памяти, поэтому после перезапуска демо запись
    `processing` означает только одно: процесс убили посреди счёта.
    Оставить её как есть — врач будет ждать результата, которого не будет.
    """
    n = 0
    for p in pat.list_patients(limit=10000):
        for st in pat.list_studies(p["id"]):
            if st["status"] in ("queued", "processing"):
                pat.set_study_status(
                    st["id"], "error",
                    "обработка прервана перезапуском сервера — запустите заново")
                n += 1
    return n
