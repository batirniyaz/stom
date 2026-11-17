"""Журнал событий: запросы, задачи, ошибки — в памяти и на диске.

Зачем он появился. Врач загрузил интраоральный скан, увидел «загружается» и
спросил, не зависло ли. Ответить было нечем: сервер запускается с уровнем
`warning`, запросы не логируются вовсе, и в файле лежали только строчка о
старте и предупреждение nnU-Net. То есть отличить «задача идёт», «запрос не
дошёл» и «упало молча» было невозможно — ни ему, ни мне.

Что пишем:
  * REQ  — каждый запрос к API: метод, путь, код ответа, длительность;
  * JOB  — жизнь задачи: поставлена, стадия, готово, ошибка;
  * ERR  — исключения с типом и сообщением.

Две копии намеренно. В памяти — кольцо на MAX_EVENTS записей, его читает
экран журнала: быстро и без диска. На диске — файл, он переживает
перезапуск: половина интересных случаев (упал процесс, кончилась память)
как раз и означает, что в памяти уже ничего нет.

Тяжёлые тела запросов сюда не попадают: журнал должен помогать, а не
превращаться в свалку на гигабайт.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

MAX_EVENTS = 2000            # столько держим в памяти для экрана журнала
MAX_FILE_MB = 20             # больше — старый файл уходит в .1, начинаем новый

_lock = threading.Lock()
_events: deque = deque(maxlen=MAX_EVENTS)
_seq = 0

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = Path(os.environ.get("STOMADENT_DATA") or (REPO / "data")) / "logs"
LOG_FILE = LOG_DIR / "events.jsonl"


def _write_file(rec: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > MAX_FILE_MB * 1_000_000:
            LOG_FILE.replace(LOG_FILE.with_suffix(".jsonl.1"))
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:                       # noqa: BLE001
        # Журнал не имеет права ронять работу: если диск занят или полон,
        # событие просто не сохранится, а сервис продолжит принимать врача.
        pass


def add(kind: str, message: str, /, **fields) -> dict:
    """Записать событие. kind: REQ | JOB | ERR | INFO.

    Первые два параметра — строго позиционные (`/`). Иначе поле с именем
    `kind` в `**fields` сталкивается с именем параметра, и вызов падает с
    «got multiple values for argument». Это уже случилось: загрузка скана
    отвечала 500, а причина была в журнале, который её же и записывал.
    """
    global _seq
    with _lock:
        _seq += 1
        rec = {"id": _seq, "t": datetime.now().isoformat(timespec="seconds"),
               "ts": time.time(), "kind": kind, "msg": message, **fields}
        _events.append(rec)
    _write_file(rec)
    return rec


def tail(limit: int = 200, after: int = 0, kind: str = "") -> list[dict]:
    """Последние события. `after` — отдать только новее этого id."""
    with _lock:
        rows = [e for e in _events if e["id"] > after]
    if kind:
        wanted = set(kind.split(","))
        rows = [e for e in rows if e["kind"] in wanted]
    return rows[-limit:]


def last_id() -> int:
    with _lock:
        return _seq


def stats() -> dict:
    with _lock:
        rows = list(_events)
    errs = [e for e in rows if e["kind"] == "ERR"]
    return {
        "events": len(rows),
        "errors": len(errs),
        "last_error": errs[-1] if errs else None,
        "file": str(LOG_FILE),
        "file_mb": round(LOG_FILE.stat().st_size / 1e6, 2) if LOG_FILE.exists() else 0,
    }
