"""Хранилище пациентов, исследований и анализов.

Продукт до сих пор был устроен вокруг ФАЙЛА: кейс КЛКТ в одной папке, скан в
другой, снимок ОПТГ вообще нигде (предсказывался из временного файла и
удалялся). Врач работает вокруг ПАЦИЕНТА, и на вопрос «что у нас есть по
этому человеку» ответить было нечем. Этот модуль — тот слой, которого не
хватало.

Три уровня, и схлопывать их нельзя:

  ПАЦИЕНТ       человек. Возраст НЕ хранится, а вычисляется из года рождения:
                записанный возраст устаревает молча, и через год в карточке
                будет неправда, которую никто не заметит.
  ИССЛЕДОВАНИЕ  один загруженный набор данных: КЛКТ, ОПТГ, прицельный снимок
                или интраоральный скан. У каждого свой статус обработки.
  АНАЛИЗ        находки по одному исследованию, версии правил и подпись врача.
                Лежит рядом с исследованием, а не с пациентом: подпись
                относится к конкретному снимку и конкретной версии правил.

Почему индекс в SQLite, а не в JSON, как всё остальное в проекте. Список
пациентов нужен с поиском по фамилии, сортировкой по дате и проверкой на
дубли — на россыпи JSON-файлов это разваливается уже на первой сотне
записей, и каждый список требует обойти все файлы. SQLite есть в стандартной
библиотеке, сервера не требует и переживает параллельное чтение из демо и
скриптов. Тяжёлые данные (тома, маски, меши) остаются файлами: в базе им
делать нечего.

Где лежит: переменная окружения `STOMADENT_DATA`, по умолчанию `data/` рядом
с репозиторием. Через переменную — потому что данные переедут на другой том,
и переезд не должен требовать правки кода.

Запуск как CLI:
    .venv/Scripts/python.exe research/scripts/patients.py --list
    .venv/Scripts/python.exe research/scripts/patients.py --add "Иванов" "Пётр" --year 1985
    .venv/Scripts/python.exe research/scripts/patients.py --migrate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Путь настраивается извне: данные пациентов переедут на отдельный том, и
# переезд не должен быть правкой кода.
DATA_DIR = Path(os.environ.get("STOMADENT_DATA") or (REPO / "data"))
DB_PATH = DATA_DIR / "index.db"
PATIENTS_DIR = DATA_DIR / "patients"

# Типы исследований. Названия — то, как их называет врач, а не как называется
# формат файла: он приносит «панораму», а не «одиночный DICOM».
STUDY_KINDS = {
    "cbct": "КЛКТ",
    "opg": "Панорамный снимок (ОПТГ)",
    "periapical": "Прицельный снимок",
    "ios": "Интраоральный скан",
}

# Статусы обработки. `queued` отдельно от `processing` намеренно: видеокарта
# одна, задачи идут по очереди, и врачу важно различать «считается сейчас» и
# «третий в очереди» — иначе пустой экран выглядит как поломка.
STUDY_STATUS = ("uploaded", "queued", "processing", "ready", "error")

# Жалобы — свободный текст, но не бесконечный: он уходит и в контекст модели,
# и в шапку отчёта. 500 символов это 3-4 фразы врача, больше в жалобу не
# пишут. Предел ОДИН на всё: база, API и сборка контекста чата ссылаются
# сюда, иначе длинная жалоба обрезалась бы в трёх местах по-разному.
COMPLAINTS_MAX = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id          TEXT PRIMARY KEY,
    last_name   TEXT NOT NULL,
    first_name  TEXT NOT NULL DEFAULT '',
    middle_name TEXT NOT NULL DEFAULT '',
    birth_year  INTEGER,
    birth_date  TEXT,
    sex         TEXT,
    card_no     TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    is_example  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS patients_name ON patients(last_name, first_name);

-- ЗАПИСЬ НА ПРИЁМ. Отдельной таблицей, а не колонкой «следующий визит» у
-- пациента: записей у человека бывает несколько (лечение идёт этапами), и
-- колонка потеряла бы всё, кроме последней. Список показывает БЛИЖАЙШУЮ
-- будущую — прошедшие остаются историей.
--
-- Время храним строкой ISO с часовым поясом, как и всё остальное в базе:
-- сравнение строк ISO совпадает со сравнением времён, пока пояс один, а
-- клиника одна. Этого достаточно, и это не требует отдельного типа.
CREATE TABLE IF NOT EXISTS appointments (
    id          TEXT PRIMARY KEY,
    patient_id  TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    at          TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS appt_patient ON appointments(patient_id, at);

CREATE TABLE IF NOT EXISTS studies (
    id          TEXT PRIMARY KEY,
    patient_id  TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'uploaded',
    error       TEXT NOT NULL DEFAULT '',
    taken_at    TEXT,
    uploaded_at TEXT NOT NULL,
    scanner     TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    -- Ссылка на кейс старого формата (research_3d/web_cases, research_ios).
    -- Миграция не копирует гигабайты: примеры остаются на месте, а карточка
    -- пациента просто знает, где они лежат.
    legacy_case TEXT NOT NULL DEFAULT '',
    legacy_kind TEXT NOT NULL DEFAULT '',
    -- Какой моделью посчитано. Нужна в ссылке: вьюер КЛКТ без модели в
    -- адресе открывает модель по умолчанию и начинает считать заново.
    model       TEXT NOT NULL DEFAULT '',
    -- Жалобы и самочувствие НА ЭТОТ ВИЗИТ, словами врача («болит слева
    -- снизу на холодное»). У визита, а не у пациента: в тот и есть смысл
    -- истории — на следующем снимке жалоба другая, и сравнивать надо их.
    --
    -- Зачем в базе, а не параметром из фронта: с жалобой сводка отвечает на
    -- вопрос, с которым пришёл человек («проверьте сначала 36 и 37»), и она
    -- же попадает в отчёт. Храним как текст: разбирать её на симптомы — это
    -- отдельная задача, а врач пишет свободно.
    complaints  TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS studies_patient ON studies(patient_id, uploaded_at);

-- Фотопротокол. Отдельная таблица, а не вид исследования: фотографии не
-- проходят инференс, не имеют статуса обработки и грузятся пачкой по девять
-- штук за приём. Держать их в studies значило бы завести там поля, которые
-- у настоящих исследований всегда пустые.
CREATE TABLE IF NOT EXISTS photos (
    id          TEXT PRIMARY KEY,
    patient_id  TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    -- Позиция стандартного протокола (ключ из PHOTO_SLOTS) либо пусто,
    -- если снимок не укладывается в схему.
    slot        TEXT NOT NULL DEFAULT '',
    filename    TEXT NOT NULL,
    mime        TEXT NOT NULL DEFAULT 'image/jpeg',
    width       INTEGER NOT NULL DEFAULT 0,
    height      INTEGER NOT NULL DEFAULT 0,
    bytes       INTEGER NOT NULL DEFAULT 0,
    -- Дата съёмки: у фотопротокола их несколько за курс лечения, и врач
    -- сравнивает «до» и «после» именно по ней, а не по дате загрузки.
    taken_at    TEXT,
    note        TEXT NOT NULL DEFAULT '',
    uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS photos_patient ON photos(patient_id, taken_at);
"""

# Стандартный фотопротокол: девять позиций. Порядок клинический — сначала
# лицо, потом ряд, потом окклюзионные виды; в этом же порядке фотограф их и
# снимает, поэтому загруженная пачка раскладывается по слотам почти сама.
PHOTO_SLOTS = [
    {"key": "face_rest",    "label": "Лицо в покое",        "group": "Лицо"},
    {"key": "face_smile",   "label": "Улыбка",              "group": "Лицо"},
    {"key": "profile",      "label": "Профиль",             "group": "Лицо"},
    {"key": "occ_front",    "label": "Прикус спереди",      "group": "Зубной ряд"},
    {"key": "occ_right",    "label": "Прикус справа",       "group": "Зубной ряд"},
    {"key": "occ_left",     "label": "Прикус слева",        "group": "Зубной ряд"},
    {"key": "upper",        "label": "Верхняя челюсть",     "group": "Окклюзионно"},
    {"key": "lower",        "label": "Нижняя челюсть",      "group": "Окклюзионно"},
    {"key": "other",        "label": "Прочее",              "group": "Прочее"},
]
PHOTO_SLOT_KEYS = {s["key"] for s in PHOTO_SLOTS}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(5)}"


def connect() -> sqlite3.Connection:
    """Соединение с базой. Каталог создаётся сам — так просили при настройке."""
    PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    # WAL: демо читает базу, пока скрипт в неё пишет. Без него читатель
    # получал бы «database is locked» ровно в момент загрузки исследования.
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(SCHEMA)
    # База могла быть создана до появления колонки: ALTER молча пропускаем,
    # иначе демо не поднимется на уже существующих данных.
    for col in ("model TEXT NOT NULL DEFAULT ''",
                "complaints TEXT NOT NULL DEFAULT ''"):
        try:
            con.execute(f"ALTER TABLE studies ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    # Телефон и архив у пациента — тем же способом: база уже существует на
    # машине клиники, и пересоздавать её ради двух колонок нельзя.
    #
    # ТЕЛЕФОН в базе, а не в анкете: по нему ищут. Регистратура знает номер
    # и не знает номера карты — «позвонила Гулнара, у неё болит» это поиск
    # по телефону, а не по фамилии, которых в списке три одинаковых.
    for col in ("phone TEXT NOT NULL DEFAULT ''",
                # Архив — не удаление. Пациент, который не ходит третий год,
                # не должен занимать первый экран, но и терять его карту
                # нельзя: он вернётся, и история понадобится.
                "archived INTEGER NOT NULL DEFAULT 0"):
        try:
            con.execute(f"ALTER TABLE patients ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    return con


# ---------------------------------------------------------------- пациенты

def create_patient(last_name: str, first_name: str = "", *, middle_name: str = "",
                   birth_year: int | None = None, birth_date: str | None = None,
                   sex: str | None = None, card_no: str = "", note: str = "",
                   is_example: bool = False) -> str:
    last_name = (last_name or "").strip()
    if not last_name:
        raise ValueError("фамилия обязательна: без неё пациента не найти в списке")
    if birth_date and not birth_year:
        m = re.match(r"(\d{4})", birth_date)
        if m:
            birth_year = int(m.group(1))
    pid = new_id("p")
    t = now()
    with connect() as con:
        con.execute(
            "INSERT INTO patients (id, last_name, first_name, middle_name,"
            " birth_year, birth_date, sex, card_no, note, is_example,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, last_name, (first_name or "").strip(), (middle_name or "").strip(),
             birth_year, birth_date, sex, card_no.strip(), note.strip(),
             int(is_example), t, t))
    patient_dir(pid).mkdir(parents=True, exist_ok=True)
    return pid


def update_patient(pid: str, **fields) -> bool:
    allowed = {"last_name", "first_name", "middle_name", "birth_year",
               "birth_date", "sex", "card_no", "note", "phone"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return False
    sets["updated_at"] = now()
    cols = ", ".join(f"{k} = ?" for k in sets)
    with connect() as con:
        cur = con.execute(f"UPDATE patients SET {cols} WHERE id = ?",
                          (*sets.values(), pid))
    return cur.rowcount > 0


def get_patient(pid: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
    return _patient_row(row) if row else None


def list_patients(query: str = "", limit: int = 200) -> list[dict]:
    """Список пациентов. Поиск по фамилии, имени и номеру карты."""
    sql = ("SELECT p.*, (SELECT COUNT(*) FROM studies s WHERE s.patient_id = p.id)"
           " AS studies_count,"
           " (SELECT MAX(s.uploaded_at) FROM studies s WHERE s.patient_id = p.id)"
           " AS last_study_at,"
           # Типы исследований прямо в списке: врач ищет глазами «у кого есть
           # КЛКТ», и заходить в каждую карточку ради этого он не станет.
           " (SELECT GROUP_CONCAT(DISTINCT s.kind) FROM studies s"
           "  WHERE s.patient_id = p.id) AS kinds"
           " FROM patients p")
    args: list = []
    q = (query or "").strip()
    if q:
        # Телефон ищем по ЦИФРАМ: врач набирает «61 222 14 07», а в базе
        # «+998 61 222 14 07» — поиск по строке не нашёл бы ничего.
        digits = re.sub(r"\D", "", q)
        sql += (" WHERE p.last_name LIKE ? OR p.first_name LIKE ?"
                " OR p.card_no LIKE ?")
        args += [f"%{q}%"] * 3
        if len(digits) >= 3:
            sql += " OR REPLACE(REPLACE(REPLACE(p.phone,' ',''),'-',''),'+','') LIKE ?"
            args.append(f"%{digits}%")
    sql += " ORDER BY COALESCE(last_study_at, p.created_at) DESC LIMIT ?"
    args.append(int(limit))
    with connect() as con:
        rows = con.execute(sql, args).fetchall()
    return [_patient_row(r) for r in rows]


# ------------------------------------------------------------- записи на приём

def add_appointment(patient_id: str, at: str, reason: str = "") -> dict:
    """Записать пациента на приём.

    `at` — время ISO с поясом. Проверяем, что строка разбирается: битая
    запись не упадёт здесь, а тихо не покажется в списке, и врач решит, что
    запись потерялась.
    """
    if get_patient(patient_id) is None:
        raise ValueError("пациента нет")
    try:
        datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        raise ValueError("время записи должно быть в формате ISO")
    rec = {"id": "a_" + secrets.token_hex(5), "patient_id": patient_id,
           "at": str(at), "reason": (reason or "").strip()[:200],
           "created_at": now()}
    with connect() as con:
        con.execute("INSERT INTO appointments (id, patient_id, at, reason,"
                    " created_at) VALUES (:id, :patient_id, :at, :reason,"
                    " :created_at)", rec)
    return rec


def delete_appointment(appt_id: str) -> bool:
    with connect() as con:
        cur = con.execute("DELETE FROM appointments WHERE id = ?", (appt_id,))
    return cur.rowcount > 0


def next_appointments(patient_ids: list[str]) -> dict[str, dict]:
    """Ближайшая БУДУЩАЯ запись каждого пациента — одним запросом.

    Тот же приём, что в `last_studies`: запрос на пациента в цикле на двух
    сотнях карточек дал бы двести обращений ради одной страницы.

    Прошедшие записи не возвращаем: в списке нужна та, к которой готовятся,
    а история визитов и так видна по исследованиям.
    """
    ids = [i for i in (patient_ids or []) if i]
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    cutoff = now()
    sql = (f"SELECT a.* FROM appointments a"
           f" JOIN (SELECT patient_id, MIN(at) AS m FROM appointments"
           f"       WHERE patient_id IN ({marks}) AND at >= ?"
           f"       GROUP BY patient_id) t"
           f"   ON t.patient_id = a.patient_id AND t.m = a.at")
    with connect() as con:
        rows = con.execute(sql, ids + [cutoff]).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        d = dict(r)
        prev = out.get(d["patient_id"])
        if prev is None or d["id"] < prev["id"]:
            out[d["patient_id"]] = d
    return out


def patient_appointments(patient_id: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM appointments WHERE patient_id = ?"
                           " ORDER BY at", (patient_id,)).fetchall()
    return [dict(r) for r in rows]


def set_archived(patient_id: str, archived: bool) -> bool:
    """Убрать карту из списка или вернуть обратно. Не удаление: пациент,
    который не ходит третий год, не должен занимать первый экран, но его
    история нужна, когда он вернётся."""
    with connect() as con:
        cur = con.execute("UPDATE patients SET archived = ?, updated_at = ?"
                          " WHERE id = ?",
                          (1 if archived else 0, now(), patient_id))
    return cur.rowcount > 0


def _patient_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["is_example"] = bool(d.get("is_example"))
    d["archived"] = bool(d.get("archived"))
    d["kinds"] = [k for k in (d.get("kinds") or "").split(",") if k]
    d["age"] = age_of(d)
    d["display"] = display_name(d)
    return d


def display_name(p: dict) -> str:
    parts = [p.get("last_name", ""), p.get("first_name", ""), p.get("middle_name", "")]
    return " ".join(x for x in parts if x).strip()


def age_of(p: dict) -> int | None:
    """Возраст считается, а не хранится: записанный устаревает молча."""
    year = p.get("birth_year")
    if not year:
        return None
    today = datetime.now()
    bd = p.get("birth_date") or ""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", bd)
    if m:
        y, mo, dd = (int(x) for x in m.groups())
        return today.year - y - ((today.month, today.day) < (mo, dd))
    return today.year - int(year)


def patient_dir(pid: str) -> Path:
    return PATIENTS_DIR / pid


# ------------------------------------------------------------ исследования

def create_study(patient_id: str, kind: str, *, taken_at: str | None = None,
                 scanner: str = "", note: str = "", status: str = "uploaded",
                 legacy_case: str = "", legacy_kind: str = "",
                 complaints: str = "") -> str:
    if kind not in STUDY_KINDS:
        raise ValueError(f"неизвестный тип исследования: {kind}")
    if status not in STUDY_STATUS:
        raise ValueError(f"неизвестный статус: {status}")
    if not get_patient(patient_id):
        raise ValueError(f"пациента {patient_id} нет")
    sid = new_id("s")
    t = now()
    with connect() as con:
        con.execute(
            "INSERT INTO studies (id, patient_id, kind, status, taken_at,"
            " uploaded_at, scanner, note, legacy_case, legacy_kind,"
            " complaints, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, patient_id, kind, status, taken_at, t, scanner, note,
             legacy_case, legacy_kind, (complaints or "").strip(), t))
    if not legacy_case:                       # у примеров данные лежат на месте
        (study_dir(sid) / "source").mkdir(parents=True, exist_ok=True)
        (study_dir(sid) / "derived").mkdir(parents=True, exist_ok=True)
    return sid


def update_study(sid: str, **fields) -> bool:
    """Правка полей исследования: аппарат, дата съёмки, ссылка на кейс.

    `legacy_case` проставляется уже ПОСЛЕ обработки: имя кейса рождается
    внутри конвейера (импорт DICOM сам решает, как назвать папку), а связать
    его с исследованием надо сразу, иначе результат осиротеет.
    """
    allowed = {"kind", "taken_at", "scanner", "note", "legacy_case",
               "legacy_kind", "model", "complaints"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return False
    cols = ", ".join(f"{k} = ?" for k in sets)
    with connect() as con:
        cur = con.execute(f"UPDATE studies SET {cols} WHERE id = ?",
                          (*sets.values(), sid))
    return cur.rowcount > 0


def set_study_status(sid: str, status: str, error: str = "") -> bool:
    if status not in STUDY_STATUS:
        raise ValueError(f"неизвестный статус: {status}")
    with connect() as con:
        cur = con.execute("UPDATE studies SET status = ?, error = ? WHERE id = ?",
                          (status, error, sid))
    return cur.rowcount > 0


def get_study(sid: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM studies WHERE id = ?", (sid,)).fetchone()
    return _study_row(row) if row else None


def list_studies(patient_id: str) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM studies WHERE patient_id = ?"
            " ORDER BY uploaded_at DESC", (patient_id,)).fetchall()
    return [_study_row(r) for r in rows]


def last_studies(patient_ids: list[str]) -> dict[str, dict]:
    """Последнее исследование каждого пациента — одним запросом.

    Нужно списку пациентов: врач смотрит на строку и хочет видеть, с чем
    человек приходил в последний раз. Запрос на пациента в цикле на
    двух сотнях карточек дал бы двести обращений к базе ради одной
    страницы.
    """
    ids = [i for i in (patient_ids or []) if i]
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    sql = (f"SELECT s.* FROM studies s"
           f" JOIN (SELECT patient_id, MAX(uploaded_at) AS m FROM studies"
           f"       WHERE patient_id IN ({marks}) GROUP BY patient_id) t"
           f"   ON t.patient_id = s.patient_id AND t.m = s.uploaded_at")
    with connect() as con:
        rows = con.execute(sql, ids).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        d = _study_row(r)
        # У двух исследований может совпасть время загрузки (пакетный ввоз
        # кладёт их одной секундой). Тогда берём любое, но устойчиво —
        # по идентификатору, чтобы список не прыгал между перезагрузками.
        prev = out.get(d["patient_id"])
        if prev is None or d["id"] < prev["id"]:
            out[d["patient_id"]] = d
    return out


def _study_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["kind_label"] = STUDY_KINDS.get(d["kind"], d["kind"])
    d["dir"] = str(study_dir(d["id"]))
    return d


def find_by_case(case_id: str) -> dict | None:
    """Кому принадлежит кейс старого формата.

    Обратная связь «кейс -> пациент» нужна вьюеру: врач открывает ссылку на
    кейс и должен видеть, чей это снимок и как вернуться в карточку. Без неё
    просмотр остаётся тупиком, из которого нет пути к остальным
    исследованиям того же человека.
    """
    with connect() as con:
        row = con.execute("SELECT * FROM studies WHERE legacy_case = ?"
                          " ORDER BY uploaded_at DESC LIMIT 1", (case_id,)).fetchone()
    if not row:
        return None
    st = _study_row(row)
    return {"study": st, "patient": get_patient(st["patient_id"])}


def study_dir(sid: str) -> Path:
    """Папка исследования. Пациент вычисляется из базы, поэтому путь стабилен."""
    with connect() as con:
        row = con.execute("SELECT patient_id FROM studies WHERE id = ?",
                          (sid,)).fetchone()
    if not row:
        raise ValueError(f"исследования {sid} нет")
    return patient_dir(row["patient_id"]) / "studies" / sid


def delete_study(sid: str) -> bool:
    """Удаляет запись. Файлы НЕ трогает: удаление данных — отдельное решение,
    и молча стирать том пациента нельзя."""
    with connect() as con:
        cur = con.execute("DELETE FROM studies WHERE id = ?", (sid,))
    return cur.rowcount > 0


# ----------------------------------------------------------------- анализ

def analysis_path(sid: str) -> Path:
    return study_dir(sid) / "analysis.json"


def save_analysis(sid: str, data: dict) -> Path:
    p = analysis_path(sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_analysis(sid: str) -> dict | None:
    p = analysis_path(sid)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                          # noqa: BLE001 — битый файл не должен ронять список
        return None


# --------------------------------------------------------------- миграция

# ------------------------------------------------------------- фотопротокол

def photos_dir(pid: str) -> Path:
    return patient_dir(pid) / "photos"


def add_photo(patient_id: str, filename: str, *, slot: str = "",
              mime: str = "image/jpeg", width: int = 0, height: int = 0,
              size: int = 0, taken_at: str | None = None,
              note: str = "") -> str:
    """Регистрирует снимок. Файл кладёт вызывающий — сюда приходит уже
    сохранённое имя, чтобы запись в базе не появилась раньше файла."""
    if slot and slot not in PHOTO_SLOT_KEYS:
        slot = ""
    pid = new_id("ph")
    with connect() as con:
        con.execute(
            "INSERT INTO photos (id, patient_id, slot, filename, mime, width,"
            " height, bytes, taken_at, note, uploaded_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (pid, patient_id, slot, filename, mime, int(width), int(height),
             int(size), taken_at, note, now()))
    return pid


def list_photos(patient_id: str) -> list[dict]:
    """Снимки пациента в порядке протокола, а не загрузки: врач ждёт лицо
    первым независимо от того, в каком порядке файлы попали на сервер."""
    order = {s["key"]: i for i, s in enumerate(PHOTO_SLOTS)}
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM photos WHERE patient_id = ?"
            " ORDER BY COALESCE(taken_at, uploaded_at), uploaded_at",
            (patient_id,)).fetchall()
    out = [dict(r) for r in rows]
    out.sort(key=lambda r: (r.get("taken_at") or r["uploaded_at"],
                            order.get(r["slot"], len(order))))
    return out


def get_photo(photo_id: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM photos WHERE id = ?",
                          (photo_id,)).fetchone()
    return dict(row) if row else None


def update_photo(photo_id: str, **fields) -> bool:
    allowed = {"slot", "note", "taken_at"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return False
    if "slot" in sets and sets["slot"] and sets["slot"] not in PHOTO_SLOT_KEYS:
        sets["slot"] = ""
    cols = ", ".join(f"{k} = ?" for k in sets)
    with connect() as con:
        cur = con.execute(f"UPDATE photos SET {cols} WHERE id = ?",
                          (*sets.values(), photo_id))
    return cur.rowcount > 0


def delete_photo(photo_id: str) -> bool:
    """Удаляет и запись, и файл. В отличие от исследования снимок весит
    мегабайты и переснимается за минуту — держать осиротевший файл незачем."""
    ph = get_photo(photo_id)
    if not ph:
        return False
    f = photos_dir(ph["patient_id"]) / ph["filename"]
    try:
        f.unlink(missing_ok=True)
    except OSError:
        pass
    with connect() as con:
        con.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
    return True


def migrate_examples() -> dict:
    """Существующие демо-кейсы становятся пациентами-примерами.

    Данные НЕ копируются: у примера проставляется `legacy_case`, и карточка
    знает, где лежит кейс. Иначе миграция утащила бы десятки гигабайт —
    один кейс КЛКТ это около 500 МБ.
    """
    made = {"patients": 0, "studies": 0, "skipped": 0}
    cbct_dir = REPO / "research_3d" / "web_cases"
    ios_dir = REPO / "research_ios" / "web_cases"

    def person_key(name: str) -> str | None:
        """Имя человека из имени папки кейса.

        `pair001_cbct` и `pair001_ios` — ОДИН пациент, снятый двумя способами
        (набор figshare так и устроен: КЛКТ и скан одного человека). Без
        отбрасывания суффикса они попали бы в список двумя карточками, и
        связь между исследованиями, ради которой всё затевается, потерялась
        бы на первом же шаге.

        `case_<цифры>` пропускаем: это следы загрузок через демо, у них нет
        ни имени, ни принадлежности человеку.
        """
        if name.startswith("_") or re.fullmatch(r"case_\d+", name):
            return None
        return re.sub(r"_(cbct|ios)$", "", name)

    groups: dict[str, dict] = {}
    for d in sorted(cbct_dir.glob("*/")) if cbct_dir.is_dir() else []:
        key = person_key(d.name)
        if key and (d / "meta.json").is_file():
            groups.setdefault(key, {})["cbct"] = d.name
    for d in sorted(ios_dir.glob("*/")) if ios_dir.is_dir() else []:
        key = person_key(d.name)
        if key and (d / "scan.glb").is_file():
            groups.setdefault(key, {})["ios"] = d.name

    with connect() as con:
        known = {r["legacy_case"] for r in
                 con.execute("SELECT legacy_case FROM studies"
                             " WHERE legacy_case <> ''").fetchall()}

    for name, kinds in groups.items():
        if all(k in known for k in kinds.values()):
            made["skipped"] += 1
            continue
        pid = create_patient(name, is_example=True,
                             note="демо-кейс, перенесён из web_cases")
        made["patients"] += 1
        for kind, case in kinds.items():
            if case in known:
                continue
            create_study(pid, kind, status="ready", legacy_case=case,
                         legacy_kind=kind)
            made["studies"] += 1
    return made


# Жалоба демо-пациента. Она НЕ случайная: сводка чата обязана учитывать
# жалобу («проверьте сначала 36 и 37»), и показать это можно только на
# кейсе, у которого жалоба есть. Пациент вымышленный, набор публичный
# (CC BY 4.0) — настоящих данных в демо нет.
FIXTURE_COMPLAINTS = "ноет слева снизу при накусывании, третий день"


def seed_fixture_patient() -> str | None:
    """Пациент-пример для кейса из `fixtures/`. None — не понадобилось.

    Зачем. Сценарий показа начинается со СПИСКА ПАЦИЕНТОВ, а на машине,
    взятой из git, снимков нет вовсе — список пуст, и демо начинается с
    пустого экрана. Фикстура даёт кейс, но не даёт визита: кейс сам по себе
    в карточке не появляется, их связывает `legacy_case`.

    Срабатывает ТОЛЬКО на пустой базе: у клиники свои пациенты, и
    подкладывать им демо-карточку нельзя.
    """
    fixtures = REPO / "fixtures"
    if not fixtures.is_dir():
        return None
    with connect() as con:
        if con.execute("SELECT 1 FROM patients LIMIT 1").fetchone():
            return None
    for meta in sorted(fixtures.glob("*/meta.json")):
        try:
            cid = (json.loads(meta.read_text(encoding="utf-8")) or {}).get("case_id")
        except (OSError, json.JSONDecodeError):
            continue
        if not cid:
            continue
        pid = create_patient("Демонстрационный", "пациент", is_example=True,
                             card_no="DEMO-001",
                             note="кейс из публичного набора (CC BY 4.0), "
                                  "не настоящий пациент")
        create_study(pid, "cbct", status="ready", legacy_case=cid,
                     legacy_kind="cbct", complaints=FIXTURE_COMPLAINTS)
        return pid
    return None


# Приём на показе идёт по списку, а список из одного человека не похож на
# рабочий день клиники: не видно ни сортировки по свежести, ни того, что у
# разных людей разное состояние разбора. Поэтому на ПУСТОЙ базе заводится
# небольшой список-пример.
#
# Все записи помечены `is_example`, и в списке у них стоит значок «пример»:
# выдавать выдуманных людей за пациентов клиники нельзя.
#
# Снимок привязан только к первому: у остальных исследование числится
# загруженным и неразобранным — это честное состояние, которое не станет
# ложью, если кто-то откроет карточку. Рисовать им находки было бы враньём:
# находок у них нет.
DEMO_ROSTER = [
    # (фамилия, имя, год, пол, карта, тип снимка, жалоба)
    ("Кутлымуратов", "Нурлан", 1968, "М", "NK-2045", "cbct",
     "подвижность нижних резцов, десна кровит при чистке"),
    ("Алламуратова", "Айгуль", 1999, "Ж", "NK-2044", "opg",
     "скученность, направлена ортодонтом"),
    ("Оразбаев", "Бахтияр", 1976, "М", "NK-2043", "cbct",
     "тянет в верхней челюсти справа, отдаёт в висок"),
    ("Жумабаева", "Гулнара", 1991, "Ж", "NK-2042", "cbct",
     "планирует имплантацию на месте 36"),
]

# Кейс, который открывают на показе. Снимок клиники: сегментация на нём
# заметно лучше, чем на публичной фикстуре, где у модели путаются зубы.
SHOWCASE = ("Сейтмуратов", "Азамат", 1984, "М", "NK-2041", "client5",
            "ноет слева снизу при накусывании, третий день")


# Анкета карточек-ПРИМЕРОВ: контакты, здоровье, история в клинике.
#
# ЭТИХ ПОЛЕЙ В БАЗЕ НЕТ И НЕ ЗАВОДИТСЯ. Решение осознанное: вести
# медицинскую анкету — это отдельный продукт (согласия, доступ, хранение
# персональных данных), и заводить под показ девять колонок, которые никто
# не заполняет, значит получить девять пустых граф у настоящих пациентов.
#
# Поэтому анкета лежит здесь, рядом с ростером, и отдаётся ТОЛЬКО карточкам
# с пометкой «пример». У настоящего пациента её нет вовсе, и карточка
# показывает то, что в базе действительно есть.
DEMO_PROFILES = {
    "NK-2041": {
        "phone": "+998 61 222 14 07", "addr": "Нукус, ул. Дослык, 12",
        "job": "инженер-строитель",
        "emerg": "супруга, +998 61 222 14 08",
        "blood": "II (A) Rh+", "allergy": "нет",
        "chronic": "нет", "meds": "нет", "habit": "не курит",
        "since": "03.2024", "ref": "самообращение", "cleaning": "09.2026",
    },
    "NK-2042": {
        "phone": "+998 61 224 51 30", "addr": "Нукус, мкр. 4, д. 7",
        "job": "преподаватель",
        "emerg": "брат, +998 61 224 51 31",
        "blood": "I (0) Rh+", "allergy": "лидокаин — сыпь",
        "chronic": "нет", "meds": "нет", "habit": "не курит",
        "since": "11.2023", "ref": "по рекомендации", "cleaning": "05.2026",
    },
    "NK-2043": {
        "phone": "+998 61 229 03 18", "addr": "Нукус, ул. Бердаха, 44",
        "job": "предприниматель",
        "emerg": "супруга, +998 61 229 03 19",
        "blood": "III (B) Rh+", "allergy": "нет",
        "chronic": "гипертония", "meds": "бисопролол",
        "habit": "курит, около 10 сигарет в день",
        "since": "07.2022", "ref": "самообращение", "cleaning": "не было",
    },
    "NK-2044": {
        "phone": "+998 61 227 76 92", "addr": "Нукус, ул. Гарезсизлик, 3",
        "job": "студентка",
        "emerg": "отец, +998 61 227 76 93",
        "blood": "II (A) Rh−", "allergy": "пенициллин",
        "chronic": "нет", "meds": "нет", "habit": "не курит",
        "since": "11.2026", "ref": "направление ортодонта",
        "cleaning": "не было",
    },
    "NK-2045": {
        "phone": "+998 61 221 40 55", "addr": "Нукус, ул. Каракалпакстан, 9",
        "job": "водитель",
        "emerg": "сын, +998 61 221 40 56",
        "blood": "IV (AB) Rh+", "allergy": "нет",
        "chronic": "сахарный диабет 2 типа", "meds": "метформин",
        "habit": "курит",
        "since": "02.2021", "ref": "по рекомендации", "cleaning": "03.2026",
    },
    "DEMO-001": {
        "phone": "—", "addr": "—", "job": "—", "emerg": "—",
        "blood": "—", "allergy": "—", "chronic": "—", "meds": "—",
        "habit": "—",
        "since": "—", "ref": "публичный набор figshare (CC BY 4.0)",
        "cleaning": "—",
    },
}


def demo_profile(p: dict) -> dict | None:
    """Анкета карточки-примера. None — у настоящего пациента её нет.

    Проверяем именно `is_example`: анкета выдуманная, и показать её рядом с
    настоящей фамилией нельзя ни при каких обстоятельствах.
    """
    if not p or not p.get("is_example"):
        return None
    return DEMO_PROFILES.get(p.get("card_no") or "")


def seed_demo_extras() -> int:
    """Телефоны и записи на приём у карточек-ПРИМЕРОВ.

    Отдельно от `seed_demo_patients`, потому что та срабатывает один раз на
    пустой базе, а эти поля появились позже — иначе у уже заведённых
    примеров список остался бы без колонки записи.

    ГАРАНТИЯ БЕЗОПАСНОСТИ ДРУГАЯ, чем у seed_demo_patients: там проверялось
    «в базе нет настоящих пациентов», здесь мы просто НЕ ТРОГАЕМ никого,
    кроме `is_example = 1`. Это надёжнее: у клиники к тому моменту уже
    появились свои карточки, и прежняя проверка заблокировала бы заполнение
    примеров навсегда.

    Время записи считается ОТ СЕГОДНЯШНЕГО ДНЯ и переносится на завтра,
    если час уже прошёл: показ идёт и утром, и вечером, а «запись на
    11:00», которая была три часа назад, в списке не покажется вовсе —
    ближайшей считается только будущая.
    """
    now_t = datetime.now().astimezone()

    def when(day_shift: int, hour: int, minute: int = 0) -> str:
        t = now_t.replace(hour=hour, minute=minute, second=0, microsecond=0)
        t += timedelta(days=day_shift)
        if t <= now_t:                      # час уже прошёл — на завтра
            t += timedelta(days=1)
        return t.isoformat(timespec="seconds")

    # карта -> (телефон, сдвиг дня, час, минута, повод)
    EXTRAS = {
        "NK-2041": ("+998 61 222 14 07", 1, 14, 0, "контроль после разбора КЛКТ"),
        "NK-2042": ("+998 61 224 51 30", 0, 16, 30, "консультация по имплантации"),
        "NK-2043": ("+998 61 229 03 18", 2, 10, 30, "боль в верхней челюсти справа"),
        "NK-2044": ("+998 61 227 76 92", 5, 16, 0, "консультация ортодонта"),
        "NK-2045": ("+998 61 221 48 05", None, 0, 0, ""),
    }

    n = 0
    with connect() as con:
        rows = con.execute("SELECT id, card_no, phone FROM patients"
                           " WHERE is_example = 1").fetchall()
    for r in rows:
        extra = EXTRAS.get(r["card_no"])
        if not extra:
            continue
        phone, shift, hour, minute, reason = extra
        if phone and not (r["phone"] or ""):
            update_patient(r["id"], phone=phone)
            n += 1
        if shift is None:
            continue
        # Повторный запуск не плодит записи: если будущая уже есть, хватит.
        if next_appointments([r["id"]]).get(r["id"]):
            continue
        add_appointment(r["id"], when(shift, hour, minute), reason)
        n += 1

    # Пациент БЕЗ СНИМКОВ: он и есть срез «новые» — записан, но ещё не
    # снимался. Без такой карточки фильтр «Новые» на показе всегда пуст, а
    # пустой фильтр читается как поломка.
    if rows and not any(r["card_no"] == "NK-2046" for r in rows):
        pid = create_patient("Курбанова", "Шахноза", birth_year=1991, sex="Ж",
                             card_no="NK-2046", is_example=True,
                             note="пример для демонстрации, не пациент клиники")
        update_patient(pid, phone="+998 61 223 90 44")
        add_appointment(pid, when(0, 11, 0), "первичный приём, направлена на КЛКТ")
        n += 1
    return n


def seed_demo_patients() -> int:
    """Список-пример для показа. Возвращает, сколько карточек завела.

    СРАБАТЫВАЕТ ТОЛЬКО НА БАЗЕ БЕЗ НАСТОЯЩИХ ПАЦИЕНТОВ. У клиники свои
    карточки, и подкладывать им выдуманных людей нельзя ни при каких
    обстоятельствах — даже с пометкой «пример».
    """
    with connect() as con:
        real = con.execute(
            "SELECT 1 FROM patients WHERE is_example = 0 LIMIT 1").fetchone()
        if real:
            return 0
        if con.execute("SELECT 1 FROM patients WHERE card_no LIKE 'NK-20%'"
                       " LIMIT 1").fetchone():
            return 0                       # уже заводили

    def _backdate(sid: str, days: int) -> None:
        """Отодвинуть исследование на `days` дней назад.

        Список сортируется по времени последнего снимка. Заведённые подряд
        записи получают одну и ту же секунду, порядок при равенстве
        произвольный — показательный кейс оказывался в середине списка.
        Заодно это правдоподобнее: пять снимков в одну секунду в клинике
        не делают.

        Время считаем в Python, а не функцией `datetime()` в SQL: она
        отбрасывает часовой пояс и пишет «2026-11-16 21:45:44», тогда как
        все остальные записи лежат в ISO со смещением. Два формата в одной
        колонке — это разбор дат, который где-нибудь да сломается.
        """
        t = (datetime.now(timezone.utc) - timedelta(days=days))
        iso = t.astimezone().isoformat(timespec="seconds")
        with connect() as con:
            con.execute("UPDATE studies SET uploaded_at = ?, taken_at = ?"
                        " WHERE id = ?", (iso, iso, sid))

    n = 0
    for i, (last, first, year, sex, card, kind, cmp_) in enumerate(DEMO_ROSTER):
        pid = create_patient(last, first, birth_year=year, sex=sex,
                             card_no=card, is_example=True,
                             note="пример для демонстрации, не пациент клиники")
        sid = create_study(pid, kind, status="uploaded", complaints=cmp_)
        _backdate(sid, len(DEMO_ROSTER) - i)
        n += 1

    last, first, year, sex, card, case_id, cmp_ = SHOWCASE
    pid = create_patient(last, first, birth_year=year, sex=sex, card_no=card,
                         is_example=True,
                         note="пример для демонстрации, не пациент клиники")
    create_study(pid, "cbct", status="ready", legacy_case=case_id,
                 legacy_kind="cbct", complaints=cmp_)
    return n + 1


# -------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="список пациентов")
    ap.add_argument("--find", default="", help="поиск по фамилии или карте")
    ap.add_argument("--add", nargs="+", metavar=("ФАМИЛИЯ", "ИМЯ"),
                    help="завести пациента")
    ap.add_argument("--year", type=int, help="год рождения")
    ap.add_argument("--card", default="", help="номер карты клиники")
    ap.add_argument("--migrate", action="store_true",
                    help="перенести демо-кейсы в пациентов-примеров")
    ap.add_argument("--show", metavar="ID", help="карточка пациента")
    a = ap.parse_args()

    print(f"хранилище: {DATA_DIR}")
    if a.migrate:
        r = migrate_examples()
        print(f"перенесено: пациентов {r['patients']}, исследований "
              f"{r['studies']}, пропущено (уже были) {r['skipped']}")
    if a.add:
        pid = create_patient(a.add[0], a.add[1] if len(a.add) > 1 else "",
                             birth_year=a.year, card_no=a.card)
        print(f"заведён пациент {pid}")
    if a.show:
        p = get_patient(a.show)
        if not p:
            raise SystemExit(f"нет пациента {a.show}")
        print(f"{p['display']}  возраст: {p['age'] or '—'}  карта: {p['card_no'] or '—'}")
        for s in list_studies(a.show):
            print(f"   {s['kind_label']:<28} {s['status']:<11} "
                  f"{s['uploaded_at'][:16]}  {s['legacy_case'] or s['id']}")
    if a.list or a.find or not any((a.add, a.migrate, a.show)):
        rows = list_patients(a.find)
        print(f"\n{'id':<14}{'пациент':<26}{'возр':>5}{'исслед':>8}  последнее")
        for p in rows:
            print(f"{p['id']:<14}{p['display'][:25]:<26}{p['age'] or '—':>5}"
                  f"{p['studies_count']:>8}  {(p['last_study_at'] or '')[:16]}")
        print(f"всего: {len(rows)}")


if __name__ == "__main__":
    main()
