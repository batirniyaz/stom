"""Языковой слой над планом лечения: связать, а НЕ решить.

Роль модели здесь узкая, и это главное решение в модуле. Что лечить, решают
правила (`cbct_plan`): они детерминированы, воспроизводимы и опираются на
порядок этапов из реального плана клиники. Модель получает уже принятые
решения и делает то, что коду даётся плохо:

  * излагает план связным текстом для врача и для пациента;
  * ловит ПРОТИВОРЕЧИЯ между источниками — «на панораме кариес, на КЛКТ в том
    же месте реставрация»: это разные исследования об одном зубе, и правилами
    такое не свести;
  * говорит, чего не хватает для решения («для имплантации нужен прицельный
    снимок»), вместо того чтобы додумывать.

Почему модель не решает. Весь продукт держится на проверяемости: каждая
находка несёт основание — что измерено, каким методом, при каком пороге. Текст
модели звучит уверенно и не проверяем ничем, поэтому позиция плана, которой
нет в исходных находках, ОТБРАСЫВАЕТСЯ кодом (`_validate`). Это механическая
защита, а не доверие к модели: ровно по той же причине в проекте нет процента
уверенности на находках.

Что уходит наружу. Только обезличенные клинические строки: номера зубов,
измерения, названия услуг. Ни имени, ни даты рождения, ни номера карты, ни
снимков — их отсекает `_payload`, и это проверяется тестом при запуске с
`--dry-run`. У бесплатных тарифов провайдер обычно оставляет за собой право
учиться на присланном, поэтому обезличивание здесь не формальность.

Ключ берётся из переменной окружения, в код не попадает:

    setx GEMINI_API_KEY "ваш-ключ"        (получить: aistudio.google.com/apikey)

Запуск:
    .venv/Scripts/python.exe research/scripts/plan_agent.py --patient p_xxx
    .venv/Scripts/python.exe research/scripts/plan_agent.py --patient p_xxx --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Провайдер по умолчанию — Gemini: у него есть бесплатный тариф и приличная
# кириллица. Меняется переменной окружения, чтобы не править код.
PROVIDER = os.environ.get("STOMADENT_LLM", "gemini")
# СПИСОК моделей, а не одна: на бесплатном тарифе они то снимаются, то
# отвечают 503 «high demand». Проверено на ключе клиники:
#   gemini-2.0-flash — снята («no longer available»)
#   gemini-3.6-flash, gemini-3.8-flash — 503
#   gemini-3.5-flash, gemini-flash-lite-latest, gemini-2.5-flash-lite — отвечают
# Идём по списку сверху вниз; врач не должен остаться без плана из-за того,
# что конкретную модель переименовали или она занята.
MODELS = [m.strip() for m in os.environ.get(
    "STOMADENT_LLM_MODEL",
    "gemini-3.5-flash,gemini-flash-lite-latest,gemini-2.5-flash-lite"
).split(",") if m.strip()]
TIMEOUT = 60
RETRIES = 2            # попыток на КАЖДУЮ модель, прежде чем взять следующую

SYSTEM = """Ты помогаешь стоматологу оформить план лечения.

ЖЁСТКИЕ ПРАВИЛА:
1. Ты НЕ назначаешь лечение и НЕ добавляешь позиций от себя. Работаешь только
   с тем, что дано в находках и в черновике плана.
2. Каждая позиция плана обязана ссылаться на источник из входных данных.
   Если для позиции нет основания во входных данных — не включай её.
3. Не выдумывай номера зубов. Если в источнике номера нет, так и пиши:
   «номера уточнить на осмотре».
4. Не ставь диагнозов и не называй степени тяжести, которых нет во входных
   данных. Пиши измерения так, как они даны, вместе с их оговорками.
5. Отдельно перечисли ПРОТИВОРЕЧИЯ между исследованиями и то, ЧЕГО НЕ ХВАТАЕТ
   для решения.

Отвечай СТРОГО одним JSON-объектом без markdown:
{
  "summary": "2-3 предложения о состоянии, без диагнозов",
  "stages": [
    {"stage": "therapy|surgery|ortho|prostho",
     "items": [{"text": "что делать", "why": "на основании чего",
                "sources": ["точная строка из входных данных"]}]}
  ],
  "conflicts": [{"text": "в чём противоречие", "sources": ["...", "..."]}],
  "missing": ["чего не хватает для решения"],
  "questions": ["вопрос врачу, если данных недостаточно"]
}"""


def _payload(summary: dict, plan: dict) -> dict:
    """Что отправляем наружу. ФИО, дата рождения и карта сюда не попадают."""
    findings = []
    for f in summary.get("findings", []):
        findings.append({
            "источник": f.get("source_label"),
            "зубы": f.get("teeth") or [],
            "текст": f.get("text"),
            "важность": f.get("severity"),
        })
    stages = []
    for st in plan.get("stages", []):
        procs = [{"услуга": x.get("name"), "зубы": x.get("teeth") or [],
                  "количество": x.get("count"), "основание": x.get("why")}
                 for x in st.get("procedures", [])]
        if procs:
            stages.append({"этап": st.get("label"), "ключ": st.get("key"),
                           "услуги": procs})
    return {"находки": findings, "черновик_плана": stages,
            "источники": plan.get("sources", [])}


PERSONAL_KEYS = ("last_name", "first_name", "middle_name", "birth", "card",
                 "display", "patient_id", "note")


def check_anonymous(data: dict) -> list:
    """Проверка, что в отправляемом нет опознавательных данных.

    Отдельная функция, а не строчка в коде отправки: обезличивание — это то,
    что легко сломать случайной правкой, и проверять его надо явно.
    """
    text = json.dumps(data, ensure_ascii=False).lower()
    bad = [k for k in PERSONAL_KEYS if f'"{k}"' in text]
    # Даты рождения и длинные номера карт: ищем по форме, а не по ключу.
    if re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", text):
        bad.append("дата в формате ГГГГ-ММ-ДД")
    return bad


def _read_key() -> str | None:
    """Ключ: сначала переменная окружения, затем файл вне гита.

    Файл `data/secrets.env` лежит в каталоге данных, который целиком в
    .gitignore, — так ключ не попадёт в репозиторий даже случайным
    `git add`. В код ключ не вписывается никогда.
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("STOMADENT_LLM_KEY")
    if key:
        return key.strip()
    import patients as pat
    f = pat.DATA_DIR / "secrets.env"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def ask(payload: dict) -> dict:
    key = _read_key()
    if not key:
        raise SystemExit(
            "нет ключа: задайте GEMINI_API_KEY или положите его в "
            "data/secrets.env строкой GEMINI_API_KEY=... "
            "(получить: aistudio.google.com/apikey)")
    if PROVIDER != "gemini":
        raise SystemExit(f"провайдер {PROVIDER} пока не поддержан")

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [
            {"text": json.dumps(payload, ensure_ascii=False, indent=1)}]}],
        "generationConfig": {"temperature": 0.2,
                             "response_mime_type": "application/json"},
    }
    raw = json.dumps(body).encode("utf-8")
    problems = []
    data = None
    for model in MODELS:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        for attempt in range(RETRIES):
            req = urllib.request.Request(
                url, data=raw, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                try:
                    msg = json.loads(exc.read().decode()).get(
                        "error", {}).get("message", "")
                except Exception:                  # noqa: BLE001
                    msg = str(exc)
                # Сообщение провайдера извлекли выше — его и показываем.
                # Раньше оно выбрасывалось, и оставалось «HTTP 404» без
                # причины. А причины разные и лечатся по-разному: 404 «модель
                # снята» требует правки списка, 503 «высокая нагрузка» —
                # просто повтора. Один раз это уже стоило неверного вывода,
                # что модуль сломан, хотя он переходил на рабочую модель.
                problems.append(f"{model}: HTTP {exc.code} {msg[:160]}".strip())
                # 429/503 — «сейчас занято», а не отказ: ждём и повторяем,
                # затем переходим к следующей модели.
                if exc.code in (429, 503) and attempt < RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            except Exception as exc:               # noqa: BLE001
                problems.append(f"{model}: {type(exc).__name__}")
                break
        if data is not None:
            break
    if data is None:
        raise SystemExit("ни одна модель не ответила — " + "; ".join(problems))

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise SystemExit(f"неожиданный ответ модели: {exc}")
    return json.loads(text)


def _validate(answer: dict, payload: dict) -> dict:
    """Отбросить всё, что модель не подкрепила входными данными.

    Главная защита модуля. Позиция без источника, найденного во входных
    данных, не показывается врачу вовсе — независимо от того, насколько
    убедительно она сформулирована.
    """
    known = set()
    for f in payload["находки"]:
        known.add((f.get("текст") or "").strip().lower())
    for st in payload["черновик_плана"]:
        for x in st["услуги"]:
            known.add((x.get("услуга") or "").strip().lower())
            known.add((x.get("основание") or "").strip().lower())

    def grounded(srcs) -> bool:
        for s in srcs or []:
            s = str(s).strip().lower()
            if any(s in k or k in s for k in known if k):
                return True
        return False

    dropped = []
    for st in answer.get("stages", []):
        keep = []
        for item in st.get("items", []):
            if grounded(item.get("sources")):
                keep.append(item)
            else:
                dropped.append(item.get("text"))
        st["items"] = keep
    answer["stages"] = [st for st in answer.get("stages", []) if st.get("items")]
    answer["dropped"] = dropped        # показываем, что отбросили и почему
    answer["draft"] = True
    return answer


def build(pid: str, host: str = "http://127.0.0.1:7861",
          dry_run: bool = False) -> dict:
    def get(path):
        with urllib.request.urlopen(host + path, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))

    summary = get(f"/api/patients/{pid}/summary")
    plan = get(f"/api/patients/{pid}/plan")
    payload = _payload(summary, plan)

    leak = check_anonymous(payload)
    if leak:
        raise SystemExit(f"в отправляемых данных найдено личное: {leak}")
    if dry_run:
        return {"payload": payload, "anonymous": True, "sent": False}
    return _validate(ask(payload), payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--patient", required=True)
    ap.add_argument("--host", default="http://127.0.0.1:7861")
    ap.add_argument("--dry-run", action="store_true",
                    help="показать, что ушло бы наружу, и не отправлять")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    res = build(a.patient, a.host, a.dry_run)
    if a.dry_run:
        print("НАРУЖУ УШЛО БЫ (личных данных нет):")
        print(json.dumps(res["payload"], ensure_ascii=False, indent=1)[:2500])
    else:
        print(res.get("summary", ""))
        for st in res.get("stages", []):
            print(f"\n{st.get('stage')}")
            for it in st.get("items", []):
                print(f"  • {it.get('text')}")
                if it.get("why"):
                    print(f"      основание: {it['why']}")
        if res.get("conflicts"):
            print("\nПРОТИВОРЕЧИЯ МЕЖДУ ИССЛЕДОВАНИЯМИ:")
            for c in res["conflicts"]:
                print(f"  • {c.get('text')}")
        if res.get("missing"):
            print("\nЧЕГО НЕ ХВАТАЕТ:")
            for m in res["missing"]:
                print(f"  • {m}")
        if res.get("dropped"):
            print("\nОТБРОШЕНО (нет основания во входных данных):")
            for d in res["dropped"]:
                print(f"  • {d}")
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"\nсохранено: {a.json}")


if __name__ == "__main__":
    main()
