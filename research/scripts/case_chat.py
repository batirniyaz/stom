"""Чат по кейсу КЛКТ: отвечает ТОЛЬКО по данным этого пациента.

Зачем. Врач на приёме читает ленту находок и держит в голове, что значит
«расстояние корень → канал 1.6 мм при пороге 2.0». Пациент этого не держит, да
и врачу быстрее спросить словами. Чат закрывает этот разрыв: он пересказывает
уже посчитанное, и ничего кроме.

ГЛАВНОЕ РЕШЕНИЕ: модель здесь НЕ ставит диагноз и не добавляет фактов. Весь
продукт держится на проверяемости — у каждой находки есть основание: что
измерено, сколько, при каком пороге, при каком вокселе. Текст языковой модели
звучит уверенно и не проверяем ничем. Поэтому:

  * в контекст уходят ТОЛЬКО находки, зубы и план ЭТОГО кейса;
  * ответ обязан назвать находки, на которые опирается, и эти ключи
    проверяются по контексту — несуществующие отбрасываются;
  * если после проверки не осталось ни одной опоры, показывается не ответ
    модели, а прямое «в данных этого пациента такого нет»;
  * ЧИСЛА С ЕДИНИЦАМИ сверяются с контекстом отдельно. Это место, где
    выдумка опаснее всего: «1.6 мм» и «6.1 мм» выглядят одинаково
    убедительно, а решение по ним разное.

Чего чат не делает намеренно: не прогнозирует, не назначает лечение, не
отвечает на общие медицинские вопросы. Любое из этого превратило бы систему
измерений в систему, которая «сама диагностирует», и мы потеряли бы
единственное, что у нас есть, — доказуемость каждого утверждения.

Обезличивание переиспользуется из `plan_agent`: ни ФИО, ни даты рождения, ни
номера карты, ни снимков наружу не уходит. Проверяется явно, а не на доверии.

Провайдер. Предпочтение — Anthropic Haiku (быстрый и дешёвый, а задача здесь
простая: пересказать данные). Если ключа Anthropic нет, работает через Gemini
тем ключом, который уже лежит в `data/secrets.env`. Ключ в код не вписывается
никогда.

Запуск:

    .venv/Scripts/python.exe research/scripts/case_chat.py client4 \\
        --model sp --q "почему у 36 предупреждение"

    # что именно уйдёт наружу, без обращения к модели:
    .venv/Scripts/python.exe research/scripts/case_chat.py client4 --dry-run
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cbct_backend as cb      # noqa: E402
import plan_agent              # noqa: E402

TIMEOUT = 60
MAX_QUESTION = 500

ANTHROPIC_MODEL = "claude-haiku-5-5"
# Список с запасом: провайдер снимает модели без предупреждения, и ответ
# «HTTP 404, модель больше не доступна» мы уже получали на gemini-2.0/2.5.
# Поэтому перебираем сверху вниз и падаем только если не ответила ни одна.
GEMINI_MODELS = ("gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash")

SYSTEM = """Ты помощник стоматолога в системе анализа КЛКТ. Тебе дают находки
ОДНОГО пациента, уже посчитанные измерениями по снимку, и вопрос.

Правила, которые нельзя нарушать:

1. Отвечай ТОЛЬКО по данным в блоке «данные». Если ответа в них нет, так и
   скажи: «в данных этого пациента такого нет». Не догадывайся и не дополняй
   общими знаниями о стоматологии.
2. Не ставь диагнозов и не делай прогнозов. Находка — это измерение, а не
   болезнь. Говори «измерено столько при пороге таком», а не «у пациента
   пародонтит».
3. Числа бери из данных дословно. Не округляй, не пересчитывай, не выводи
   новых величин.
4. Не назначай лечение. Поле «кому» в находке говорит, к какому специалисту
   это относится, — его пересказать можно.
5. Отвечай коротко и по-русски. Два-четыре предложения, если вопрос не
   требует перечисления.
6. Если спрашивают про другого пациента, про общую медицину или просят
   что-то вне данных — откажись одной фразой и скажи, что знаешь только этот
   случай.

Ответ верни строго как JSON:
{"answer": "текст для человека",
 "used": ["ключи находок, на которые опираешься"],
 "no_data": false}

«used» — это значения поля «ключ» из данных. Если опоры нет, ставь
"no_data": true и пустой «used»."""


def context_for(case_id: str, model: str | None = None) -> dict:
    """Данные ОДНОГО кейса в том виде, в каком они уходят модели.

    Отдельная функция, а не сборка внутри запроса: это то, что покидает
    машину, и смотреть на это надо глазами (`--dry-run`).
    """
    c = cb.get_case(case_id)
    model = model or c.get("model") or cb.DEFAULT_MODEL
    res = cb.findings_for(c, model)

    findings = []
    for f in (res.get("findings") or []):
        teeth = f.get("teeth") or ([f["tooth"]] if f.get("tooth") else [])
        findings.append({
            "ключ": cb.finding_key(f),
            "что": f.get("text"),
            "зубы": teeth,
            "важность": f.get("severity"),
            "достоверность": f.get("confidence"),
            "кому": f.get("action"),
            # Основание — сердце ответа: именно по нему чат объясняет
            # «почему так», а не по собственным представлениям.
            "основание": [
                {"измерено": b.get("label"), "значение": b.get("value"),
                 "единица": b.get("unit"), "порог": b.get("rule"),
                 "оговорка": b.get("note")}
                for b in (f.get("basis") or [])
            ],
        })

    # Раздел ручной проверки идёт отдельно и помечен: это сомнения в маске, а
    # не находки у пациента. Смешать их означало бы пугать человека нашими
    # же артефактами сегментации.
    quality = [{"ключ": cb.finding_key(q), "что": q.get("text")}
               for q in (res.get("quality") or [])]

    return {
        "кейс": case_id,
        "модель_сегментации": res.get("model_label") or model,
        "зубы_на_снимке": res.get("teeth_present") or [],
        "зубы_отсутствуют": res.get("teeth_missing") or [],
        "находки": findings,
        "требует_ручной_проверки": quality,
        "черновик": bool(res.get("draft", True)),
    }


def _read_key(name: str) -> str | None:
    """Ключ: сначала окружение, затем файл вне гита. В код — никогда."""
    key = os.environ.get(name)
    if key:
        return key.strip()
    import patients as pat
    f = pat.DATA_DIR / "secrets.env"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _post(url: str, body: dict, headers: dict) -> dict:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=raw, headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def ask_llm(question: str, ctx: dict) -> tuple[dict, str]:
    """Спросить модель. Возвращает разобранный ответ и имя провайдера.

    Anthropic предпочтительнее: задача простая (пересказать данные), а Haiku
    быстрый и дешёвый. Gemini оставлен рабочим путём, потому что его ключ в
    проекте уже есть, и без ключа Anthropic чат не должен переставать
    работать.
    """
    user = ("данные:\n" + json.dumps(ctx, ensure_ascii=False, indent=1)
            + "\n\nвопрос: " + question)

    akey = _read_key("ANTHROPIC_API_KEY")
    if akey:
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"model": ANTHROPIC_MODEL, "max_tokens": 700, "temperature": 0.2,
             "system": SYSTEM,
             "messages": [{"role": "user", "content": user}]},
            {"content-type": "application/json", "x-api-key": akey,
             "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return _loads(text), ANTHROPIC_MODEL

    gkey = _read_key("GEMINI_API_KEY") or _read_key("STOMADENT_LLM_KEY")
    if not gkey:
        raise RuntimeError(
            "нет ключа. Положите ANTHROPIC_API_KEY в data/secrets.env "
            "(файл в .gitignore) либо оставьте GEMINI_API_KEY.")
    problems = []
    for m in GEMINI_MODELS:
        try:
            data = _post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{m}:generateContent?key={gkey}",
                {"system_instruction": {"parts": [{"text": SYSTEM}]},
                 "contents": [{"role": "user", "parts": [{"text": user}]}],
                 "generationConfig": {"temperature": 0.2,
                                      "response_mime_type": "application/json"}},
                {"Content-Type": "application/json"})
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return _loads(text), m
        except urllib.error.HTTPError as exc:
            # Сообщение провайдера обязательно: «HTTPError» не говорит ничего,
            # а причины разные и лечатся по-разному — просроченный ключ,
            # исчерпанная квота, непринятая форма запроса.
            try:
                why = json.loads(exc.read().decode()).get(
                    "error", {}).get("message", "")[:160]
            except Exception:                       # noqa: BLE001
                why = ""
            problems.append(f"{m}: HTTP {exc.code} {why}".strip())
        except (KeyError, IndexError) as exc:
            problems.append(f"{m}: неожиданная форма ответа ({exc})")
    raise RuntimeError("ни одна модель не ответила — " + "; ".join(problems))


def _loads(text: str) -> dict:
    """Разобрать JSON, даже если модель обернула его в ```json."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # Модель ответила прозой. Это не ошибка связи, а несоблюдение формата:
        # считаем ответ неподкреплённым, и его отбросит проверка ниже.
        return {"answer": t, "used": [], "no_data": False}


# Число с единицей: именно здесь выдумка опаснее всего. «1.6 мм» и «6.1 мм»
# выглядят одинаково убедительно, а решение по ним разное.
NUM_UNIT = re.compile(r"(\d+[.,]?\d*)\s*(мм|%|HU|см3|см³)", re.I)


def _numbers_in(obj) -> set[str]:
    out = set()
    for m in NUM_UNIT.finditer(json.dumps(obj, ensure_ascii=False)):
        out.add(m.group(1).replace(",", "."))
    # Значения в основании лежат числами без единицы в той же строке —
    # собираем и их, иначе честное «1.56» сочтётся выдуманным.
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("значение", "value") and isinstance(v, (int, float)):
                    out.add(f"{float(v):g}")
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return out


def _unsupported_numbers(answer: str, ctx: dict) -> list[str]:
    """Числа с единицами, которых нет в данных.

    Сверка нестрогая: модель законно округляет 1.56 до 1.6, и запрещать это
    значило бы требовать от ответа машинного вида. Поэтому допуск 0.1 по
    абсолютной величине и 2% по относительной.
    """
    known = []
    for s in _numbers_in(ctx):
        try:
            known.append(float(s))
        except ValueError:
            pass
    bad = []
    for m in NUM_UNIT.finditer(answer):
        try:
            v = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        if not any(abs(v - k) <= max(0.1, abs(k) * 0.02) for k in known):
            bad.append(m.group(0))
    return bad


REFUSAL = ("В данных этого пациента такого нет. Я отвечаю только по "
           "находкам, посчитанным по этому снимку.")


def _validate(reply: dict, ctx: dict) -> dict:
    """Отбросить всё, что не подкреплено данными кейса.

    Главная защита модуля, и она механическая. Убедительность формулировки
    роли не играет: опора либо есть в контексте, либо ответ не показывается.
    """
    keys = {f["ключ"] for f in ctx["находки"]}
    keys |= {q["ключ"] for q in ctx["требует_ручной_проверки"]}

    used = [k for k in (reply.get("used") or []) if k in keys]
    dropped = [k for k in (reply.get("used") or []) if k not in keys]
    answer = str(reply.get("answer") or "").strip()

    bad_nums = _unsupported_numbers(answer, ctx)
    # Отказ модели — законный ответ, и опоры он не требует: «этого в данных
    # нет» и не может ни на что опираться.
    refused = bool(reply.get("no_data")) or not answer

    if bad_nums:
        return {"answer": REFUSAL, "used": [], "grounded": False,
                "blocked": f"числа вне данных: {', '.join(bad_nums)}",
                "dropped_keys": dropped}
    if refused:
        return {"answer": answer or REFUSAL, "used": [], "grounded": False,
                "blocked": None, "dropped_keys": dropped}
    if not used:
        return {"answer": REFUSAL, "used": [], "grounded": False,
                "blocked": "ответ не опирался ни на одну находку кейса",
                "dropped_keys": dropped}
    return {"answer": answer, "used": used, "grounded": True,
            "blocked": None, "dropped_keys": dropped}


def chat(case_id: str, question: str, model: str | None = None) -> dict:
    question = (question or "").strip()[:MAX_QUESTION]
    if not question:
        return {"answer": "Задайте вопрос по этому случаю.", "used": [],
                "grounded": False, "blocked": None}
    ctx = context_for(case_id, model)

    leak = plan_agent.check_anonymous(ctx)
    if leak:
        # Лучше отказать, чем отправить. Обезличивание легко сломать правкой,
        # поэтому отказ здесь жёсткий и без обходного пути.
        raise RuntimeError("в контексте опознавательные данные: "
                           + ", ".join(leak))

    reply, provider = ask_llm(question, ctx)
    out = _validate(reply, ctx)
    out["provider"] = provider
    out["case"] = case_id
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("--model", default=None, help="модель сегментации кейса")
    ap.add_argument("--q", default="Что нашли на этом снимке?")
    ap.add_argument("--dry-run", action="store_true",
                    help="показать, что уйдёт наружу, и не обращаться к модели")
    a = ap.parse_args()

    ctx = context_for(a.case, a.model)
    if a.dry_run:
        print(json.dumps(ctx, ensure_ascii=False, indent=1))
        leak = plan_agent.check_anonymous(ctx)
        print("\nобезличено: " + ("НЕТ — " + ", ".join(leak) if leak else "да"))
        print(f"находок: {len(ctx['находки'])}, "
              f"ручная проверка: {len(ctx['требует_ручной_проверки'])}, "
              f"числа в данных: {len(_numbers_in(ctx))}")
        return

    out = chat(a.case, a.q, a.model)
    print(f"\n[{out['provider']}]  опора: "
          + (", ".join(out["used"]) if out["used"] else "нет"))
    if out.get("blocked"):
        print(f"ОТБРОШЕНО: {out['blocked']}")
    print("\n" + out["answer"])


if __name__ == "__main__":
    main()
