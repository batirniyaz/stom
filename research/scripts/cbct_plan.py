"""Сборка плана лечения из находок: находки -> услуги -> этапы.

Между находкой («зуб 46 отсутствует») и строкой плана («Установка дентального
имплантата ×3, зубы 36, 37, 46») лежит агрегация: одинаковые находки
схлопываются в услугу с количеством и списком зубов. Именно в таком виде
клиника выдаёт план пациенту.

Порядок этапов взят из реального плана клиники (План_лечения_..._v5.pptx):
    1 Терапия (санация, гигиена внутри)   1-2 месяца
    2 Хирургия и имплантация              ПАРАЛЛЕЛЬНО этапу 1
    3 Ортодонтия                          16-20 месяцев
    4 Ортопедия                           финальный этап
То есть гигиена не отдельный этап, а часть терапии, и хирургия идёт не после
терапии, а вместе с ней. Стоимости и фотопротокол сюда не входят — их считает
отдельный сервис клиники.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_plan.py client2
    .venv/Scripts/python.exe research/scripts/cbct_plan.py client4 --model pretrained --json plan.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_findings as cf  # noqa: E402
import clinic_routes as cr  # noqa: E402

# Этапы плана: ключ -> (номер, название, срок/примечание)
STAGES = {
    "therapy":  (1, "Терапия", "санация полости рта · 1–2 месяца"),
    "surgery":  (2, "Хирургия и имплантация", "параллельно этапу 1"),
    "ortho":    (3, "Ортодонтия", "исправление прикуса · 16–20 месяцев"),
    "prostho":  (4, "Ортопедия", "реставрация · финальный этап"),
}

# Куда попадает находка каждой роли. Гнатолог отдельным этапом не идёт —
# анализ окклюзии ведёт к ортодонтии; гигиена, как в плане клиники, внутри
# терапии. Роль review в план не входит вовсе, она уходит в ручную проверку.
ROUTE_STAGE = {
    "urgent": "therapy", "hygiene": "therapy", "therapy": "therapy",
    "surgery": "surgery", "gnatho": "ortho", "ortho": "ortho",
    "prostho": "prostho",
}

WISDOM = {18, 28, 38, 48}       # восьмёрки не имплантируют
CANINES = {13, 23, 33, 43}      # ретенированный клык выводят, а не удаляют
# Протяжённый дефект: подряд отсутствующих зубов, при котором ортопед скорее
# предложит мост или съёмный протез, чем цепочку имплантов.
SPAN_DEFECT = 4


def _proc(name, teeth, stage, why, confidence="high", note=None):
    teeth = sorted(set(teeth))   # одна реставрация может дать несколько находок
    return {"name": name, "teeth": teeth, "count": len(teeth),
            "stage": stage, "why": why, "confidence": confidence, "note": note}


def _runs(teeth: list[int]) -> list[list[int]]:
    """Разбивает отсутствующие зубы на подряд идущие группы внутри квадранта."""
    out, cur = [], []
    for t in sorted(teeth):
        if cur and t == cur[-1] + 1 and t // 10 == cur[-1] // 10:
            cur.append(t)
        else:
            if cur:
                out.append(cur)
            cur = [t]
    if cur:
        out.append(cur)
    return out


def _ios_procedures(findings: list, review: list) -> list:
    """Находки по интраоральному скану -> позиции плана.

    Скан закрывает то, чего на КЛКТ не видно: смыкание, прикус, положение
    зубов в ряду. Именно он наполняет третий этап (ортодонтию), который по
    одной томограмме оставался пустым.

    Собираем по смыслу, а не по одной находке на строку: врачу нужна услуга
    с основанием, а не перечисление измерений.
    """
    by_code = {}
    for f in findings:
        by_code.setdefault(f["code"], []).append(f)
    procs = []

    # --- ортодонтия: прикус и положение зубов ---
    ortho_reasons, ortho_teeth, weak = [], set(), False
    for f in by_code.get("occlusion", []):
        ortho_reasons.append(f["text"])
        ortho_teeth.update(f.get("teeth") or [])
        weak = weak or f.get("confidence") == "low"
    for f in by_code.get("off_arch", []):
        ortho_reasons.append(f["text"])
        ortho_teeth.update(f.get("teeth") or [])
    if ortho_reasons:
        procs.append(_proc(
            "Ортодонтическое лечение (исправление прикуса)",
            sorted(ortho_teeth), "ortho", "; ".join(ortho_reasons),
            confidence="low" if weak else "medium",
            note=("часть измерений прикуса разделяет диагноз врача слабо — "
                  "см. основание" if weak else None)))

    # --- гнатолог: как распределена жевательная нагрузка ---
    gnatho = [f for f in findings if f["route"] == "gnatho"]
    if gnatho:
        procs.append(_proc(
            "Консультация гнатолога (анализ окклюзии)",
            sorted({t for f in gnatho for t in (f.get("teeth") or [])}),
            "ortho", "; ".join(f["text"] for f in gnatho),
            confidence="low",
            note="проверяется копиркой на приёме за минуту"))

    # --- терапия: состояние десны ---
    perio = [f for f in findings if f["route"] == "hygiene"]
    if perio:
        procs.append(_proc(
            "Осмотр пародонтолога (рецессия десны)",
            sorted({t for f in perio for t in (f.get("teeth") or [])}),
            "therapy", "; ".join(f["text"] for f in perio),
            confidence="low"))

    # --- ортопедия: дефекты ряда и зубы без пары ---
    for code, name in (("defect", "Восстановление дефекта зубного ряда"),
                       ("defect_free_end", "Концевой дефект: имплантация "
                                           "или съёмный протез"),
                       ("no_antagonist", "Оценка зубов без антагониста")):
        items = by_code.get(code, [])
        if items:
            procs.append(_proc(
                name, sorted({t for f in items for t in (f.get("teeth") or [])}),
                "prostho", "; ".join(f["text"] for f in items),
                confidence="medium"))

    for f in findings:
        if f["route"] == "review":
            review.append({"code": f["code"], "text": f["text"],
                           "severity": f.get("severity", "info"),
                           "confidence": f.get("confidence", "medium")})
    return procs


# Насколько наборы зубов на КЛКТ и на скане могут расходиться, оставаясь
# одним пациентом. Расхождение в один-два зуба объяснимо: разное время съёмки,
# зуб удалён между исследованиями, восьмёрка не попала в поле зрения. Больше —
# это почти наверняка разные люди или устаревшее исследование.
PATIENT_MISMATCH_TEETH = 3


def _patient_match(cbct_teeth, ios_teeth) -> dict | None:
    """Сверка наборов зубов двух исследований. -> запись для ручной проверки.

    Пациентской сущности, связывающей кейс КЛКТ и кейс скана, пока нет: оба
    идентификатора врач указывает руками. Значит перепутать их легко, а план
    при этом соберётся молча и будет выглядеть правдоподобно. Дешёвая защита —
    сверить, об одних ли зубах идёт речь.
    """
    if not cbct_teeth or not ios_teeth:
        return None
    a, b = set(cbct_teeth), set(ios_teeth)
    only_cbct, only_ios = sorted(a - b), sorted(b - a)
    diff = len(only_cbct) + len(only_ios)
    if diff < PATIENT_MISMATCH_TEETH:
        return None
    return {
        "code": "patient_mismatch",
        "severity": "alert",
        "confidence": "high",
        "text": (f"КЛКТ и скан расходятся по составу зубов на {diff} позиций: "
                 f"только на КЛКТ {only_cbct or '—'}, только на скане "
                 f"{only_ios or '—'}. Это разные пациенты, либо одно из "
                 f"исследований устарело. План собран из обоих — не показывать, "
                 f"пока не проверено"),
    }


def build_plan(res: dict | None = None, ios: list | None = None,
               ios_teeth: list | None = None) -> dict:
    """res — результат cbct_findings.analyze_case; ios — находки по скану.

    Любой из источников может отсутствовать: план собирается из того, что есть.
    """
    if res is None:
        res = {"case_id": None, "model_label": "—", "findings": [], "quality": []}
    f_by_code: dict[str, list] = {}
    for f in res["findings"]:
        f_by_code.setdefault(f["code"], []).append(f)

    procs: list[dict] = []
    review: list[dict] = list(res.get("quality") or [])
    mismatch = _patient_match(res.get("teeth_present"), ios_teeth)
    if mismatch:
        review.insert(0, mismatch)
    if ios:
        procs += _ios_procedures(ios, review)

    # ---- ретенция: тактика зависит от того, какой это зуб ----
    impacted = [f["tooth"] for f in f_by_code.get("tooth_impacted", [])]
    wisdom = [t for t in impacted if t in WISDOM]
    canines = [t for t in impacted if t in CANINES]
    unusual = [t for t in impacted if t not in WISDOM and t not in CANINES]
    if wisdom:
        procs.append(_proc(
            "Удаление зуба мудрости", wisdom, "surgery",
            "коронка зуба закрыта костью — подозрение на ретенцию",
            confidence="medium"))
    if canines:
        # ретенированный клык обычно выводят в дугу ортодонтически,
        # а не удаляют — поэтому это этап 3, а не хирургическое удаление
        procs.append(_proc(
            "Ретенированный клык: вывод в зубную дугу (ортодонт + хирург)",
            canines, "ortho",
            "коронка клыка закрыта костью", confidence="medium"))
    if unusual:
        # премоляры и вторые моляры ретенируются редко; куда вероятнее,
        # что зуб просто плохо сегментирован — в план такое не пускаем
        review.append({
            "code": "impacted_unusual",
            "text": f"Подозрение на ретенцию в нетипичной позиции: зубы "
                    f"{sorted(unusual)}. Чаще это ошибка сегментации, "
                    f"а не ретенция — проверить по срезам",
            "severity": "warning", "confidence": "low"})

    # ---- риск повреждения нерва при вмешательстве ----
    canal = [f for f in f_by_code.get("canal_contact", [])
             + f_by_code.get("canal_close", [])]
    if canal:
        low = [f["tooth"] for f in canal if f["confidence"] != "high"]
        procs.append(_proc(
            "Планирование вмешательства по КЛКТ (близость нижнечелюстного канала)",
            [f["tooth"] for f in canal], "surgery",
            "корень ближе 2 мм к каналу — риск повреждения n. alveolaris inferior",
            confidence="low" if low else "high",
            note=(f"по зубам {low} маска канала ненадёжна, перепроверить"
                  if low else None)))

    # ---- имплантация / протяжённые дефекты ----
    missing = [f["tooth"] for f in f_by_code.get("tooth_missing", [])]
    implant_candidates = [t for t in missing if t not in WISDOM]
    bone = {f["tooth"]: f["value"] for f in f_by_code.get("tooth_missing", [])
            if f["value"] is not None}

    spans, singles = [], []
    for run in _runs(implant_candidates):
        (spans if len(run) >= SPAN_DEFECT else singles).append(run)

    if singles:
        teeth = [t for run in singles for t in run]
        weak = [t for t in teeth
                if bone.get(t) is not None and cf.misch_class(bone[t]) == "D4"]
        procs.append(_proc(
            "Установка дентального имплантата", teeth, "surgery",
            "зуб отсутствует, дефект в пределах 1–3 зубов",
            confidence="medium",
            note=(f"по зубам {weak} кость ориентировочно D4 — вероятна костная "
                  f"пластика" if weak else None)))
    for run in spans:
        procs.append(_proc(
            "Консультация ортопеда: протяжённый дефект "
            "(мост / съёмный протез / имплантация)", run, "prostho",
            f"подряд отсутствует {len(run)} зуба(ов) — цепочка имплантов "
            f"не единственный вариант", confidence="medium"))

    # ---- синус-лифтинг ----
    # Берём только ВЫРАЖЕННУЮ близость (severity warning): просто близкий
    # корень — это норма, две трети зубов лежат ближе 2 мм от пазухи, и
    # выносить их все в план значит утопить в нём настоящие показания.
    sinus = [f["tooth"] for f in f_by_code.get("sinus_contact", [])
             + f_by_code.get("sinus_close", [])
             if f.get("severity") in ("alert", "warning")]
    # значим для имплантации: боковой отдел верхней челюсти
    sinus_sites = sorted({t for t in sinus} |
                         {t for t in implant_candidates
                          if t // 10 in (1, 2) and t % 10 >= 4})
    if sinus_sites:
        procs.append(_proc(
            "Оценка необходимости синус-лифтинга", sinus_sites, "surgery",
            "имплантация в боковом отделе верхней челюсти либо корень у "
            "дна пазухи ближе, чем у 97% зубов референсной выборки",
            confidence="medium"))

    # ---- пародонт: убыль костной ткани ----
    # Тяжёлая и умеренная убыль ведут к хирургическому этапу пародонтологии,
    # лёгкая закрывается гигиеной внутри терапии. Разделяем, потому что это
    # разные по объёму и стоимости вмешательства.
    bone_loss = f_by_code.get("bone_loss", [])
    # Степени названы от ВЫБОРКИ, а не от клинической шкалы: см. cbct_perio,
    # там объяснено, почему миллиметры из литературы к нашему измерению
    # неприменимы (медиана 3.9 мм против литературной нормы 2 мм).
    heavy = [f for f in bone_loss
             if f.get("grade") in ("резко выше нормы", "заметно выше нормы")]
    light = [f for f in bone_loss if f.get("grade") == "выше нормы"]
    def _bone_why(items):
        # Полный перечень участков (у client4 их 22) в строку плана не влезает
        # и не читается. План — это услуга с количеством; поучастковые
        # миллиметры уже лежат в находках, врач видит их в карточке зуба.
        parts = []
        for f in items:
            teeth = f.get("teeth") or []
            parts.append(f"{f['grade']} до {f['value']:.1f} мм "
                         f"({len(teeth)} зуб.: {', '.join(map(str, teeth))})")
        return "; ".join(parts)

    if heavy:
        procs.append(_proc(
            "Осмотр пародонтолога: уровень кости заметно ниже обычного",
            [t for f in heavy for t in (f.get("teeth") or [])], "therapy",
            _bone_why(heavy), confidence="medium",
            note="измерено по КЛКТ; степень пародонтита по томограмме мы "
                 "не ставим — нужна оценка пародонтолога"))
    if light:
        procs.append(_proc(
            "Профессиональная гигиена и наблюдение",
            [t for f in light for t in (f.get("teeth") or [])], "therapy",
            _bone_why(light), confidence="medium"))

    # ---- эндодонтия: ранее пролеченные каналы ----
    # Пломба в канале — это ФАКТ, а не показание. Периапикальные очаги мы на
    # КЛКТ не детектируем, поэтому сказать «нужно перелечивание» нечем.
    # Ставим оценочную позицию, как для реставраций: «посмотрите на эти зубы».
    endo = f_by_code.get("endo_filled", [])
    if endo:
        endo_teeth = [t for f in endo for t in (f.get("teeth") or [])]
        procs.append(_proc(
            "Оценка ранее пролеченных каналов",
            endo_teeth, "therapy",
            "плотность внутри канала выше дентина того же зуба — канал "
            "запломбирован; качество пломбирования по КЛКТ не оценивается",
            confidence="medium",
            note="депульпированный зуб хрупче — учесть при протезировании"))

    # ---- ортопедия: существующие реставрации ----
    rest = [f["tooth"] for f in f_by_code.get("restoration", [])]
    if rest:
        procs.append(_proc(
            "Оценка состояния имеющихся реставраций", rest, "prostho",
            "на снимке видны коронки / мосты / импланты", confidence="medium"))

    # ---- сборка по этапам ----
    stages = []
    for key, (num, label, note) in sorted(STAGES.items(), key=lambda kv: kv[1][0]):
        items = [p for p in procs if p["stage"] == key]
        stages.append({"num": num, "key": key, "label": label,
                       "note": note, "procedures": items})

    # Что закрыл скан, то из списка «не определяется» убираем: врач не должен
    # читать, что прикус нужен со скана, когда скан уже посчитан.
    covered = {"Прикус"} if ios else set()
    return {
        "case_id": res.get("case_id"),
        "model_label": res.get("model_label", "—"),
        "sources": [s for s, on in (("КЛКТ", bool(res.get("findings"))),
                                    ("интраоральный скан", bool(ios))) if on],
        "draft": True,
        "patient_mismatch": bool(mismatch),
        "stages": stages,
        "review": review,
        "not_determined": [(what, need) for what, need in NOT_DETERMINED
                           if not any(what.startswith(c) for c in covered)],
    }


# Чего в плане клиники нет и не может быть из одной КЛКТ. Пишем явно, чтобы
# врач видел границу: заголовочный диагноз их плана («Class II, Deep bite,
# дВНЧС») строится по прикусу и суставу, а не по томограмме.
NOT_DETERMINED = [
    ("Прикус: класс по Энглю, глубина перекрытия, перекрёстный прикус",
     "нужен интраоральный скан (Bits2Bites / Bite2Text)"),
    ("Дисфункция ВНЧС", "нужна аксиография (ProAxis / P-Art)"),
    ("Кариес и его глубина", "берётся с 2D ОПТГ; на КЛКТ не видно"),
    ("Качество пломбировки каналов, показания к перелечиванию",
     "пульпа и каналы сегментируются моделью ToothFairy3, но оценка их "
     "состояния не сделана: нужны критерии от терапевта клиники"),

]


def print_plan(plan: dict) -> None:
    print(f"\n{'='*74}")
    src = ", ".join(plan.get("sources") or ["нет данных"])
    print(f"ПЛАН ЛЕЧЕНИЯ (черновик) — кейс {plan['case_id']} / "
          f"{plan['model_label']}")
    print(f"источники: {src}")
    print(f"{'='*74}")
    total = 0
    for st in plan["stages"]:
        procs = st["procedures"]
        head = f"\nЭТАП {st['num']} · {st['label']}  ({st['note']})"
        print(head)
        if not procs:
            print("   по данным КЛКТ показаний не выявлено")
            continue
        for p in procs:
            total += p["count"]
            conf = "" if p["confidence"] == "high" else f"  [доверие: {p['confidence']}]"
            print(f"   {p['name']}")
            if p["teeth"]:
                teeth = ", ".join(str(t) for t in p["teeth"])
                print(f"      ×{p['count']}   зубы: {teeth}{conf}")
            else:
                # услуга на весь прикус целиком, а не на отдельные зубы
                print(f"      на всю челюсть{conf}")
            print(f"      основание: {p['why']}")
            if p["note"]:
                print(f"      ! {p['note']}")
    print(f"\nВсего позиций по зубам: {total}")

    if plan.get("patient_mismatch"):
        print(f"\n{'!'*74}")
        print("ВНИМАНИЕ: КЛКТ и скан, похоже, от РАЗНЫХ пациентов — см. ниже")
        print(f"{'!'*74}")
    if plan["review"]:
        print(f"\n{'-'*74}\nТРЕБУЕТ РУЧНОЙ ПРОВЕРКИ ПЕРЕД ПОКАЗОМ ПАЦИЕНТУ")
        for q in plan["review"]:
            print(f"   • {q['text']}")

    print(f"\n{'-'*74}\nПО КЛКТ НЕ ОПРЕДЕЛЯЕТСЯ (нужны другие данные)")
    for what, need in plan["not_determined"]:
        print(f"   • {what}\n       {need}")
    print(f"\n{'-'*74}")
    print("ЧЕРНОВИК. Пороги и правила не подтверждены врачом; не является планом лечения.")


def load_ios(case_id: str) -> tuple:
    """Находки по скану из кейса -> (находки, список зубов на скане)."""
    root = (Path(__file__).resolve().parents[2] / "research_ios" / "web_cases"
            / case_id)
    teeth: list = []
    out, seen = [], set()
    for name in ("findings.json", "occlusion.json"):
        f = root / name
        if not f.is_file():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for row in (d.get("teeth") or []):
            if isinstance(row, dict) and row.get("fdi") and row["fdi"] not in teeth:
                teeth.append(row["fdi"])
        for item in (d.get("findings") or []) + (d.get("quality") or []):
            # один и тот же факт приходит из обоих файлов (например «нет
            # антагониста» — и из геометрии, и из карты контактов)
            key = (item["code"], tuple(item.get("teeth") or []))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out, sorted(teeth)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", nargs="?", default=None,
                    help="кейс КЛКТ; можно опустить, если считаем только скан")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ios", default=None, help="кейс интраорального скана")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.case_id and not args.ios:
        raise SystemExit("нужен хотя бы один источник: кейс КЛКТ или --ios")

    res = cf.analyze(args.case_id, args.model) if args.case_id else None
    ios, ios_teeth = load_ios(args.ios) if args.ios else (None, None)
    plan = build_plan(res, ios, ios_teeth)
    print_plan(plan)
    if args.json:
        Path(args.json).write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"сохранено: {args.json}")


if __name__ == "__main__":
    main()
