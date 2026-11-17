"""Слой находок по КЛКТ: измеримые факты на зуб из готовой маски сегментации.

Находка — это не диагноз. Это число с порогом, из которого диагноз собирает
врач (или движок правил): расстояние корня до нижнечелюстного канала, близость
к дну пазухи, плотность кости в лунке, реставрации, подозрение на ретенцию.
Новых моделей здесь нет — всё считается геометрией поверх маски nnU-Net.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_findings.py client4
    .venv/Scripts/python.exe research/scripts/cbct_findings.py client4 --model ours --json out.json

ВАЖНО: все пороги ниже — черновик, их подписывает врач клиники. Пока они не
подтверждены, вывод помечается как черновой и врачу как назначение не отдаётся.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402
import clinic_routes as cr  # noqa: E402
import cbct_endo  # noqa: E402
import cbct_perio  # noqa: E402
import cbct_bone_quality as bq  # noqa: E402

# --- пороги (ЧЕРНОВИК, на подпись врачу) ---
CANAL_CONTACT_MM = 0.5    # корень касается канала / входит в него
CANAL_CLOSE_MM = 2.0      # корень близко: удаление и имплантация с риском

# Кортикальная стенка канала — признак СИЛЬНЕЕ расстояния. Метаанализ 18 работ:
# при её отсутствии риск повреждения нерва 15.65%, при наличии 0.39% (отношение
# шансов 9.87). По одному лишь расстоянию обрыв риска лежит около 1 мм, а не 2:
# при 1-2 мм нарушений 0%, при 0-1 мм — 68%. Наш порог 2.0 мм пришёл из
# рекомендации 1990 года под ПЛЁНОЧНУЮ панораму, где 2 мм закладывали на
# искажение снимка, а не на биологию.
#
# Стенка тонкая: 0.68-1.10 мм по 642 томам, то есть 2-3 вокселя при нашем 0.31.
# Поэтому признак считается, но идёт с пониженным доверием и явной оговоркой
# в основании — на пределе разрешения мы можем принять шум за отсутствие кости.
WALL_HU_MIN = 300         # кость вокруг канала: тот же порог, что у HU-гейта
WALL_RING_MM = 1.2        # кольцо вокруг канала, в котором ищем кортикал
WALL_CROP_MM = 8.0       # запас кропа: заведомо больше кольца стенки
WALL_MIN_FRACTION = 0.35  # доля плотных вокселей в кольце со стороны зуба
# Близость корня к пазухе — это НОРМА, а не находка. Замер по эталонным маскам
# ToothFairy3 (341 верхний премоляр и моляр, 42 тома):
#
#   ближе 0.5 мм — 41% зубов      медиана расстояния 0.8 мм
#   ближе 1.0 мм — 55%            75-й перцентиль 3.0 мм
#   ближе 2.0 мм — 66%            90-й перцентиль 6.5 мм
#
# То есть прежние пороги (контакт 0.5, близко 2.0) давали находку у двух третей
# зубов. На client5 это и вышло: 62% верхних моляров, на client4 — 40%.
# Литература подтверждает независимо: контакт корня с дном пазухи есть у 75-78%
# моляров, у части людей корень вообще внутри пазухи.
#
# Поэтому отдаём ПЕРЦЕНТИЛЬ в референсной выборке, как это уже сделано для
# уровня кости, а не тревогу. Границы — 10-й и 3-й перцентиль: настолько
# близко лежит лишь каждый десятый и каждый тридцатый зуб соответственно.
#
# ВАЖНОЕ ОГРАНИЧЕНИЕ. Протрузию корня В пазуху (клинически самое значимое)
# по этим данным измерить НЕЛЬЗЯ: в разметке ToothFairy3 классы взаимоисключающие,
# один воксель не бывает одновременно зубом и пазухой, поэтому пересечений
# ровно 0% при литературных 75%. Это свойство разметки, а не анатомии.
SINUS_REF_PCTL = [3, 10, 25, 50, 75, 90]
SINUS_REF_MM = [0.15, 0.3, 0.3, 0.8, 3.0, 6.5]
SINUS_REF_N_TEETH, SINUS_REF_N_CASES = 341, 42
# Показываем зуб, лежащий ближе медианы выборки. Более строгий порог брать
# НЕЛЬЗЯ: 10-й перцентиль выборки — это 0.3 мм, меньше нашего вокселя (0.31),
# то есть градации в этой зоне неразличимы физически. Поэтому находка
# ИНФОРМАЦИОННАЯ (severity info) и в план не идёт — она отвечает на вопрос
# «сколько там миллиметров», а не «есть ли патология».
SINUS_P_SHOW = 50.0


def sinus_percentile(mm: float) -> float:
    """Какая доля зубов выборки лежит БЛИЖЕ к пазухе, чем этот."""
    return float(np.interp(mm, SINUS_REF_MM, SINUS_REF_PCTL,
                           left=0.0, right=100.0))

# Плотность кости по Misch. На КЛКТ значения НЕ калиброваны как на КТ — это
# grey values, они зависят от аппарата, FOV и артефактов. Поэтому класс
# отдаём как ориентир с пониженным доверием, а не как измерение.
# Пороги Миша оставлены как СПРАВКА и в находки НЕ идут.
#
# Классификация стоит на единицах Хаунсфилда, а КЛКТ их не даёт: после
# перекалибровки ошибка на 13 аппаратах составляет 35..1562 — шире, чем весь
# диапазон D1-D4. Значение плывёт от поля зрения, укладки, металла ВНЕ кадра
# и даже от программы просмотра. Систематический обзор отобрал 3 работы из
# 2710 и заключил, что переводить серые значения в HU нельзя; у Diagnocat
# такой находки нет вовсе.
#
# Вместо класса отдаём долю костной ткани (`cbct_bone_quality`): она
# предсказывает первичную стабильность импланта лучше плотности (R² 0.61
# против 0.39) и переносима между аппаратами (+5% при смене протокола против
# +65% у толщины трабекул).
MISCH = ((1250, "D1"), (850, "D2"), (350, "D3"))

# Доля оболочки КОРОНКИ, упирающейся в кость (см. crown_encased_fraction).
# Замер по client4: 38 — 88%, 18 — 68%, 48 — 51%, дальше обрыв до 25%;
# по client2 максимум 57%. Порог 0.60 ловит две верхние восьмёрки client4.
# На ретенцию проверено НЕ по ground truth, а по разрыву в распределении —
# врач должен подтвердить на этих же кейсах (см. --calibrate).
RETENTION_ENCASED = 0.60

# ВТОРОЙ признак ретенции: зуб не вышел в зубной ряд.
#
# Одной окружённости костью мало, и это показал врач. У client4 не прорезались
# ОБЕ верхние восьмёрки, обе стоят на 10.7-10.8 мм выше уровня ряда, но по
# кости они разошлись: 18 дал 76%, а 28 — всего 27%, потому что часть его
# коронки граничит не с костью, а с полостью. Находка выдавалась только на
# один зуб из двух.
#
# Непрорезавшийся зуб отличается от соседей прежде всего ПОЛОЖЕНИЕМ: его
# коронка не дошла до окклюзионной плоскости. Уровень ряда берём как медиану
# по зубам своей челюсти БЕЗ восьмёрок — сами восьмёрки и есть проверяемые.
#
# Порог откалиброван по РАЗРЫВУ в распределении, а не подобран. Замер по
# 115 зубам четырёх кейсов:
#
#     перцентили: 50%=0.7  75%=2.9  90%=4.9  95%=6.3  97%=10.8  99%=18.9
#
#     выше ряда, мм        зуб
#         29.5             client2 / 28
#         19.4             client2 / 18
#         15.3             client3 / 28
#         10.8             client4 / 18
#         10.7             client4 / 28
#     --- пусто ---
#          6.5             client5 / 27   <- обычный зуб
#          и ниже          остальные 110
#
# Между 10.7 и 6.5 мм нет ни одного зуба, поэтому любой порог от 7 до 10 даёт
# один и тот же результат — ПЯТЬ находок, и все пять это восьмёрки. Берём
# середину зазора. Первая версия стояла на 4 мм и тянула четыре обычных зуба
# (27, 31, 33, 46): разброс высоты у прорезавшихся зубов доходит до 6.3 мм,
# потому что клык длиннее моляра, и медиана по ряду это не выравнивает.
RETENTION_ABOVE_MM = 8.0

MIN_TOOTH_VOX = 100       # меньше — это шум сегментации, а не зуб

# Объём ЗУБА (коронка + корень), ниже которого маска — огрызок, а не зуб.
# Раньше «огрызок» опознавался только глазами: на client4 у зубов 17 и 46
# длина корня вышла 3.6 и 2.7 мм при норме 12.4-12.6 мм, и находки по ним были
# недостоверны, но формально они проходили порог присутствия.
#
# Ориентир взят из работы на 2312 зубах, снятых на КЛКТ 0.3 мм с
# автосегментацией — то есть ровно наш конвейер. Объёмы КОРНЯ по группам:
#   верхний резец 208+-46, клык 288+-61, моляр 433+-77
#   нижний резец  106+-24, клык 246+-53, моляр 405+-79
# Порог поставлен по самой мелкой группе (нижний резец) минус три сигмы: 34 мм3
# на корень. С коронкой полный зуб примерно вдвое больше, поэтому 70 мм3 —
# это заведомо ниже любого настоящего зуба и выше шума.
#
# ВАЖНО: объём на КЛКТ 0.3 мм систематически ЗАВЫШЕН примерно на 28%
# относительно микро-КТ, и порог сегментации влияет на него сильнее, чем тип
# зуба. Поэтому литературные числа взяты как форма распределения, а порог
# намеренно мягкий — он ловит явные огрызки, а не пограничные случаи.
STUB_VOLUME_MM3 = 70.0
PROSTH_MAX_DIST_MM = 25.0  # реставрация дальше этого от зуба = артефакт маски
CANAL_ASYMMETRY = 2.5     # во столько раз объёмы каналов L/R могут отличаться

JAW_CLASSES = (1, 2)
CANAL_LEFT, CANAL_RIGHT = 3, 4
SINUS_CLASSES = (5, 6)
PROSTH_RU = {8: "мост", 9: "коронка", 10: "имплант"}

# Маршрутизация находок общая для КЛКТ и скана — см. clinic_routes.
ROUTE = cr.ROUTES


def fdi_quadrant(tooth: int) -> int:
    return tooth // 10


def is_lower(tooth: int) -> bool:
    return fdi_quadrant(tooth) in (3, 4)


def misch_class(hu: float) -> str:
    """Класс кости по Мишу. НЕ ИСПОЛЬЗУЕТСЯ в находках — см. комментарий к
    MISCH. Оставлено для сравнения при отладке."""
    for thr, name in MISCH:
        if hu > thr:
            return name
    return "D4"


def _finding(code, text, route, severity="info", tooth=None, value=None,
             unit=None, action=None, confidence="high", basis=None):
    return {"code": code, "tooth": tooth, "value": value, "unit": unit,
            "severity": severity, "text": text, "action": action,
            "route": route, "confidence": confidence, "basis": basis or []}


def _basis(label, value, unit, rule, note=None):
    """Одна строка основания находки: ЧТО измерено, СКОЛЬКО и ПРИ КАКОМ ПОРОГЕ.

    Зачем не процент уверенности, как у Diagnocat. Процент — это вероятность,
    а вероятности у нас нет: находки считаются геометрией по маске, разметки
    рентгенолога, на которой её можно было бы откалибровать, в проекте нет
    (её ждём от клиники). Нарисованные «87%» врач не сможет ни проверить, ни
    оспорить — а «1.2 мм при пороге 2.0, воксель 0.31 мм» проверяется прямо
    на срезе. Это ровно то же правило, по которому пародонт отдаёт перцентиль
    выборки, а не стадию пародонтита.
    """
    row = {"label": label, "value": value, "unit": unit, "rule": rule}
    if note:
        row["note"] = note
    return row


def _vox_note(spacing) -> str:
    """Разрешающая способность измерения: меньше вокселя мы не различаем."""
    return f"воксель {float(np.max(spacing)):.2f} мм"


def cortical_wall_fraction(vol, canal_bm, tooth_bm, spacing):
    """Доля плотной кости в кольце вокруг канала СО СТОРОНЫ ЗУБА.

    Клинически это сильнее расстояния: при отсутствии кортикальной стенки риск
    повреждения нерва 15.65% против 0.39% при её наличии. Расстояние же само по
    себе разделяет хуже — при 1-2 мм нарушений не бывает вовсе.

    Считаем только тот сектор кольца, который смотрит на зуб: стенка может
    отсутствовать с одной стороны и быть целой с другой, а хирургу важна именно
    та, через которую пойдёт инструмент.

    Возвращает (доля, число вокселей в кольце). Доля None, если кольцо пустое.
    """
    if not canal_bm.any() or not tooth_bm.any():
        return None, 0

    # Считаем на КРОПЕ вокруг зуба и канала, а не по всему тому. Функция
    # вызывается на каждый нижний зуб, и каждый вызов делал два
    # distance_transform_edt по всему объёму: на кейсе с вокселем 0.2 мм
    # (client3, 174 млн вокселей) слой находок из-за этого считался больше
    # сорока минут, тогда как на обычных кейсах — минуты. Обе величины
    # ЛОКАЛЬНЫ: кольцо стенки толщиной WALL_RING_MM (около 1 мм) и расстояние
    # до зуба внутри него, поэтому запас в WALL_CROP_MM вокруг объединения
    # масок результат не меняет — за его пределами кольца всё равно нет.
    both = canal_bm | tooth_bm
    idx = np.where(both)
    pad = [int(np.ceil(WALL_CROP_MM / float(sp))) for sp in spacing]
    lo = [max(int(i.min()) - p, 0) for i, p in zip(idx, pad)]
    hi = [min(int(i.max()) + p + 1, n) for i, p, n in zip(idx, pad, both.shape)]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    canal_c, tooth_c, vol_c = canal_bm[box], tooth_bm[box], vol[box]

    # Кольцо вокруг канала толщиной WALL_RING_MM
    d_can = ndi.distance_transform_edt(~canal_c, sampling=spacing)
    ring = (d_can > 0) & (d_can <= WALL_RING_MM)
    if not ring.any():
        return None, 0
    # Оставляем сторону зуба: воксель кольца ближе к зубу, чем к
    # противоположной стенке. Грубо — по расстоянию до зуба.
    d_tooth = ndi.distance_transform_edt(~tooth_c, sampling=spacing)
    near = ring & (d_tooth <= np.percentile(d_tooth[ring], 35))
    sector = near if near.sum() >= 30 else ring
    n = int(sector.sum())
    if n < 30:
        return None, n
    dense = int((vol_c[sector] >= WALL_HU_MIN).sum())
    return dense / n, n


def load_case(case_id: str, model: str | None):
    """Кейс и модель, чьей маской считаем."""
    c = cb.get_case(case_id)
    if c is None:
        raise SystemExit(f"кейс {case_id} не найден")
    model = model or cb.mask_model_for(c)
    if model is None or not c["masks"].get(model):
        raise SystemExit(f"у кейса {case_id} нет готовой маски")
    return c, model


def analyze(case_id: str, model: str | None = None) -> dict:
    c, model = load_case(case_id, model)
    return analyze_case(c, model)


def analyze_case(c: dict, model: str) -> dict:
    # общий кэш с срезами и мешингом: маска уже отфильтрована
    vol, mask = cb.arrays_for(c, model)
    spacing = np.array(c["spacing"], dtype=float)
    case_id = c["case_id"]
    findings: list[dict] = []

    present = [t for t in cb.TEETH_CLASSES if (mask == t).sum() >= MIN_TOOTH_VOX]
    missing = [t for t in cb.TEETH_CLASSES if t not in present]
    jaw = np.isin(mask, JAW_CLASSES)

    # ---- качество маски: без этого находки нельзя показывать как факты ----
    quality = []
    vox_l = int((mask == CANAL_LEFT).sum())
    vox_r = int((mask == CANAL_RIGHT).sum())
    canal_ok = {"left": vox_l > 500, "right": vox_r > 500}
    if vox_l and vox_r:
        ratio = max(vox_l, vox_r) / max(min(vox_l, vox_r), 1)
        if ratio > CANAL_ASYMMETRY:
            weak = "справа" if vox_r < vox_l else "слева"
            quality.append(_finding(
                "canal_asymmetry",
                f"Канал {weak} сегментирован в {ratio:.1f}× хуже противоположного "
                f"({vox_l} / {vox_r} вокс.) — измерения на этой стороне ненадёжны",
                "review", severity="warning", confidence="high"))
            canal_ok["right" if vox_r < vox_l else "left"] = False
    for side, key, n in (("слева", "left", vox_l), ("справа", "right", vox_r)):
        if n <= 500:
            quality.append(_finding(
                "canal_missing",
                f"Нижнечелюстной канал {side} не сегментирован ({n} вокс.) — "
                f"расстояния до канала с этой стороны не считались",
                "review", severity="warning"))

    # ---- качество маски: зубы-огрызки ----
    # Зуб, от которого в маске остался огрызок, формально проходит порог
    # присутствия, и находки по нему считаются как по нормальному — а они
    # недостоверны все сразу: и длина корня, и уровень кости, и расстояния.
    # Раньше это ловилось только глазами (client4: зубы 17 и 46).
    vox_mm3 = float(np.prod(spacing))
    stubs = []
    for t in present:
        v = float((mask == t).sum()) * vox_mm3
        if v < STUB_VOLUME_MM3:
            stubs.append((t, v))
    if stubs:
        quality.append(_finding(
            "tooth_stub",
            "Маска зуба — огрызок у " + ", ".join(
                f"{t} ({v:.0f} мм³)" for t, v in sorted(stubs))
            + f": это ниже {STUB_VOLUME_MM3:.0f} мм³, при норме от 140 мм³ у "
              f"самого мелкого зуба (нижний резец). Измерения по этим зубам "
              f"недостоверны — сегментация не восстановила зуб целиком",
            "review", severity="warning", confidence="high",
            basis=[_basis("объём в маске", round(min(v for _t, v in stubs)),
                          "мм³", f"огрызок ниже {STUB_VOLUME_MM3:.0f} мм³",
                          note="объём на КЛКТ 0.3 мм завышен примерно на 28% "
                               "против микро-КТ, поэтому порог мягкий")]))
        # Из клинических находок такие зубы убираем: недостоверное измерение
        # хуже отсутствующего, врач не отличит его от настоящего.
        stub_teeth = {t for t, _v in stubs}
        present = [t for t in present if t not in stub_teeth]

    # ---- 1. отсутствующие зубы + плотность кости в лунке ----
    # Порог «это кость» — от медианы дентина ЭТОГО ЖЕ скана. Абсолютный не
    # годится: шкала КЛКТ не количественная (см. cbct_bone_quality).
    bone_thr = bq.dentin_threshold(vol, mask, present)

    for t in missing:
        nb = [x for x in (t - 1, t + 1) if x in present]
        text = f"Зуб {t} отсутствует"
        bvtv = None
        if nb and bone_thr is not None:
            cen = np.array([np.array(np.where(mask == x)).mean(axis=1) for x in nb])
            pt = np.round(cen.mean(axis=0)).astype(int)
            r = bq.bone_fraction(vol, jaw, np.array(spacing, float), pt, bone_thr)
            if r:
                bvtv = r["bvtv"]
                text += "; " + bq.describe(bvtv, bq.zone_of(t))
        basis = [_basis("зуб в маске", 0, "вокс.",
                        f"порог присутствия {MIN_TOOTH_VOX} вокс.")]
        if bvtv is not None:
            zone = bq.zone_of(t)
            basis.append(_basis(
                "доля костной ткани в ложе", round(bvtv, 2), "",
                f"ориентир зоны {bq.ZONE_REF[zone]:.2f} "
                f"({bq.ZONE_LABEL[zone]}), проба радиусом "
                f"{bq.PROBE_R_MM:.0f} мм",
                note="порог кости взят от дентина ЭТОГО снимка; классы D1-D4 "
                     "не выдаём — они определены через единицы Хаунсфилда, "
                     "которых КЛКТ не даёт"))
        findings.append(_finding(
            "tooth_missing", text, "prostho",
            severity="warning", tooth=t, value=(round(bvtv, 2) if bvtv else None),
            unit="доля кости",
            action="ортопед: оценить показания к имплантации/протезированию",
            confidence="medium" if bvtv is not None else "high", basis=basis))

    # ---- 2. расстояние корня до нижнечелюстного канала ----
    canal = np.isin(mask, [CANAL_LEFT, CANAL_RIGHT])
    if canal.sum() > 500:
        dmap = ndi.distance_transform_edt(~canal, sampling=spacing)
        for t in sorted(present):
            if not is_lower(t):
                continue
            side = "right" if fdi_quadrant(t) == 4 else "left"
            d = float(dmap[mask == t].min())
            conf = "high" if canal_ok.get(side) else "low"
            cbasis = [_basis(
                "расстояние корень → канал", round(d, 2), "мм",
                f"контакт ≤{CANAL_CONTACT_MM}, близко ≤{CANAL_CLOSE_MM}",
                note=_vox_note(spacing) + (
                    "" if canal_ok.get(side)
                    else "; канал с этой стороны сегментирован плохо"))]
            # Кортикальная стенка — признак сильнее расстояния, поэтому
            # считаем её у всех, кто вообще близко, и добавляем в основание.
            wall = wall_n = None
            if d <= CANAL_CLOSE_MM:
                wall, wall_n = cortical_wall_fraction(
                    vol, canal, mask == t, spacing)
                if wall is not None:
                    cbasis.append(_basis(
                        "кортикальная стенка канала со стороны зуба",
                        round(100 * wall), "%",
                        f"считаем стенку утраченной ниже {100*WALL_MIN_FRACTION:.0f}%",
                        note=f"стенка 0.68-1.10 мм это 2-3 вокселя при нашем "
                             f"{float(np.max(spacing)):.2f} — признак на пределе "
                             f"разрешения; риск при её отсутствии 15.7% против "
                             f"0.39% (метаанализ 18 работ)"))
            no_wall = wall is not None and wall < WALL_MIN_FRACTION
            if d <= CANAL_CONTACT_MM:
                findings.append(_finding(
                    "canal_contact",
                    f"Корень {t} контактирует с нижнечелюстным каналом ({d:.1f} мм)"
                    + (", кортикальная стенка не прослеживается" if no_wall
                       else ", кортикальная стенка сохранена" if wall is not None
                       else ""),
                    "surgery", severity="alert", tooth=t, value=round(d, 2),
                    unit="мм", confidence=conf, basis=cbasis,
                    action="хирург: удаление с высоким риском повреждения "
                           "n. alveolaris inferior, обсудить с пациентом"))
            elif d <= CANAL_CLOSE_MM:
                # Без стенки это уже не «близко», а настоящий риск: по
                # отношению шансов 9.87 признак весит больше самого расстояния.
                findings.append(_finding(
                    "canal_close",
                    f"Корень {t} близко к нижнечелюстному каналу ({d:.1f} мм)"
                    + (", кортикальная стенка не прослеживается" if no_wall else ""),
                    "surgery", severity="alert" if no_wall else "warning",
                    tooth=t, value=round(d, 2),
                    unit="мм", confidence=conf, basis=cbasis,
                    action="хирург: планировать удаление/имплантацию по КЛКТ"))

    # ---- 3. расстояние до дна гайморовой пазухи ----
    #
    # ВАЖНО, ЧТО ИМЕННО МЫ МЕРЯЕМ. Маска пазухи покрывает только ДНО полости,
    # а не пазуху целиком: замер на client5 дал 1.1 и 1.8 см3 у `ours`, 0.2 и
    # 1.2 у `pretrained` при анатомической норме около 15 см3 на пазуху. Это
    # не наша фильтрация (сырая маска с диска такая же, HU-гейт срезает 8-23%)
    # и не свойство одной модели — у ToothFairy3 пазухи размечены на 10%
    # объёма, а `pretrained` (ToothFairy2), которую держали донором пазух
    # именно как «размеченную нормально», даёт не больше.
    #
    # Для ЭТОГО измерения ограничение безвредно: размечена та часть полости,
    # которая прилегает к кости над корнями, то есть ровно дно, от которого и
    # считают расстояние при имплантации. Внутри маски 92% вокселей — воздух,
    # значит попадание точное, беда только с охватом.
    #
    # Чего из-за этого НЕЛЬЗЯ считать: объём пазухи, утолщение слизистой,
    # признаки синусита. Дорастить полость по воздуху не выходит — она через
    # соустье соединена с носовой полостью, та через ноздри с наружным
    # воздухом: свободная заливка забирает 1297 см3 из 1299, то есть весь
    # кадр, а сужение маски рвёт саму пазуху раньше, чем соустье.
    sinus = np.isin(mask, SINUS_CLASSES)
    if sinus.sum() > 500:
        dmap = ndi.distance_transform_edt(~sinus, sampling=spacing)
        # Объём маски пазух — врачу он нужен как честная мера охвата: по нему
        # видно, что мерялось дно, а не полость.
        sinus_cm3 = float(sinus.sum()) * float(np.prod(spacing)) / 1000.0
        for t in sorted(present):
            if is_lower(t) or t % 10 < 4:      # премоляры и моляры верхней
                continue
            d = float(dmap[mask == t].min())
            pct = sinus_percentile(d)
            if pct > SINUS_P_SHOW:
                continue        # дальше медианы выборки — не о чем говорить
            sbasis = [
                _basis("расстояние корень → дно пазухи", round(d, 2), "мм",
                       f"ближе, чем у {pct:.0f}% зубов выборки",
                       note=_vox_note(spacing)),
                _basis("медиана выборки", SINUS_REF_MM[SINUS_REF_PCTL.index(50)],
                       "мм",
                       f"{SINUS_REF_N_TEETH} верхних жевательных зубов, "
                       f"{SINUS_REF_N_CASES} томов",
                       note="близость к пазухе — норма: две трети зубов лежат "
                            "ближе 2 мм, поэтому отдаём перцентиль, а не тревогу"),
                _basis("охват маски пазухи", round(sinus_cm3, 1), "см³",
                       "размечено дно полости, не пазуха целиком",
                       note="анатомическая норма около 15 см³ на пазуху; "
                            "измеряется расстояние до дна, объём пазухи и "
                            "состояние слизистой по этой маске не оцениваются"),
            ]
            findings.append(_finding(
                "sinus_close",
                f"Корень {t} у дна гайморовой пазухи: {d:.1f} мм "
                f"(ближе, чем у {pct:.0f}% зубов выборки). Близость к пазухе "
                f"типична и сама по себе патологией не является",
                "prostho", severity="info",
                tooth=t, value=round(d, 2), unit="мм",
                confidence="medium", basis=sbasis,
                action="учесть при планировании удаления или имплантации "
                       "в этой позиции"))

    # ---- 4. подозрение на ретенцию ----
    z_up = occlusal_up(mask)
    # Уровень зубного ряда каждой челюсти отдельно: верхние и нижние зубы
    # смыкаются на разной высоте, общая медиана смешала бы их.
    occl = {}
    for jaw_up in (True, False):
        peers = [x for x in present
                 if (fdi_quadrant(x) in (1, 2)) == jaw_up and x % 10 <= 7]
        if len(peers) >= 4:
            occl[jaw_up] = float(np.median(
                np.nonzero(np.isin(mask, peers))[0]))
    for t in sorted(present):
        enc = crown_encased_fraction(mask, t, jaw, z_up)
        upper = fdi_quadrant(t) in (1, 2)
        above = None
        if upper in occl:
            z = float(np.nonzero(mask == t)[0].mean())
            # «выше ряда» — в сторону от окклюзионной плоскости, то есть
            # вверх для верхних зубов и вниз для нижних.
            d = (z - occl[upper]) if z_up else (occl[upper] - z)
            above = (d if upper else -d) * float(spacing[0])
        by_bone = enc >= RETENTION_ENCASED
        by_pos = above is not None and above >= RETENTION_ABOVE_MM
        if not (by_bone or by_pos):
            continue
        why = []
        if by_bone:
            why.append(f"окружён костью на {enc*100:.0f}%")
        if by_pos:
            why.append(f"не дошёл до зубного ряда на {above:.1f} мм")
        sbasis = [_basis("доля коронки, упирающейся в кость", round(enc, 2),
                         "доля", f"порог ретенции ≥{RETENTION_ENCASED}",
                         note="порог по разрыву в распределении на "
                              "client2/client4, врачом не подписан")]
        if above is not None:
            sbasis.append(_basis("выше уровня зубного ряда", round(above, 1),
                                 "мм", f"порог ≥{RETENTION_ABOVE_MM} мм"))
        findings.append(_finding(
            "tooth_impacted",
            f"Зуб {t} — подозрение на ретенцию: " + ", ".join(why),
            "surgery", severity="warning", tooth=t, value=round(enc, 3),
            unit="доля", confidence="medium", basis=sbasis,
            action="хирург: оценить показания к удалению ретенированного зуба"))

    # ---- 5. реставрации ----
    for cls, name in PROSTH_RU.items():
        bm = mask == cls
        if bm.sum() < 50:
            continue
        lab, n = ndi.label(bm)
        for i in range(1, n + 1):
            comp = lab == i
            if comp.sum() < 50:
                continue
            cen = np.array(np.where(comp)).mean(axis=1)
            best, bd = None, 1e9
            for t in present:
                tc = np.array(np.where(mask == t)).mean(axis=1)
                dd = float(np.linalg.norm((tc - cen) * spacing))
                if dd < bd:
                    best, bd = t, dd
            if bd > PROSTH_MAX_DIST_MM:
                quality.append(_finding(
                    "prosth_artifact",
                    f"Объект класса «{name}» в {bd:.0f} мм от ближайшего зуба — "
                    f"это артефакт маски, а не реставрация",
                    "review", severity="info", confidence="high"))
                continue
            findings.append(_finding(
                "restoration",
                f"{name.capitalize()} в области зуба {best}", "prostho",
                severity="info", tooth=best, value=round(bd, 1), unit="мм",
                confidence="medium",
                basis=[_basis("до центра ближайшего зуба", round(bd, 1), "мм",
                              f"дальше {PROSTH_MAX_DIST_MM:.0f} мм — артефакт",
                              note="привязка к зубу по расстоянию центров: "
                                   "у моста она условна"),
                       _basis("объём объекта", int(comp.sum()), "вокс.",
                              "минимум 50 вокс.")],
                action="ортопед: оценить состояние реставрации"))

    # ---- 6. уровень костной ткани (пародонт) ----
    # Считает отдельный модуль: измерение объёмное, а сбой в нём не должен
    # ронять остальные находки.
    try:
        perio = cbct_perio.analyze_case(c, model)
        for f in cbct_perio.build_findings(perio):
            (quality if f["route"] == "review" else findings).append(f)
    except Exception as exc:  # noqa: BLE001
        print(f"КЛКТ: уровень кости не посчитан для {case_id}: {exc}")
        perio = None

    # ---- 7. эндодонтический статус: пролечен канал или зуб витален ----
    # Есть только у моделей с пульпой (tf3): без маски пульпы сравнивать нечего.
    try:
        endo = cbct_endo.analyze_case(c, model)
        for f in cbct_endo.build_findings(endo):
            (quality if f["route"] == "review" else findings).append(f)
    except Exception as exc:  # noqa: BLE001
        print(f"КЛКТ: эндодонтия не посчитана для {case_id}: {exc}")
        endo = None

    return {
        "case_id": case_id,
        "model": model,
        "perio": perio,
        "endo": endo,
        "model_label": cb.MODEL_LABELS.get(model, model),
        "draft": True,          # пороги не подписаны врачом
        "teeth_present": present,
        "teeth_missing": missing,
        "findings": findings,
        "quality": quality,
        # порядок этапов отдаём фронту, чтобы он не дублировал клинику у себя
        "routes": cr.route_list(),
    }


def occlusal_up(mask: np.ndarray) -> bool:
    """Растёт ли ось срезов вверх. Один детектор на все модули — в backend."""
    return cb.occlusal_up(mask)


def crown_encased_fraction(mask: np.ndarray, tooth: int, jaw: np.ndarray,
                           z_up: bool) -> float:
    """Доля оболочки КОРОНКИ зуба, упирающаяся в кость или соседний зуб.

    Считать по всему зубу нельзя: корень в кости и у здорового зуба, поэтому
    на client4 весь диапазон сжимается в 40-56% и ретенцию не отделить.
    Диагностична именно коронка: у прорезавшегося зуба она открыта в полость
    рта (оболочка граничит с воздухом), у ретенированного — накрыта костью.
    Берём 25% вокселей зуба со стороны окклюзионной плоскости.
    """
    bm = mask == tooth
    idx = np.where(bm)
    z = idx[0]
    upper = fdi_quadrant(tooth) in (1, 2)
    # коронка направлена к окклюзионной плоскости: у верхних вниз, у нижних вверх
    crown_at_max_z = (not upper) if z_up else upper
    thr = np.percentile(z, 75 if crown_at_max_z else 25)
    keep = z >= thr if crown_at_max_z else z <= thr

    crown = np.zeros_like(bm)
    crown[idx[0][keep], idx[1][keep], idx[2][keep]] = True
    ci = np.where(crown)
    lo = [max(int(i.min()) - 2, 0) for i in ci]
    hi = [min(int(i.max()) + 3, s) for i, s in zip(ci, mask.shape)]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = crown[box]
    # оболочка коронки, из неё исключаем сам зуб целиком (включая шейку)
    shell = ndi.binary_dilation(sub, iterations=1) & ~bm[box]
    n = int(shell.sum())
    if n == 0:
        return 0.0
    neighbours = jaw[box] | (mask[box] > 10)   # кость или соседний зуб
    return float((shell & neighbours).sum()) / n


def route_report(res: dict) -> list[tuple]:
    """Группировка находок по специалистам в клиническом порядке."""
    buckets: dict[str, list] = {}
    for f in res["findings"] + res["quality"]:
        buckets.setdefault(f["route"], []).append(f)
    out = []
    for key, items in buckets.items():
        label, order = ROUTE.get(key, (key, 8))
        out.append((order, label, items))
    return sorted(out)


SEV = {"alert": "!!", "warning": " !", "info": "  "}


def print_report(res: dict) -> None:
    print(f"\n{'='*74}")
    print(f"КЛКТ, черновик находок — кейс {res['case_id']} / {res['model_label']}")
    print(f"{'='*74}")
    print(f"Зубов в наличии: {len(res['teeth_present'])}   "
          f"отсутствуют: {res['teeth_missing'] or '—'}")

    for order, label, items in route_report(res):
        print(f"\n[{order}] {label}")
        for f in sorted(items, key=lambda x: (x["severity"] != "alert",
                                              x["tooth"] or 0)):
            mark = SEV.get(f["severity"], "  ")
            conf = "" if f["confidence"] == "high" else f"  (доверие: {f['confidence']})"
            print(f"  {mark} {f['text']}{conf}")
            if f["action"]:
                print(f"       -> {f['action']}")
    if res["draft"]:
        print(f"\n{'-'*74}")
        print("ЧЕРНОВИК. Пороги не подтверждены врачом; не является назначением.")


def calibrate(case_id: str, model: str | None) -> None:
    """Печатает долю «замурованности» по всем зубам — для подбора порога."""
    c, model = load_case(case_id, model)
    _, mask = cb.arrays_for(c, model)
    jaw = np.isin(mask, JAW_CLASSES)
    z_up = occlusal_up(mask)
    rows = []
    for t in cb.TEETH_CLASSES:
        if (mask == t).sum() < MIN_TOOTH_VOX:
            continue
        rows.append((crown_encased_fraction(mask, t, jaw, z_up), t))
    rows.sort(reverse=True)
    print(f"{case_id}: доля оболочки коронки в кости (по убыванию), "
          f"ось срезов вверх={z_up}")
    for frac, t in rows:
        print(f"  зуб {t}: {frac*100:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--model", default=None, help="ours | pretrained")
    ap.add_argument("--json", default=None, help="сохранить находки в файл")
    ap.add_argument("--calibrate", action="store_true",
                    help="печать доли замурованности зубов (подбор порога ретенции)")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.case_id, args.model)
        return

    res = analyze(args.case_id, args.model)
    print_report(res)
    if args.json:
        Path(args.json).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nсохранено: {args.json}")


if __name__ == "__main__":
    main()
