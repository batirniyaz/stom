"""Находки по интраоральному скану для всех специальностей сразу.

Один скан — четыре разных взгляда на него. Модуль устроен как `cbct_findings`
для КЛКТ: новых моделей нет, всё считается геометрией поверх сегментации
(общий фундамент — `ios_geometry`), каждая находка адресована конкретному
врачу, всё помечено draft до подписи.

    ortho     тремы, скученность, ширина зубов, индекс Болтона
    prostho   дефекты зубного ряда, место под имплант
    perio     длина клинической коронки, рецессия
    gnatho    фасетки стирания

Карта окклюзионных контактов считается отдельно (`ios_occlusion_map`): она
тяжёлая, у неё свой файл с зазорами на каждую вершину. Здесь только то, что
считается по геометрии дуги.

ВАЖНО: все пороги — ЧЕРНОВИК. Зуб, чей меш не прошёл проверку качества
(`ios_geometry.mesh_quality`), в находки идёт с пониженным доверием либо не
идёт вовсе — иначе врач получит цифру, посчитанную по обрывку сетки.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_findings.py batir
    .venv/Scripts/python.exe research/scripts/ios_findings.py batir --group ortho
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clinic_routes as cr  # noqa: E402
import ios_geometry as G  # noqa: E402

CASES = G.CASES

# Роли общие для всех источников находок (см. clinic_routes).
ROUTE = cr.ROUTES

# --- пороги ортодонтии (ЧЕРНОВИК) ---
TREMA_MM = 0.50          # промежуток между соседями больше — трема
DIASTEMA_TEETH = {(11, 21), (31, 41)}   # между центральными резцами — диастема
OFF_ARCH_MM = 1.2        # отклонение зуба от линии дуги больше — вне ряда
ARCH_SMOOTH_MM = 0.5     # допуск сглаживания дуги на зуб
ARCH_RMS_MAX = 0.9       # хуже подгонка — отклонениям отдельных зубов не верить

# Асимметрия дуги: перцентили СОБСТВЕННОГО измерения на 220 сканах Teeth3DS.
# Своей нормы здесь нет и взять её негде — разметки «дуга симметрична» не
# существует ни в одном доступном датасете. Поэтому, как на пародонте и
# прикусе, отдаём миллиметры И место в выборке, а степень называем от выборки
# («выше нормы»), а не клинически («перекрёстный прикус»). Порог по 90-му
# перцентилю — та же специфичность ~90%, что выбрана на прикусе: врач всё
# равно смотрит пациента сам, а отчёт, тревожащий на каждом втором, он читать
# перестанет.
#            медиана по скану:  50%=0.73  75%=1.32  85%=1.89  90%=3.32  95%=4.13
#            худшая пара:       50%=1.66  75%=2.65  85%=4.54  90%=6.37  95%=7.77
SYM_MEDIAN_REF = [0.73, 1.32, 1.89, 3.32, 4.13, 5.50]
SYM_WORST_REF = [1.66, 2.65, 4.54, 6.37, 7.77, 10.18]
SYM_REF_Q = [50, 75, 85, 90, 95, 99]
SYM_MEDIAN_MM = 3.32     # 90-й перцентиль выборки — выше него ряд перекошен
SYM_WORST_MM = 6.37      # 90-й перцентиль по худшей паре

# Индекс Болтона: отношение суммы ширин шести нижних передних к шести верхним.
# Норма 78.2 ± 1.7 % (Bolton, 1958). Отклонение означает, что зубы верхней и
# нижней челюсти не совпадают по размеру, и одним перемещением прикус не
# закрыть — нужно либо сошлифовывание, либо реставрация.
BOLTON_ANTERIOR = 78.2
BOLTON_TOL = 1.7
UPPER_ANT = [13, 12, 11, 21, 22, 23]
LOWER_ANT = [43, 42, 41, 31, 32, 33]


def _finding(code, text, route, severity="info", teeth=None, value=None,
             unit=None, action=None, confidence="medium"):
    return {"code": code, "teeth": sorted(teeth or []), "value": value,
            "unit": unit, "severity": severity, "text": text, "action": action,
            "route": route, "confidence": confidence}


class Case:
    """Всё, что нужно находкам: меши, оси дуги, размеры, качество сеток."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.teeth, self.gums = G.load_case(case_id)
        self.frame = G.arch_frame(self.teeth)
        self.seq = G.arch_sequence(self.teeth)
        self.geom = G.analyze_geometry(case_id)
        self.by_fdi = {r["fdi"]: r for r in self.geom["teeth"]}
        self.bad = self.geom["mesh_quality"]
        self._trees = {f: cKDTree(np.asarray(m.vertices))
                       for f, m in self.teeth.items()}
        self._curves = {}

    def gap(self, a: int, b: int) -> float:
        """Минимальный зазор между двумя зубами, мм."""
        d, _ = self._trees[b].query(np.asarray(self.teeth[a].vertices), workers=-1)
        return float(d.min())

    def facing_points(self, a: int, b: int) -> tuple:
        """Ближайшие друг к другу точки на поверхностях двух зубов."""
        Va, Vb = np.asarray(self.teeth[a].vertices), np.asarray(self.teeth[b].vertices)
        d, idx = self._trees[b].query(Va, workers=-1)
        i = int(np.argmin(d))
        return Va[i], Vb[int(idx[i])]

    def width(self, fdi: int):
        r = self.by_fdi.get(fdi)
        return r["mesiodistal_mm"] if r else None

    def plane(self, fdi: int) -> np.ndarray:
        """Центр зуба в осях дуги: (поперёк, вперёд)."""
        return self.frame.project(self.frame.centroids[fdi])[:2]

    def arch_curve(self, jaw: str) -> tuple:
        """Линия дуги челюсти: (точки Nx2 в осях дуги, длина вдоль неё)."""
        if jaw not in self._curves:
            from scipy.interpolate import UnivariateSpline
            row = [f for f in self.seq[jaw]]
            P = np.array([self.plane(f) for f in row])
            d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
            s = len(P) * ARCH_SMOOTH_MM ** 2
            ts = np.linspace(d[0], d[-1], 800)
            C = np.stack([UnivariateSpline(d, P[:, 0], s=s, k=3)(ts),
                          UnivariateSpline(d, P[:, 1], s=s, k=3)(ts)], axis=1)
            arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
            self._curves[jaw] = (C, arc)
        return self._curves[jaw]

    def arc_between(self, jaw: str, p: np.ndarray, q: np.ndarray) -> float:
        """Расстояние между двумя точками ВДОЛЬ дуги, а не по прямой.

        Для дефекта в один зуб разница невелика, но на трёх зубах подряд хорда
        короче дуги на 3 мм (замер на смоделированных дефектах) — ортопед
        получил бы заниженный промежуток.
        """
        C, arc = self.arch_curve(jaw)
        pp = self.frame.project(p)[:2]
        qq = self.frame.project(q)[:2]
        i = int(np.argmin(np.hypot(*(C - pp).T)))
        j = int(np.argmin(np.hypot(*(C - qq).T)))
        return float(abs(arc[j] - arc[i]))


# ============================ ортодонт ============================

def fit_arch(case: Case, span: list) -> tuple:
    """Плавная линия дуги: (отклонения зубов от неё со знаком, RMS подгонки).

    Кривая параметризуется ПОРЯДКОМ зубов вдоль дуги, а не поперечной
    координатой. Парабола `fwd = f(lat)` здесь не годится принципиально: она
    не умеет повторять подкову, срезает углы и провисает в середине. Замер на
    скане клиники: RMS отклонений 2.22 мм у параболы против 0.39 мм у сплайна,
    причём у параболы клыки уходили «вестибулярно», а центральные резцы и
    моляры «язычно» — это её собственная форма, а не положение зубов.

    Знак: + зуб смещён язычно (внутрь дуги), − вестибулярно (наружу).

    Длину дуги отсюда НЕ отдаём. Сумма ширин зубов и длина кривой через их
    центры — величины разной природы (кривая не содержит половин крайних
    зубов), а после поправки на них точность метода выходит ±2-3 мм. Дефицит
    места такой точностью не измеришь. Настоящая мера скученности — индекс
    Литтла по анатомическим контактным точкам, а они появятся с моделью
    ориентиров (3DTeethLand).
    """
    from scipy.interpolate import UnivariateSpline
    P = np.array([case.plane(f) for f in span])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s = len(P) * ARCH_SMOOTH_MM ** 2
    sx = UnivariateSpline(d, P[:, 0], s=s, k=3)
    sy = UnivariateSpline(d, P[:, 1], s=s, k=3)
    ts = np.linspace(d[0], d[-1], 800)
    C = np.stack([sx(ts), sy(ts)], axis=1)

    center = P.mean(axis=0)
    dev = {}
    for f, p in zip(span, P):
        k = int(np.argmin(np.hypot(*(C - p).T)))
        off = p - C[k]
        inward = center - C[k]
        n = np.linalg.norm(inward)
        sign = 1.0 if n < 1e-9 or off @ (inward / n) > 0 else -1.0
        dev[f] = sign * float(np.hypot(*off))
    rms = float(np.sqrt(np.mean(np.square(list(dev.values())))))
    return dev, rms


def _sym_pct(v: float, ref: list) -> int:
    """Место значения в референсной выборке, в процентах."""
    for q, r in zip(SYM_REF_Q, ref):
        if v <= r:
            return q
    return 99


def findings_ortho(case: Case) -> list:
    out = []

    # --- 1. промежутки между соседними зубами ---
    tremas = []
    for jaw, row in case.seq.items():
        for a, b in zip(row, row[1:]):
            g = case.gap(a, b)
            if g >= TREMA_MM:
                tremas.append((a, b, g))
    for a, b, g in tremas:
        pair = tuple(sorted((a, b)))
        is_dia = pair in DIASTEMA_TEETH
        conf = "low" if (a in case.bad or b in case.bad) else "medium"
        out.append(_finding(
            "diastema" if is_dia else "trema",
            f"{'Диастема' if is_dia else 'Трема'} между зубами {a} и {b}: "
            f"{g:.1f} мм",
            "ortho", severity="warning" if is_dia else "info",
            teeth=[a, b], value=round(g, 2), unit="мм", confidence=conf,
            action="ортодонт: оценить причину промежутка "
                   "(размер зубов, положение, уздечка)"))

    # --- 2. ровность ряда: зубы, выбитые из линии дуги ---
    for jaw in ("upper", "lower"):
        row = [f for f in case.seq[jaw] if f % 10 <= 6]      # от шестёрки до шестёрки
        if len(row) < 8:
            continue
        dev, rms = fit_arch(case, row)
        jaw_ru = "верхней" if jaw == "upper" else "нижней"      # дуги
        row_ru = "верхнего" if jaw == "upper" else "нижнего"    # ряда
        if rms > ARCH_RMS_MAX:
            out.append(_finding(
                "arch_fit_poor",
                f"Линия {jaw_ru} дуги подгоняется плохо (RMS {rms:.1f} мм) — "
                f"положение отдельных зубов по ней не оценивалось",
                "ortho", severity="info", teeth=row, confidence="high",
                action="проверить сегментацию: обычно виноват меш, а не зубы"))
            continue
        off = [(f, d) for f, d in dev.items()
               if abs(d) >= OFF_ARCH_MM and f not in case.bad]
        for f, d in sorted(off, key=lambda x: -abs(x[1])):
            out.append(_finding(
                "off_arch",
                f"Зуб {f} стоит вне линии {jaw_ru} дуги на {abs(d):.1f} мм "
                f"({'язычно' if d > 0 else 'вестибулярно'})",
                "ortho", severity="warning", teeth=[f], value=round(float(d), 2),
                unit="мм", confidence="medium",
                action="ортодонт: положение зуба в ряду"))
        out.append(_finding(
            "arch_irregularity",
            f"Неровность {row_ru} ряда: среднее отклонение от линии дуги "
            f"{rms:.1f} мм, наибольшее {max(abs(d) for d in dev.values()):.1f} мм",
            "ortho", value=round(rms, 2), unit="мм", teeth=row,
            action="ортодонт: описательная оценка. Дефицит места в мм по этому "
                   "скану не считается — нужна разметка контактных точек"))

    # --- 3. симметрия дуги относительно средней линии ---
    # Ряд может быть ровным (пункт 2) и при этом перекошенным целиком: пункт 2
    # меряет, насколько зуб выбился из ПЛАВНОЙ линии, а этот — насколько правая
    # половина совпадает с зеркальной левой. Рентгенолог смотрит именно на это.
    for jaw in ("upper", "lower"):
        teeth = {f: m for f, m in case.teeth.items()
                 if f in case.seq[jaw]}
        if len(teeth) < 6:
            continue
        sym = G.arch_symmetry(case.frame, teeth, case.bad)
        if not sym["reliable"] or sym["median_mm"] is None:
            continue
        jaw_ru = "верхней" if jaw == "upper" else "нижней"
        med, worst = sym["median_mm"], sym["worst"]
        p_med = _sym_pct(med, SYM_MEDIAN_REF)
        p_worst = _sym_pct(abs(worst["total_mm"]), SYM_WORST_REF)
        row_teeth = sorted({p["right"] for p in sym["pairs"] if p["reliable"]} |
                           {p["left"] for p in sym["pairs"] if p["reliable"]})
        # Основание — что измерено и с чем сравнивается. Процента уверенности
        # у нас нет и не будет: вероятности здесь взять неоткуда.
        basis = (f"расхождение одноимённых пар с зеркальным отражением: "
                 f"медиана {med:.1f} мм (выше {p_med}% сканов выборки), "
                 f"наибольшее {abs(worst['total_mm']):.1f} мм у пары "
                 f"{worst['right']}/{worst['left']} (выше {p_worst}%); "
                 f"выборка — 220 сканов Teeth3DS, порог по 90-му перцентилю "
                 f"({SYM_MEDIAN_MM} и {SYM_WORST_MM} мм)")
        skew = med > SYM_MEDIAN_MM or abs(worst["total_mm"]) > SYM_WORST_MM
        if skew:
            out.append(_finding(
                "arch_asymmetry",
                f"Перекос {jaw_ru} дуги: медиана расхождения парных зубов "
                f"{med:.1f} мм, наибольшее {abs(worst['total_mm']):.1f} мм "
                f"у пары {worst['right']}/{worst['left']} — выше {max(p_med, p_worst)}% "
                f"сканов референсной выборки",
                "ortho", severity="warning", teeth=row_teeth,
                value=round(med, 2), unit="мм", confidence="medium",
                action="ортодонт: оценить причину асимметрии "
                       "(смещение средней линии, потеря зуба, привычка жевать "
                       "одной стороной)"))
        else:
            out.append(_finding(
                "arch_asymmetry",
                f"Симметрия {jaw_ru} дуги: медиана расхождения парных зубов "
                f"{med:.1f} мм — в пределах выборки (выше {p_med}%)",
                "ortho", severity="info", teeth=row_teeth,
                value=round(med, 2), unit="мм", confidence="medium"))
        out[-1]["basis"] = basis
        out[-1]["pairs"] = sym["pairs"]

    # --- 4. индекс Болтона ---
    missing = [f for f in UPPER_ANT + LOWER_ANT
               if case.width(f) is None or f in case.bad]
    if missing:
        out.append(_finding(
            "bolton_skipped",
            f"Индекс Болтона не посчитан: зубы {sorted(missing)} измерены "
            f"ненадёжно (сегментация). Считать по ним соотношение размеров "
            f"нельзя — ошибка уйдёт прямо в план лечения",
            "ortho", severity="info", teeth=missing, confidence="high",
            action="переснять или поправить сегментацию этих зубов"))
    else:
        up = sum(case.width(f) for f in UPPER_ANT)
        lo = sum(case.width(f) for f in LOWER_ANT)
        ratio = lo / up * 100
        delta = ratio - BOLTON_ANTERIOR
        if abs(delta) > BOLTON_TOL:
            bigger = "нижние" if delta > 0 else "верхние"
            out.append(_finding(
                "bolton",
                f"Передний индекс Болтона {ratio:.1f}% при норме "
                f"{BOLTON_ANTERIOR}±{BOLTON_TOL}% — {bigger} передние зубы "
                f"крупнее пары (сумма {lo:.1f} мм против {up:.1f} мм)",
                "ortho", severity="warning", value=round(ratio, 1), unit="%",
                teeth=UPPER_ANT + LOWER_ANT,
                action="ортодонт: одним перемещением не закрыть — нужна "
                       "сепарация или реставрация"))
        else:
            out.append(_finding(
                "bolton",
                f"Передний индекс Болтона {ratio:.1f}% — в норме "
                f"({BOLTON_ANTERIOR}±{BOLTON_TOL}%)",
                "ortho", value=round(ratio, 1), unit="%", confidence="medium"))
    return out


# ============================ ортопед ============================

# --- пороги ортопедии (ЧЕРНОВИК) ---
WISDOM = {18, 28, 38, 48}
IMPLANT_MIN_MM = 6.0        # уже — имплант обычного диаметра не встанет
CLEARANCE_MIN_MM = 5.0      # ниже — коронке не хватит высоты
CLEARANCE_TIGHT_MM = 7.0    # ниже — тесно, считать высоту прицельно
CLEARANCE_IMPLAUSIBLE_MM = 2.0  # меньше — гребень не прочитался, не сообщаем
OVERERUPT_MM = 1.5          # выступ зуба за плавную линию смыкания
GAP_REGION_PAD_MM = 2.0     # запас вокруг дефекта при поиске гребня
RIDGE_LATERAL_MM = 4.0      # ширина полосы вдоль дуги, в которой ищем гребень
RIDGE_CORE_FRACTION = 0.6   # доля дефекта по центру: края заняты сосочками

# Идеальный ряд без восьмёрок: по нему ищем, чего не хватает. Восьмёрки
# исключены намеренно — их не восстанавливают, и «отсутствие» там не дефект.
IDEAL = {
    "upper": [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27],
    "lower": [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37],
}


def antagonist(fdi: int) -> int:
    return {1: 4, 4: 1, 2: 3, 3: 2}[fdi // 10] * 10 + fdi % 10


def defect_runs(case: Case, jaw: str) -> list:
    """Подряд отсутствующие позиции ряда -> [(список позиций, соседи)]."""
    row = IDEAL[jaw]
    runs, cur = [], []
    for f in row:
        if f in case.teeth:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(f)
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        i, j = row.index(run[0]) - 1, row.index(run[-1]) + 1
        left = row[i] if i >= 0 and row[i] in case.teeth else None
        right = row[j] if j < len(row) and row[j] in case.teeth else None
        out.append((run, left, right))
    return out


def ridge_clearance(case: Case, jaw: str, left: int, right: int):
    """Высота от гребня в дефекте до зубов-антагонистов, ПО ВЕРТИКАЛИ, мм.

    Свободное расстояние в 3D здесь не годится: оно цепляется не за
    вертикальный зазор, а за щёчную десну, которая в преддверии проходит в
    паре миллиметров от боковой поверхности зубов напротив. Замер на
    смоделированном дефекте 46 давал 1.6 мм вместо ожидаемых 6-8.

    Поэтому: берём вершину гребня в области дефекта и меряем разницу высот до
    поверхности антагонистов, отобранных по близости В ПЛОСКОСТИ дуги.
    """
    gum = case.gums.get(jaw)
    opp = [m for f, m in case.teeth.items() if G.is_upper(f) != (jaw == "upper")]
    if gum is None or not opp:
        return None
    up = case.frame.up
    upper = jaw == "upper"
    C, arc = case.arch_curve(jaw)
    pl, pr = case.facing_points(left, right)
    s_lo, s_hi = sorted(
        arc[int(np.argmin(np.hypot(*(C - case.frame.project(p)[:2]).T)))]
        for p in (pl, pr))

    # Отбираем десну ВДОЛЬ дуги, а не шаром: шар захватывал щёчную десну
    # преддверия, которая у нижней челюсти поднимается почти до уровня
    # смыкания и по вертикали упирается в боковую поверхность верхних зубов.
    # Берём СЕРЕДИНУ дефекта: у краёв стоят межзубные сосочки, а они
    # поднимаются почти до контактных пунктов и выдают себя за гребень
    # (на смоделированном дефекте 46 давали 2 мм вместо 6-8).
    pad = (s_hi - s_lo) * (1.0 - RIDGE_CORE_FRACTION) / 2
    s_lo, s_hi = s_lo + pad, s_hi - pad

    V = np.asarray(gum.vertices)
    Vxy = case.frame.project(V)[:, :2]
    dist = np.hypot(*(C[:, None, :] - Vxy[None, :, :]).transpose(2, 0, 1))
    k = np.argmin(dist, axis=0)
    off, s = dist[k, np.arange(len(V))], arc[k]
    sel = V[(s >= s_lo) & (s <= s_hi) & (off <= RIDGE_LATERAL_MM)]
    if len(sel) < 20:
        return None
    # гребень: у нижней челюсти самые высокие точки, у верхней самые низкие
    hs = sel @ up
    crest = sel[hs <= np.percentile(hs, 10)] if upper else \
        sel[hs >= np.percentile(hs, 90)]

    P = np.vstack([np.asarray(m.vertices) for m in opp])
    Pxy = case.frame.project(P)[:, :2]
    Cxy = case.frame.project(crest)[:, :2]
    tree = cKDTree(Pxy)
    gaps = []
    for c_xy, c_h in zip(Cxy, crest @ up):
        near = tree.query_ball_point(c_xy, r=1.5)     # столбик над точкой
        if not near:
            continue
        h = P[near] @ up
        # у нижнего гребня антагонисты сверху, у верхнего снизу
        gaps.append(float(h.min() - c_h) if not upper else float(c_h - h.max()))
    if len(gaps) < 5:
        return None
    return float(np.median(gaps))


def occlusal_protrusion(case: Case, jaw: str) -> tuple:
    """Насколько каждый зуб выступает за плавную линию смыкания, мм.

    Кривая Шпее делает высоту бугров закономерно разной вдоль ряда, поэтому
    сравнивать зуб надо не со средним по дуге, а с плавной линией по соседям.
    """
    from scipy.interpolate import UnivariateSpline
    row = [f for f in case.seq[jaw] if f not in case.bad]
    if len(row) < 6:
        return {}, 0.0
    up = case.frame.up
    h = {}
    for f in row:
        V = np.asarray(case.teeth[f].vertices) @ up
        h[f] = float(V.min() if G.is_upper(f) else V.max())
    P = np.array([case.plane(f) for f in row])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    y = np.array([h[f] for f in row])
    sp = UnivariateSpline(d, y, s=len(row) * 0.6 ** 2, k=3)
    resid = y - sp(d)
    # выступ «к антагонисту»: у верхних зубов это вниз, у нижних вверх
    sign = -1.0 if jaw == "upper" else 1.0
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return {f: sign * float(r) for f, r in zip(row, resid)}, rms


def findings_prostho(case: Case) -> list:
    out = []

    # --- 1. дефекты зубного ряда ---
    for jaw in ("upper", "lower"):
        jaw_ru = "верхней" if jaw == "upper" else "нижней"
        for run, left, right in defect_runs(case, jaw):
            if left is None or right is None:
                side = left or right
                out.append(_finding(
                    "defect_free_end",
                    f"Концевой дефект {jaw_ru} челюсти: нет зубов "
                    f"{run}, ряд обрывается за зубом {side}",
                    "prostho", severity="warning", teeth=run, confidence="high",
                    action="ортопед: имплантация или съёмный протез — "
                           "мост опереть не на что"))
                continue
            pl, pr = case.facing_points(left, right)
            width = case.arc_between(jaw, pl, pr)     # вдоль дуги, не по хорде
            clear = ridge_clearance(case, jaw, left, right)
            per_tooth = width / len(run)
            txt = (f"Дефект {jaw_ru} челюсти: нет {len(run)} зуб(ов) {run}. "
                   f"Промежуток {width:.1f} мм между {left} и {right}")
            act = "ортопед: имплантация или мост с опорой на соседние зубы"
            sev = "warning"
            # Защита от заведомо ложного результата: у настоящего беззубого
            # участка высота не бывает меньше пары миллиметров. Если вышло
            # меньше — гребень не прочитался (так ведёт себя смоделированный
            # дефект, где на месте зуба дыра). Лучше промолчать, чем выдать
            # врачу «места нет» на ровном месте.
            if clear is not None and clear < CLEARANCE_IMPLAUSIBLE_MM:
                txt_tail = "; высоту до антагониста измерить не удалось"
                clear = None
            else:
                txt_tail = ""
            if clear is not None:
                # Расчёт высоты НЕ ПРОВЕРЕН на настоящем дефекте: у
                # единственного нашего пациента нет удалённых зубов, а
                # смоделированный дефект гребня не воспроизводит — в сетке
                # на его месте дыра. Пока не появится реальный кейс,
                # цифра идёт с оговоркой прямо в тексте.
                txt += (f", высота до антагониста ~{clear:.1f} мм "
                        f"(расчёт не проверен на реальном дефекте)")
                if clear < CLEARANCE_MIN_MM:
                    txt += " — похоже, коронке места не хватает"
                    act = ("ортопед: проверить высоту вручную, по расчёту её "
                           "недостаточно для коронки")
                elif clear < CLEARANCE_TIGHT_MM:
                    txt += " — тесно"
            txt += txt_tail
            if per_tooth < IMPLANT_MIN_MM:
                txt += (f"; на зуб приходится {per_tooth:.1f} мм — "
                        f"для импланта обычного диаметра узко")
            out.append(_finding(
                "defect", txt, "prostho", severity=sev, teeth=run,
                value=round(width, 1), unit="мм", confidence="medium",
                action=act))

    # --- 2. зубы без антагониста ---
    lonely = [f for f in sorted(case.teeth)
              if antagonist(f) not in case.teeth]
    if lonely:
        out.append(_finding(
            "no_antagonist",
            f"Нет одноимённого антагониста: {lonely}",
            "prostho", teeth=lonely, confidence="high",
            action="ортопед: оценить, нужна ли пара — зуб без нагрузки "
                   "склонен выдвигаться"))

    # --- 3. выдвижение зуба за линию смыкания ---
    for jaw in ("upper", "lower"):
        prot, rms = occlusal_protrusion(case, jaw)
        if rms > 1.5:            # линия смыкания и так рваная — не судим
            continue
        for f, p in sorted(prot.items(), key=lambda x: -x[1]):
            if p < OVERERUPT_MM:
                continue
            no_pair = antagonist(f) not in case.teeth
            out.append(_finding(
                "over_erupted",
                f"Зуб {f} выступает за линию смыкания на {p:.1f} мм"
                + (" при отсутствующем антагонисте" if no_pair else ""),
                "prostho", severity="warning", teeth=[f], value=round(p, 2),
                unit="мм", confidence="medium" if no_pair else "low",
                action="ортопед: выдвижение мешает протезированию "
                       "противоположной челюсти"))
    return out


# ======================= пародонтолог =======================

# Длина клинической коронки сама по себе мало что говорит: у людей она
# разная. Диагностична АСИММЕТРИЯ — свой же одноимённый зуб служит эталоном.
# Замер на скане клиники: у здоровых пар расхождение 0.0-0.6 мм (16/26: 6.60
# против 6.61), так что порог 1.5 мм заведомо выше разброса метода.
RECESSION_ASYM_MM = 1.5


def findings_perio(case: Case) -> list:
    out = []
    seen = set()
    for r in sorted(case.geom["teeth"], key=lambda x: x["fdi"]):
        f = r["fdi"]
        m = G.mirror_fdi(f)
        pair = tuple(sorted((f, m)))
        if pair in seen or m not in case.by_fdi:
            continue
        seen.add(pair)
        if f in case.bad or m in case.bad:
            continue
        a, b = r["crown_height_mm"], case.by_fdi[m]["crown_height_mm"]
        if a is None or b is None:
            continue
        diff = a - b
        if abs(diff) < RECESSION_ASYM_MM:
            continue
        longer, shorter = (f, m) if diff > 0 else (m, f)
        out.append(_finding(
            "crown_asymmetry",
            f"Коронка зуба {longer} длиннее одноимённого {shorter} на "
            f"{abs(diff):.1f} мм ({max(a, b):.1f} против {min(a, b):.1f})",
            "hygiene", severity="warning", teeth=[f, m],
            value=round(abs(diff), 2), unit="мм", confidence="low",
            action="пародонтолог: похоже на рецессию десны у зуба "
                   f"{longer}; отличить от стирания антагониста и от разницы "
                   "самих зубов можно только осмотром"))
    return out


# ======================= гнатолог: фасетки стирания =======================

FACET_ANGLE_MAX_DEG = 12.0   # разброс нормалей внутри площадки
FACET_MIN_MM2 = 1.0          # мельче — не фасетка, а пятно контакта
FACET_ASYM_MM2 = 4.0         # разница площадей сторон больше — перекос


def wear_facets(case: Case) -> dict:
    """Плоские площадки в зоне смыкания -> {fdi: (площадь мм², число)}.

    Фасетка стирания — это ПЛОСКАЯ площадка там, где зуб бьётся об антагониста.
    Отдельно «плоскость» ничего не значит (у моляра и в норме есть ровные
    участки), отдельно «контакт» тоже; диагностично их совпадение.

    Отличить стирание от анатомии по одному скану всё равно нельзя — для
    этого нужен второй скан спустя время. Поэтому отдаём измерение площадки,
    а не диагноз.
    """
    import trimesh
    path = CASES / case.case_id / "occlusion.json"
    if not path.is_file():
        return {}
    occl = json.loads(path.read_text(encoding="utf-8"))
    thr = occl["thresholds_mm"]["contact"]
    out = {}
    for name, b64 in occl.get("vertex_gap_b64", {}).items():
        fdi = int(name.split("_")[1])
        mesh = case.teeth.get(fdi)
        if mesh is None or fdi in case.bad:
            continue
        import base64
        gap = np.frombuffer(base64.b64decode(b64), dtype=np.uint8) / 100.0
        if len(gap) != len(mesh.vertices):
            continue
        sel = np.flatnonzero(gap[mesh.faces].mean(axis=1) <= thr)
        if len(sel) < 3:
            continue
        sub = mesh.submesh([sel], append=True, repair=False)
        if sub is None or len(sub.faces) < 3:
            continue
        lab = trimesh.graph.connected_component_labels(
            sub.face_adjacency, node_count=len(sub.faces))
        area, count = 0.0, 0
        for l in range(int(lab.max()) + 1):
            m = lab == l
            a = float(sub.area_faces[m].sum())
            if a < FACET_MIN_MM2:
                continue
            n = sub.face_normals[m]
            mean = n.mean(axis=0)
            mean /= np.linalg.norm(mean) + 1e-12
            spread = np.degrees(np.arccos(np.clip(n @ mean, -1, 1)))
            if float(np.sqrt(np.mean(spread ** 2))) <= FACET_ANGLE_MAX_DEG:
                area += a
                count += 1
        if count:
            out[fdi] = (round(area, 1), count)
    return out


def findings_gnatho(case: Case) -> list:
    facets = wear_facets(case)
    if not facets:
        return []
    out = []
    big = sorted(facets.items(), key=lambda kv: -kv[1][0])[:6]
    listed = ", ".join(f"{f} ({a:.0f} мм²)" for f, (a, _) in big)
    out.append(_finding(
        "wear_facets",
        f"Плоские площадки в зоне смыкания: {listed}",
        "gnatho", teeth=[f for f, _ in facets.items()], confidence="low",
        action="гнатолог: похоже на фасетки стирания. По одному скану "
               "отличить стирание от анатомии нельзя — нужен повторный скан "
               "через время либо осмотр"))

    right = sum(a for f, (a, _) in facets.items() if f // 10 in (1, 4))
    left = sum(a for f, (a, _) in facets.items() if f // 10 in (2, 3))
    if abs(right - left) >= FACET_ASYM_MM2:
        heavy = "справа" if right > left else "слева"
        out.append(_finding(
            "wear_asymmetry",
            f"Площадки стирания распределены неравномерно: {right:.0f} мм² "
            f"справа против {left:.0f} мм² слева — больше {heavy}",
            "gnatho", severity="warning", value=round(abs(right - left), 1),
            unit="мм²", confidence="low",
            action="гнатолог: сопоставить с картой контактов — сторона, "
                   "которая жуёт, стирается быстрее"))
    return out


# ======================= прикус =======================
#
# Насколько верить каждой находке — измерено, а не назначено: доля пар
# «больной/здоровый», упорядоченных верно, на 180 пациентах Bits2Bites с
# врачебной разметкой (ios_frame_validate.py). Пишем это прямо врачу в
# основание — иначе он не отличит надёжную находку от слабой.
OCCL_TRUST = {
    "Глубокое резцовое перекрытие": (
        "medium", "на 180 пациентах: находит 93% случаев, ложно тревожит 11%"),
    "Открытый прикус": (
        "medium", "на 180 пациентах: находит 95% случаев, ложно тревожит 14%"),
    "Обратное резцовое перекрытие": (
        "low", "знак сагиттальной щели проверен только на одном скане"),
    "Смещение средней линии": (
        "low", "находит меньше половины случаев (46%), ложно тревожит 18%; "
               "центроидами среднюю линию не взять — нужна разметка "
               "контактных точек"),
    "похоже на II класс": (
        "low", "находит около половины случаев (52-59%) при 90% специфичности; "
               "измерение по центроидам — ориентир, а не диагноз"),
    "похоже на III класс": (
        "low", "находит 44-53% случаев при 89-95% специфичности; "
               "измерение по центроидам — ориентир, а не диагноз"),
    "Перекрёстный прикус": ("low", "врачебной проверки не проходило"),
}


def _trust(name: str) -> tuple:
    for key, val in OCCL_TRUST.items():
        if name.startswith(key):
            return val
    return "low", None


def findings_occlusion(case: Case) -> list:
    """Прикус по кейсу. Сеть заново не гоняем — берём вершины из scan.glb."""
    if not case.geom["frame_reliable"]:
        return [_finding(
            "occlusion_skipped",
            "Прикус не считался: оси дуги ненадёжны (дуга не читается как "
            "плоская)", "ortho", confidence="high",
            action="проверить сегментацию — обычно виноват меш")]
    pts = {f: np.asarray(m.vertices) for f, m in case.teeth.items()
           if f not in case.bad}
    res = G.occlusion_measures(case.frame, pts)
    m = res["measurements"]
    out = []

    parts = []
    for key, label in (("перекрытие_мм", "перекрытие резцов"),
                       ("сагиттальная_щель_мм", "сагиттальная щель"),
                       ("средняя_линия_мм", "средняя линия")):
        if key in m:
            parts.append(f"{label} {m[key]:+.1f} мм")
    for side in ("справа", "слева"):
        if f"класс_{side}" in m:
            parts.append(f"класс {side} — {m[f'класс_{side}']}"
                         f" ({m[f'моляры_{side}_мм']:+.1f} мм)")
    if parts:
        out.append(_finding(
            "occlusion_summary", "Прикус: " + "; ".join(parts), "ortho",
            confidence="medium",
            action="пороги измерены по 180 пациентам с врачебной разметкой и "
                   "проверены на обоих путях расчёта — по сырой сегментации и "
                   "по очищенному мешу; расхождение в пределах нескольких "
                   "процентов. Выбраны на специфичность около 90%: лучше "
                   "пропустить, чем поднять ложную тревогу"))

    for name, value, _who in res["findings"]:
        conf, why = _trust(name)
        out.append(_finding(
            "occlusion", f"{name}: {value}", "ortho", severity="warning",
            confidence=conf, action=("надёжность: " + why) if why else None))
    return out


GROUPS = {"ortho": findings_ortho, "occlusion": findings_occlusion,
          "prostho": findings_prostho, "perio": findings_perio,
          "gnatho": findings_gnatho}


def analyze(case_id: str, groups=None) -> dict:
    case = Case(case_id)
    picked = groups or list(GROUPS)
    findings = []
    for key in picked:
        findings += GROUPS[key](case)
    quality = [
        _finding("mesh_quality",
                 f"Зуб {f}: {'; '.join(why)}", "review", severity="info",
                 teeth=[f], confidence="high",
                 action="измерения по этому зубу — с пониженным доверием")
        for f, why in sorted(case.bad.items())
    ]
    # Линия дуги строится на челюсть: у верхней и нижней своя симметрия,
    # и одна линия на обе была бы средним по двум разным дугам.
    arch_lines = {}
    for jaw in ("upper", "lower"):
        sub = {f: m for f, m in case.teeth.items() if f in case.seq[jaw]}
        if len(sub) >= 6:
            line = G.arch_ideal_line(case.frame, sub, case.bad, jaw=jaw)
            if line.get("points"):
                arch_lines[jaw] = line

    return {
        "case_id": case_id,
        "arch_lines": arch_lines,
        "draft": True,
        "planarity": case.geom["planarity"],
        "frame_reliable": case.geom["frame_reliable"],
        "teeth": case.geom["teeth"],
        "findings": findings,
        "quality": quality,
        "routes": cr.route_list({f["route"] for f in findings + quality}),
    }


def print_report(res: dict) -> None:
    print(f"\n{'=' * 74}\nНАХОДКИ ПО СКАНУ — кейс {res['case_id']}\n{'=' * 74}")
    by_route = {}
    for f in res["findings"]:
        by_route.setdefault(f["route"], []).append(f)
    for r in res["routes"]:
        items = by_route.get(r["key"])
        if not items:
            continue
        print(f"\n[{r['order']}] {r['label']}")
        for f in sorted(items, key=lambda x: x["severity"] != "warning"):
            mark = " !" if f["severity"] == "warning" else "  "
            conf = "" if f["confidence"] == "high" else f"   (доверие: {f['confidence']})"
            print(f"  {mark} {f['text']}{conf}")
            if f["action"]:
                print(f"       -> {f['action']}")
    if res["quality"]:
        print(f"\n{'-' * 74}\nКАЧЕСТВО СЕГМЕНТАЦИИ")
        for q in res["quality"]:
            print(f"   • {q['text']}")
    print(f"\n{'-' * 74}")
    print("ЧЕРНОВИК. Пороги не подтверждены врачом; не является назначением.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--group", action="append", choices=sorted(GROUPS),
                    help="считать только эту группу (можно несколько раз)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = analyze(args.case_id, args.group)
    print_report(res)
    out = Path(args.json) if args.json else CASES / args.case_id / "findings.json"
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"сохранено: {out}  ({out.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
