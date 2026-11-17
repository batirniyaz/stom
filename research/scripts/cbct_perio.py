"""Уровень костной ткани по КЛКТ: расстояние от шейки зуба до гребня кости.

Это то, чем измеряется убыль кости при пародонтите. Мы держали эту задачу
заблокированной («нужна разметка рентгенолога»), но разметка нужна не была:
у конкурента это ИЗМЕРЕНИЕ, а не классификация, и обе точки отсчёта читаются
по маскам, которые у нас уже есть.

    шейка (CEJ)   — граница эмали. Эмаль плотнее дентина, поэтому по внешнему
                    слою зуба идёт заметный перепад плотности: замер на
                    client4 даёт медиану 842 HU между коронкой и корнем.
    гребень кости — самый коронковый уровень, где зуб ещё окружён костью.

Меряем ОТДЕЛЬНО с мезиальной и дистальной стороны: убыль кости почти всегда
локальная, и усреднение по зубу её прячет.

ВАЖНО ПРО ШКАЛУ. Мы отдаём миллиметры И ПЕРЦЕНТИЛЬ в референсной выборке,
но НЕ ставим клиническую степень пародонтита. Причина в замере: на 60
пациентах (2528 участков) медиана нашего измерения 3.9 мм, а литературный
порог нормы — 2 мм, и его превышают 93% участков. Мы меряем не то же, что
пародонтолог зондом, и разница систематическая. Подробности ниже у REF_MM.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_perio.py client4
    .venv/Scripts/python.exe research/scripts/cbct_perio.py client4 --model tf3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

JAW_CLASSES = (1, 2)

# --- ПОРОГИ: перцентиль в референсной выборке, а не миллиметры из книги ---
#
# Абсолютные пороги 2/4/6 мм взяты были из литературы и ОКАЗАЛИСЬ НЕПРИМЕНИМЫ.
# Замер на 60 пациентах ToothFairy по эталонным маскам (2528 участков):
#
#     перцентиль   5%   25%   50%   75%   85%   90%   95%   99%
#     мм          1.9   3.0   3.9   5.0   5.8   6.5   7.3   9.0
#
# Порог «норма <= 2 мм» превышают 93% участков. Это не значит, что больны 93%
# людей: мы меряем не то, что пародонтолог зондом. Наша шейка — точка
# наибольшего спада плотности по внешнему слою, наш гребень — 95-й перцентиль
# уровня кости в секторе; обе определены геометрически, и между ними
# систематически примерно на 1.9 мм больше, чем между клиническими точками.
#
# Это третий случай подряд с той же природой: у класса по Энглю норма −1.95 мм,
# у индекса Литтла 1.2 мм на пару, здесь 3.9 мм. ПРЕЖДЕ ЧЕМ БРАТЬ ПОРОГ ИЗ
# ЛИТЕРАТУРЫ, ПРОВЕРЬ БАЗУ СОБСТВЕННОГО ИЗМЕРЕНИЯ.
#
# Поэтому отдаём перцентиль: «выше 97% участков выборки» — утверждение о нашем
# измерении, которое можно проверить. «8 мм = тяжёлый пародонтит» — утверждение
# о болезни, и на него у нас прав нет: разметки пародонтолога нет, а выборка
# ToothFairy состоит из пациентов КЛКТ, то есть сама не здорова.
REF_PCTL = [5, 25, 50, 75, 85, 90, 95, 99]
REF_MM = [1.9, 3.0, 3.9, 5.0, 5.8, 6.5, 7.3, 9.0]
REF_N_SITES = 2528          # на скольких участках снята выборка
REF_N_PATIENTS = 60

# Границы категорий В ПЕРЦЕНТИЛЯХ. Специфичность выбрана как на прикусе:
# тревожить примерно на 15% участков, а не на 93%.
P_MILD = 85.0
P_MODERATE = 95.0
P_SEVERE = 99.0

# Оставлены для совместимости и как справка о клинической шкале; в решениях
# НЕ участвуют (см. выше, почему).
NORMAL_MM = 2.0
MILD_MM = 4.0
MODERATE_MM = 6.0


def percentile_of(mm: float) -> float:
    """Где measured-значение стоит в референсной выборке, в процентах."""
    return float(np.interp(mm, REF_MM, REF_PCTL,
                           left=REF_PCTL[0], right=100.0))

# Доля длины корня — то, чем стадируют пародонтит по классификации 2017 года.
STAGE_I_FRAC = 0.15    # до 15% длины корня
STAGE_II_FRAC = 0.33   # 15-33%; больше — III/IV стадия

MIN_TOOTH_VOX = 300
MIN_ENAMEL_GAP = 400   # меньший перепад — коронка закрыта, эмали не видно
SITE_ANGLE_DEG = 50    # полураствор сектора «мезиально» / «дистально»
MIN_SITE_VOX = 15      # меньше вокселей кости в секторе — не мерим
SHELL_ERODE = 2        # толщина внешнего слоя зуба для профиля плотности
SMOOTH_WIN = 5
TOLERANCE_MM = 0.5     # гребень выше шейки в пределах шума — считаем нулём


def tooth_axis(idx: np.ndarray, spacing: np.ndarray, upper: bool,
               z_up: bool = True) -> tuple:
    """Длинная ось зуба и её начало, в мм. Ось направлена К КОРОНКЕ.

    Берём главную ось самого зуба, а не общую вертикаль: зубы наклонены, и
    у наклонённого моляра измерение вдоль общей оси завышает высоту.

    `z_up` — растёт ли ось срезов вверх (`cb.occlusal_up`). Раньше это было
    ВШИТЫМ допущением, верным для наших четырёх клинических кейсов с одного
    аппарата. На ToothFairy ориентация обратная, ось смотрела от коронки к
    апексу, и уровень кости не измерялся ВООБЩЕ (0 зубов из 22). Определять
    по данным, а не предполагать.
    """
    pts = idx.T * spacing                       # (N, 3) в мм, порядок (z, y, x)
    c = pts.mean(0)
    # full_matrices=False обязателен. По умолчанию numpy считает матрицу U
    # размера N x N, где N — число ВОКСЕЛЕЙ зуба, хотя нужен только первый из
    # трёх правых сингулярных векторов. Замер: 5 тыс. вокселей — 0.21 с,
    # 30 тыс. — 9.2 с, 120 тыс. — MemoryError, тогда как экономная версия
    # везде укладывается в 0.001 с и даёт ТУ ЖЕ ось (расхождение 1e-6 град).
    # Именно из-за этого слой находок на кейсе с вокселем 0.2 мм (client3,
    # вокселей на зуб втрое больше) считался больше сорока минут против пяти
    # на обычных кейсах: 180 с из 466 уходило в этот вызов.
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    a = Vt[0]
    # Куда должна смотреть ось: при z вверх у верхнего зуба коронка снизу,
    # у нижнего сверху; при обратной ориентации всё зеркалится.
    want_positive = upper != z_up
    if (a[0] > 0) != want_positive:
        a = -a
    return a / np.linalg.norm(a), c


def cej_level(vol, bm, idx, axis, origin, spacing):
    """Уровень шейки вдоль оси зуба, мм. None — эмаль не читается.

    Профиль плотности строим по ВНЕШНЕМУ слою: пломбировочный материал в
    канале плотнее эмали и по всему сечению забивает сигнал (на зубах 21 и 47
    детектор из-за этого уезжал к апексу).
    """
    pts = idx.T * spacing
    proj = (pts - origin) @ axis                # мм вдоль оси, больше — к коронке
    lo, hi = proj.min(), proj.max()
    n = max(int((hi - lo) / spacing[0]), 12)
    if n < 12:
        return None, None
    edges = np.linspace(lo, hi, n + 1)
    # внешний слой — тоже по кропу, иначе эрозия идёт по всему объёму
    lo3 = [max(int(i.min()) - 2, 0) for i in idx]
    hi3 = [min(int(i.max()) + 3, s) for i, s in zip(idx, bm.shape)]
    box3 = tuple(slice(a, b) for a, b in zip(lo3, hi3))
    sub3 = bm[box3]
    shell = sub3 & ~ndi.binary_erosion(sub3, iterations=SHELL_ERODE)
    sh_idx = np.array(np.where(shell))
    if sh_idx.shape[1] < 50:
        return None, None
    sh_idx = sh_idx + np.array(lo3)[:, None]
    sh_proj = (sh_idx.T * spacing - origin) @ axis
    sh_val = vol[tuple(sh_idx)]

    prof = np.full(n, np.nan)
    for i in range(n):
        m = (sh_proj >= edges[i]) & (sh_proj < edges[i + 1])
        if m.sum() >= 8:
            prof[i] = float(np.median(sh_val[m]))
    ok = ~np.isnan(prof)
    if ok.mean() < 0.7:
        return None, None
    prof = np.interp(np.arange(n), np.where(ok)[0], prof[ok])

    crown = float(np.median(prof[int(n * 0.75):]))
    root = float(np.median(prof[:int(n * 0.35)]))
    if crown - root < MIN_ENAMEL_GAP:
        return None, None
    sm = np.convolve(prof, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")
    a, b = int(n * 0.25), int(n * 0.9)
    if b <= a + 3:
        return None, None
    k = a + int(np.argmax(np.gradient(sm)[a:b]))
    return float(edges[k]), float(lo)           # уровень шейки и апекса, мм


def site_dirs(centroids: dict, tooth: int, axis: np.ndarray):
    """Направление «мезиально» в плоскости, поперечной оси зуба.

    Сосед по номеру надёжнее ближайшего по расстоянию: при отсутствующем зубе
    ближайшим может оказаться зуб через промежуток. Но у крайних в ряду
    соседа по номеру может не быть вовсе — тогда берём ближайший зуб того же
    квадранта, иначе восьмёрки и семёрки выпадают из измерения.
    """
    q, i = tooth // 10, tooth % 10
    for cand, sign in ((q * 10 + i - 1, 1.0), (q * 10 + i + 1, -1.0)):
        if cand in centroids:
            d = (centroids[cand] - centroids[tooth]) * sign
            d = d - axis * (d @ axis)
            n = np.linalg.norm(d)
            if n > 1e-6:
                return d / n
    same_q = [(np.linalg.norm(centroids[o] - centroids[tooth]), o)
              for o in centroids if o // 10 == q and o != tooth]
    if same_q:
        _, nb = min(same_q)
        sign = 1.0 if nb % 10 < i else -1.0
        d = (centroids[nb] - centroids[tooth]) * sign
        d = d - axis * (d @ axis)
        n = np.linalg.norm(d)
        if n > 1e-6:
            return d / n
    return None


def crest_levels(mask, bm, tooth, axis, origin, mesial, spacing):
    """Уровень гребня кости с мезиальной и дистальной стороны, мм вдоль оси.

    Всё считаем в кропе по зубу: расширение маски по полному объёму
    (577x516x516) на каждый из 28 зубов — это минуты на кейс.
    """
    idx = np.where(bm)
    lo = [max(int(i.min()) - 3, 0) for i in idx]
    hi = [min(int(i.max()) + 4, s) for i, s in zip(idx, bm.shape)]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = bm[box]
    ring = ndi.binary_dilation(sub, iterations=2) & ~sub
    bone = ring & np.isin(mask[box], JAW_CLASSES)
    bi = np.array(np.where(bone))
    if bi.size:
        bi = bi + np.array(lo)[:, None]      # обратно в координаты объёма
    if bi.shape[1] < MIN_SITE_VOX:
        return {}
    pts = bi.T * spacing
    rel = pts - origin
    along = rel @ axis
    lateral = rel - np.outer(along, axis)
    norm = np.linalg.norm(lateral, axis=1)
    good = norm > 1e-6
    cosang = np.zeros(len(pts))
    cosang[good] = (lateral[good] @ mesial) / norm[good]
    lim = np.cos(np.radians(SITE_ANGLE_DEG))

    out = {}
    for name, sel in (("mesial", cosang >= lim), ("distal", cosang <= -lim)):
        if sel.sum() >= MIN_SITE_VOX:
            # гребень — самый коронковый уровень кости в секторе; берём 95-й
            # перцентиль, чтобы одинокий воксель не задрал результат
            out[name] = float(np.percentile(along[sel], 95))
    return out


def severity(mm: float, root_len: float) -> tuple:
    """(степень, доля длины корня, перцентиль) по расстоянию шейка-гребень.

    Степень названа ОТ ВЫБОРКИ, а не от клинической шкалы: «выше нормы» —
    утверждение о нашем измерении, которое проверяется; «тяжёлый пародонтит» —
    утверждение о болезни, и на него у нас прав нет.
    """
    frac = mm / root_len if root_len and root_len > 1 else float("nan")
    pct = percentile_of(mm)
    if pct < P_MILD:
        return "норма", frac, pct
    if pct < P_MODERATE:
        return "выше нормы", frac, pct
    if pct < P_SEVERE:
        return "заметно выше нормы", frac, pct
    return "резко выше нормы", frac, pct


def analyze_case(c: dict, model: str) -> dict:
    """Уровень кости по кейсу демо."""
    vol, mask = cb.arrays_for(c, model)
    return analyze_arrays(vol, mask, np.array(c["spacing"], float))


def analyze_arrays(vol, mask, spacing) -> dict:
    """То же, но по готовым массивам.

    Отдельным входом, чтобы измерение можно было прогнать на эталонных масках
    ToothFairy (532 тома с разметкой) и отделить ошибку ИЗМЕРЕНИЯ от ошибки
    сегментации. Пороги калибровались на четырёх клинических кейсах, и без
    такой проверки неизвестно, что из этого перенесётся на нового пациента.
    """
    z_up = cb.occlusal_up(mask)     # по данным, а не по допущению
    present = [t for t in cb.TEETH_CLASSES if (mask == t).sum() >= MIN_TOOTH_VOX]
    centroids = {t: (np.array(np.where(mask == t)).T * spacing).mean(0)
                 for t in present}

    rows, skipped = [], []
    for t in present:
        bm = mask == t
        idx = np.array(np.where(bm))
        upper = t // 10 in (1, 2)
        axis, origin = tooth_axis(idx, spacing, upper, z_up)
        cej, apex = cej_level(vol, bm, idx, axis, origin, spacing)
        if cej is None:
            skipped.append(t)
            continue
        mesial = site_dirs(centroids, t, axis)
        if mesial is None:
            skipped.append(t)
            continue
        crest = crest_levels(mask, bm, t, axis, origin, mesial, spacing)
        if not crest:
            skipped.append(t)
            continue
        root_len = cej - apex
        row = {"tooth": t, "root_len_mm": round(root_len, 1), "sites": {}}
        for site, lvl in crest.items():
            mm = cej - lvl
            # Гребень выше шейки анатомически невозможен. Раньше здесь стоял
            # max(mm, 0), и сбой измерения выдавал себя за идеальную норму —
            # врач увидел бы 0.0 мм там, где расчёт развалился.
            if mm < -TOLERANCE_MM:
                continue
            mm = max(mm, 0.0)
            sev, frac, pct = severity(mm, root_len)
            row["sites"][site] = {
                "mm": round(mm, 1),
                "root_fraction": None if frac != frac else round(frac, 3),
                "percentile": round(pct),
                "severity": sev,
            }
        if row["sites"]:
            rows.append(row)
        else:
            skipped.append(t)
    return {"teeth": rows, "not_measured": skipped,
            "reference": {"n_sites": REF_N_SITES, "n_patients": REF_N_PATIENTS,
                          "median_mm": REF_MM[REF_PCTL.index(50)],
                          "pctl": REF_PCTL, "mm": REF_MM},
            "cutoffs_pctl": {"mild": P_MILD, "moderate": P_MODERATE,
                             "severe": P_SEVERE},
            "draft": True}


SITE_RU = {"mesial": "мезиально", "distal": "дистально"}


def build_findings(res: dict) -> list:
    """Находки пародонтолога. Формат общий с cbct_findings."""
    out = []
    worst = {}
    for row in res["teeth"]:
        for site, v in row["sites"].items():
            if v["severity"] == "норма":
                continue
            worst.setdefault(v["severity"], []).append(
                (row["tooth"], site, v["mm"], v["root_fraction"]))

    order = ["резко выше нормы", "заметно выше нормы", "выше нормы"]
    sev_map = {"резко выше нормы": "alert", "заметно выше нормы": "warning",
               "выше нормы": "warning"}
    for name in order:
        items = worst.get(name)
        if not items:
            continue
        listed = ", ".join(
            f"{t} {SITE_RU[s]} {mm:.1f} мм"
            + (f" ({fr*100:.0f}% корня)" if fr else "")
            for t, s, mm, fr in sorted(items))
        worst_pct = percentile_of(max(mm for _t, _s, mm, _fr in items))
        out.append({
            "code": "bone_loss", "tooth": None,
            "teeth": sorted({t for t, _s, _mm, _fr in items}),
            "value": round(max(mm for _t, _s, mm, _fr in items), 1),
            "unit": "мм", "grade": name,
            "severity": sev_map[name],
            "text": (f"Уровень кости {name}: {listed}. Худший участок выше "
                     f"{worst_pct:.0f}% участков в выборке из "
                     f"{REF_N_PATIENTS} пациентов (медиана выборки "
                     f"{REF_MM[REF_PCTL.index(50)]:.1f} мм — это база НАШЕГО "
                     f"измерения, не клиническая норма)"),
            "action": "пародонтолог: подтвердить зондированием; клиническую "
                      "степень по КЛКТ мы не ставим — нужна его разметка",
            "route": "hygiene", "confidence": "medium",
            # Числа, по которым находка возникла: врач должен иметь
            # возможность проверить их на срезе, а не верить проценту.
            "basis": [
                {"label": "худший участок шейка → гребень",
                 "value": round(max(mm for _t, _s, mm, _fr in items), 1),
                 "unit": "мм",
                 "rule": f"выше нормы с {P_MILD:.0f}-го перцентиля выборки, "
                         f"заметно с {P_MODERATE:.0f}-го, резко с "
                         f"{P_SEVERE:.0f}-го",
                 "note": f"медиана выборки {REF_MM[REF_PCTL.index(50)]:.1f} мм "
                         f"— база нашего измерения, не клиническая норма"},
                {"label": "перцентиль в выборке", "value": round(worst_pct),
                 "unit": "%",
                 "rule": f"выборка {REF_N_SITES} участков, "
                         f"{REF_N_PATIENTS} пациентов ToothFairy",
                 "note": "выборка из пациентов КЛКТ — она сама не здорова"},
                {"label": "участков с превышением", "value": len(items),
                 "unit": "шт.", "rule": "мезиально и дистально у каждого зуба"},
            ],
        })
    if res["not_measured"]:
        out.append({
            "code": "bone_not_measured", "tooth": None,
            "teeth": sorted(res["not_measured"]), "value": None,
            "unit": None, "severity": "info",
            "text": f"Уровень кости не измерен у зубов "
                    f"{sorted(res['not_measured'])}: не читается граница эмали "
                    f"(коронка, реставрация) либо рядом нет кости",
            "action": "оценить по срезам вручную",
            "route": "review", "confidence": "high",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    c = cb.get_case(args.case_id)
    if c is None:
        raise SystemExit(f"кейс {args.case_id} не найден")
    model = args.model or cb.mask_model_for(c)
    if model is None:
        raise SystemExit("у кейса нет готовой маски")

    res = analyze_case(c, model)
    print(f"\n{'='*66}\nУРОВЕНЬ КОСТНОЙ ТКАНИ — {args.case_id} / {model}\n{'='*66}")
    print(f"{'зуб':>4} {'корень':>8} {'мезиально':>22} {'дистально':>22}")
    for row in res["teeth"]:
        cells = []
        for site in ("mesial", "distal"):
            v = row["sites"].get(site)
            cells.append("—".rjust(22) if not v else
                         f"{v['mm']:5.1f} мм  {v['severity']:<12}".rjust(22))
        print(f"{row['tooth']:>4} {row['root_len_mm']:7.1f} {cells[0]} {cells[1]}")
    if res["not_measured"]:
        print(f"\nне измерено: {sorted(res['not_measured'])}")
    print(f"\n{'-'*66}\nНАХОДКИ")
    for f in build_findings(res):
        print(f"  {'!' if f['severity'] != 'info' else ' '} {f['text']}")
    print(f"\n{'-'*66}\nЧЕРНОВИК. Пороги из литературы, врачом не подтверждены.")


if __name__ == "__main__":
    main()
