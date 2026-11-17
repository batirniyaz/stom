"""Геометрия зубной дуги по сегментированному скану: оси, соседи, размеры зуба.

Общий фундамент для всех находок по интраоральному скану. Модуль намеренно
лёгкий: только numpy/scipy/trimesh, без torch и без сети — сегментация к этому
моменту уже сделана и лежит в scan.glb.

Главное здесь — как определяется ВЕРТИКАЛЬ, потому что от неё зависят и высота
коронки, и перекрытие резцов, и всё остальное. Способ выбран не на глаз:
эталоном служит нормаль окклюзионной плоскости (её задают точки, в которых
зубы реально смыкаются, — это анатомическое определение, а не наше допущение).
Замер на скане клиники:

    нормаль плоскости смыкания           отклонение
    разница центров верх/низ                28.8°     плечо всего 4 мм — шум
    от десны к коронкам                     17.4°     зависит от того,
                                                      сколько нёба попало в скан
    PCA центров зубов                        0.8°     <- берём это

Дуга почти плоская (разброс поперёк плоскости к разбросу вдоль неё 0.03–0.04
на каждой челюсти), поэтому перпендикуляр к ней определяется устойчиво, а
нормали, посчитанные по верхней и нижней дуге отдельно, сходятся в пределах 5°.

Раньше `ios_occlusion.build_frame` считал вертикаль первым способом, и это
оказалось причиной провала измерений прикуса. Проверено на 180 пациентах
Bits2Bites: разделение врачебных категорий выросло с 0.573 до 0.971 по
глубокому прикусу и с 0.56-0.65 до 0.74-0.83 по классу Энгля. Теперь тот
модуль берёт оси отсюда — второй реализации быть не должно.

Запуск (печатает всё, что умеет, по кейсу):
    .venv/Scripts/python.exe research/scripts/ios_geometry.py batir
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

CASES = Path(__file__).resolve().parents[2] / "research_ios" / "web_cases"

MIN_TOOTH_FACES = 120
NECK_TOL_MM = 1e-6      # вершины зуба и десны на линии шейки буквально общие
MIN_NECK_VERTS = 10     # меньше — шейку считать не по чему
# Дуга должна быть плоской, иначе перпендикуляр к ней не определён. Замер по
# клиническому скану: 0.029 (верх) и 0.039 (низ). Порог с запасом втрое.
PLANARITY_MAX = 0.12

UPPER, LOWER = (1, 2), (3, 4)


def is_upper(fdi: int) -> bool:
    return fdi // 10 in UPPER


def load_case(case_id: str) -> tuple[dict, dict]:
    """-> ({fdi: Trimesh}, {'upper'|'lower': Trimesh десны})"""
    path = CASES / case_id / "scan.glb"
    if not path.is_file():
        raise SystemExit(f"нет {path}")
    scene = trimesh.load(str(path), process=False)
    teeth, gums = {}, {}
    for name, geom in scene.geometry.items():
        short = name.split("/")[-1]
        if short.startswith("tooth_") and len(geom.faces) >= MIN_TOOTH_FACES:
            teeth[int(short.split("_")[1])] = geom
        elif short.startswith("gum_"):
            gums[short.split("_")[1]] = geom
    return teeth, gums


class ArchFrame:
    """Оси дуги в миллиметрах: вверх, влево (к левой стороне пациента), вперёд.

    Каждая ось получена из данных и отдельно проверена на анатомический знак,
    а не унаследована от предыдущей через векторное произведение: так ошибка
    в одной оси не расползается на остальные.
    """

    def __init__(self, up, lat, fwd, planarity, centroids):
        self.up, self.lat, self.fwd = up, lat, fwd
        self.planarity = planarity          # разброс поперёк дуги / вдоль неё
        self.centroids = centroids
        self.reliable = planarity <= PLANARITY_MAX

    def project(self, pts):
        """Точки -> координаты (поперёк, вперёд, вверх) в мм."""
        p = np.asarray(pts, dtype=np.float64)
        return np.stack([p @ self.lat, p @ self.fwd, p @ self.up], axis=-1)


def arch_frame(teeth: dict) -> ArchFrame:
    """Оси дуги по мешам зубов из scan.glb."""
    return arch_frame_from_centroids(
        {t: np.asarray(m.vertices).mean(0) for t, m in teeth.items()})


def arch_frame_from_centroids(cen: dict) -> ArchFrame:
    """Оси дуги по одним центрам зубов {fdi: точка}.

    Отдельный вход нужен там, где мешей на зуб нет, а есть облако точек с
    метками — например при проверке на Bits2Bites. Реализация одна на оба
    случая: разводить построение осей по двум местам нельзя, вертикаль и так
    уже посчитана по-разному в `ios_occlusion.build_frame` (ошибка 28.8°).
    """
    if len(cen) < 8:
        raise SystemExit("слишком мало зубов, чтобы построить оси дуги")
    P = np.array(list(cen.values()))

    # вверх: перпендикуляр к плоскости дуги (см. докстринг модуля)
    # full_matrices=False: P — облако из десятков тысяч точек, полная
    # матрица U была бы N x N (см. cbct_perio.tooth_axis).
    _, sv, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    up = Vt[2]
    planarity = float(sv[2] / sv[0])
    hi = [c for t, c in cen.items() if is_upper(t)]
    lo = [c for t, c in cen.items() if not is_upper(t)]
    if hi and lo and (np.mean(hi, axis=0) - np.mean(lo, axis=0)) @ up < 0:
        up = -up                      # знак: верхние зубы обязаны быть выше
    up /= np.linalg.norm(up)

    # влево: от правых квадрантов (1, 4) к левым (2, 3)
    right = [c for t, c in cen.items() if t // 10 in (1, 4)]
    left = [c for t, c in cen.items() if t // 10 in (2, 3)]
    lat = np.mean(left, axis=0) - np.mean(right, axis=0)
    lat -= up * (lat @ up)
    lat /= np.linalg.norm(lat)

    # вперёд: поперёк первых двух, знак — по резцам против моляров
    fwd = np.cross(lat, up)
    fwd /= np.linalg.norm(fwd)
    inc = [c for t, c in cen.items() if t % 10 <= 2]
    mol = [c for t, c in cen.items() if t % 10 >= 6]
    if inc and mol and (np.mean(inc, axis=0) - np.mean(mol, axis=0)) @ fwd < 0:
        fwd = -fwd
    return ArchFrame(up, lat, fwd, planarity, cen)


def arch_sequence(teeth: dict) -> dict:
    """Порядок зубов вдоль каждой дуги: от последнего моляра справа к
    последнему слева. Соседство берём по FDI, а не по расстоянию: при
    отсутствующем зубе сосед по номеру остаётся тем же, а ближайший — нет."""
    out = {}
    for jaw, quads in (("upper", (1, 2)), ("lower", (4, 3))):
        seq = ([t for t in sorted((x for x in teeth if x // 10 == quads[0]),
                                  reverse=True)] +
               [t for t in sorted(x for x in teeth if x // 10 == quads[1])])
        out[jaw] = seq
    return out


def neck_mask(mesh, gum_tree) -> np.ndarray:
    """Вершины зуба на линии шейки — те, что общие с десной.

    ios_build_mesh режет зуб и десну из одной упрощённой поверхности, поэтому
    на линии среза вершины совпадают точно, а не приблизительно (замер: у зуба
    11 таких 149 из 255 граничных). Граница с соседним зубом сюда не попадает —
    она общая с ним, а не с десной.
    """
    d, _ = gum_tree.query(np.asarray(mesh.vertices), workers=-1)
    return d < NECK_TOL_MM


def tooth_axes(fdi: int, frame: ArchFrame, seq: dict) -> tuple:
    """Локальные оси зуба: (вдоль дуги к соседу сзади, наружу, к смыканию).

    Направление вдоль дуги берём по соседям, а не по форме самого зуба:
    форма у моляра почти симметрична, и главная ось у него не мезио-дистальная.
    """
    # Ось «вдоль дуги» направлена МЕЗИАЛЬНО, то есть к средней линии: к зубу
    # с номером на единицу меньше в том же квадранте. Брать «следующего в
    # ряду» нельзя — ряд идёт справа налево, и для левых квадрантов это дало
    # бы обратный знак, а таблица положений ориентиров (LANDMARK_POS) учена
    # строго в мезиальном направлении.
    q, idx = fdi // 10, fdi % 10
    here = frame.centroids[fdi]
    along = None
    for cand, sign in ((q * 10 + idx - 1, 1.0), (q * 10 + idx + 1, -1.0)):
        if cand in frame.centroids:
            along = (frame.centroids[cand] - here) * sign
            break
    if along is None:            # одинокий зуб в квадранте — берём по ряду
        jaw = "upper" if is_upper(fdi) else "lower"
        row = seq[jaw]
        i = row.index(fdi)
        nb = row[i + 1] if i + 1 < len(row) else row[i - 1]
        along = frame.centroids[nb] - here
    along -= frame.up * (along @ frame.up)          # держим в плоскости дуги
    along /= np.linalg.norm(along)

    # наружу (щёчно): от центра дуги к зубу, поперёк направления вдоль дуги
    arch_center = np.mean(list(frame.centroids.values()), axis=0)
    out = here - arch_center
    out -= frame.up * (out @ frame.up)
    out -= along * (out @ along)
    n = np.linalg.norm(out)
    out = out / n if n > 1e-9 else np.cross(along, frame.up)

    # к смыканию: у верхнего зуба вниз, у нижнего вверх
    toward = -frame.up if is_upper(fdi) else frame.up
    return along, out, toward


def contact_points(fdi: int, teeth: dict, seq: dict) -> list:
    """Точки касания зуба с соседями по дуге: [мезиальная, дистальная].

    Ближайшая к соседу точка на поверхности зуба. У крайнего в ряду зуба
    сосед один, и вторая точка отсутствует.
    """
    jaw = "upper" if is_upper(fdi) else "lower"
    row = seq[jaw]
    i = row.index(fdi)
    V = np.asarray(teeth[fdi].vertices)
    pts = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(row):
            d, _ = cKDTree(np.asarray(teeth[row[j]].vertices)).query(V, workers=-1)
            pts.append(V[int(np.argmin(d))])
    return pts


def tooth_metrics(fdi: int, mesh, frame: ArchFrame, seq: dict, teeth: dict,
                  gum_tree) -> dict:
    """Размеры одного зуба. Всё в миллиметрах."""
    V = np.asarray(mesh.vertices)
    along, out, toward = tooth_axes(fdi, frame, seq)
    neck = neck_mask(mesh, gum_tree)

    h = V @ toward                     # чем больше, тем ближе к смыканию
    tip_level = float(h.max())
    if neck.sum() >= MIN_NECK_VERTS:
        # Высота клинической коронки — от САМОЙ АПИКАЛЬНОЙ точки края десны
        # (это её зенит на щёчной стороне; межзубной сосочек лежит выше и
        # минимумом не бывает) до режущего края. Сверка с нормой: зуб 11 даёт
        # 10.6 мм при 9–11, клыки 9.7 и 9.4 при ~10. Медиана по всей шейке
        # занижает на 3 мм и клинического смысла не имеет.
        crown = tip_level - float(h[neck].min())
        crown_med = tip_level - float(np.median(h[neck]))
    else:
        crown = crown_med = float("nan")

    # Мезио-дистальный размер — между контактными точками, как меряет врач.
    # Габарит вдоль дуги для этого не годится: сегментация прицепляет к зубу
    # посторонние лоскуты, и у 42-45 габарит выходил вдвое больше самого зуба
    # (44: габарит 14.0 мм против 7.1 по контактам при норме 7.1).
    cps = contact_points(fdi, teeth, seq)
    md_contact = (float(np.linalg.norm(cps[0] - cps[1])) if len(cps) == 2
                  else float("nan"))
    extent = float((V @ along).max() - (V @ along).min())
    bl = V @ out
    nan = lambda x: None if x != x else round(x, 2)
    return {
        "fdi": fdi,
        "crown_height_mm": nan(crown),
        "crown_height_median_mm": nan(crown_med),
        "mesiodistal_mm": nan(md_contact),
        "arch_extent_mm": round(extent, 2),
        # >1 означает, что меш длиннее самого зуба: признак посторонних кусков
        "mesh_bloat": (nan(extent / md_contact) if md_contact == md_contact
                       else None),
        "buccolingual_mm": round(float(bl.max() - bl.min()), 2),
        "neck_verts": int(neck.sum()),
        "surface_mm2": round(float(mesh.area), 1),
    }


def mirror_fdi(fdi: int) -> int:
    """Одноимённый зуб противоположной стороны: 16 <-> 26, 44 <-> 34."""
    return {1: 2, 2: 1, 3: 4, 4: 3}[fdi // 10] * 10 + fdi % 10


# Пороги качества меша зуба. Измерения по зубу, не прошедшему их, отдаются
# с пониженным доверием — как маска канала в находках по КЛКТ.
BLOAT_MAX = 1.40        # меш длиннее зуба: посторонние лоскуты (42-45 у batir)
MIN_BL_MM = 3.0         # тоньше — это не зуб, а обрывок (41 у batir: 1.4 мм)
MIN_MIRROR_RATIO = 0.40  # площадь меньше 40% от парного зуба — недосегментирован


def arch_symmetry(frame: "ArchFrame", teeth: dict, bad: dict | None = None):
    """Насколько правая половина дуги совпадает с зеркальной левой.

    Зачем. Рентгенолог показал на скане то, что мы не считали: у здоровой
    челюсти ряд симметричен относительно средней линии, и врач смотрит именно
    на это. «Неровность ряда», которую мы уже меряем, отвечает на другой
    вопрос — насколько зуб выбился из ПЛАВНОЙ линии; ряд может быть гладким и
    при этом перекошенным влево целиком.

    Как считаем. Средняя линия — плоскость через центр дуги перпендикулярно
    оси «влево». Центр левого зуба отражаем в неё и меряем, насколько он
    разошёлся с одноимённым правым. Раскладываем расхождение на две части,
    потому что клинически это разные вещи:
      * ПОПЕРЁК (вдоль дуги, координата «влево») — один зуб стоит дальше от
        средней линии, чем парный: перекос ряда;
      * ВПЕРЁД — один зуб выдвинут относительно парного: разворот дуги.

    Меряем по ЦЕНТРАМ зубов, а не по контактным точкам: центр усредняет тысячи
    вершин и потому устойчив к неточностям сегментации — та же причина, по
    которой класс по Энглю у нас считается по центроидам, а не по ориентирам.

    Зубы с сомнительным мешем в расчёт не берём: перекос по раздутому зубу —
    это измерение нашей ошибки, а не пациента.
    """
    bad = bad or {}
    cen = {t: np.asarray(m.vertices).mean(0) for t, m in teeth.items()}
    if len(cen) < 6:
        return {"pairs": [], "reliable": False,
                "note": "мало зубов для оценки симметрии"}
    # Средняя линия: центр по одноимённым парам, а не по всем зубам —
    # отсутствующий зуб на одной стороне иначе утянул бы её вбок.
    mids = []
    for t in cen:
        m = mirror_fdi(t)
        if m in cen:
            mids.append((frame.project(cen[t])[0] + frame.project(cen[m])[0]) / 2)
    if not mids:
        return {"pairs": [], "reliable": False, "note": "нет одноимённых пар"}
    mid_lat = float(np.median(mids))

    pairs = []
    for t in sorted(cen):
        if t // 10 not in (1, 4):        # идём по правой стороне, парную ищем
            continue
        m = mirror_fdi(t)
        if m not in cen:
            continue
        a = frame.project(cen[t])
        b = frame.project(cen[m])
        # отражаем левый зуб через среднюю линию
        b_mirrored_lat = 2 * mid_lat - b[0]
        d_lat = float(a[0] - b_mirrored_lat)
        d_fwd = float(a[1] - b[1])
        row = {
            "right": int(t), "left": int(m),
            "d_lat_mm": round(d_lat, 2),
            "d_fwd_mm": round(d_fwd, 2),
            "total_mm": round(float(np.hypot(d_lat, d_fwd)), 2),
            "reliable": not (t in bad or m in bad),
        }
        pairs.append(row)
    good = [p for p in pairs if p["reliable"]]
    return {
        "pairs": pairs,
        "midline_lat": round(mid_lat, 2),
        "worst": max(good, key=lambda p: abs(p["total_mm"])) if good else None,
        "median_mm": (round(float(np.median([abs(p["total_mm"]) for p in good])), 2)
                      if good else None),
        "reliable": bool(good) and frame.reliable,
    }


def arch_ideal_line(frame: "ArchFrame", teeth: dict, bad: dict | None = None,
                    samples: int = 90, jaw: str | None = None,
                    lift_mm: float = 0.35) -> dict:
    """Симметричная («идеальная») линия дуги и средняя линия — для вьюера.

    Что это такое. Берём одноимённые пары зубов и усредняем их положение
    относительно средней линии: правый зуб и отражённый левый дают одну точку,
    равноудалённую от середины. По этим точкам идёт сплайн. Получается дуга,
    которая проходит через реальные зубы пациента, но СИММЕТРИЧНА — с ней и
    сравнивают фактическое положение.

    Почему только по парам. Непарный зуб (соседа нет или он с плохим мешем)
    задаёт положение лишь одной стороны, и линия в этом месте перестала бы
    быть симметричной — то есть перестала бы быть тем, ради чего строится.

    Почему сплайн по порядку зубов, а не парабола. То же, что и в arch_curve:
    парабола не повторяет подкову, RMS 2.22 мм против 0.39 у сплайна, и она
    сама выдаёт клыки «вестибулярно». Линия рисуется поверх модели, и такая
    ошибка была бы видна врачу глазом.

    Координаты отдаются МИРОВЫЕ (те же, что у мешей в scan.glb), потому что
    вьюер рисует линию прямо в сцене.
    """
    bad = bad or {}
    cen = {t: np.asarray(m.vertices).mean(0) for t, m in teeth.items()}
    mids = []
    for t in cen:
        m = mirror_fdi(t)
        if m in cen:
            mids.append((frame.project(cen[t])[0] + frame.project(cen[m])[0]) / 2)
    if not mids:
        return {"points": [], "midline": [], "reliable": False,
                "note": "нет одноимённых пар — симметричную линию строить не по чему"}
    mid_lat = float(np.median(mids))

    # Симметризованные точки: по одной на пару, зеркально на обе стороны.
    right = []
    for t in sorted(cen, key=lambda x: -(x % 10)):     # от восьмёрки к резцу
        if t // 10 not in (1, 4):
            continue
        m = mirror_fdi(t)
        if m not in cen or t in bad or m in bad:
            continue
        a, b = frame.project(cen[t]), frame.project(cen[m])
        d = (abs(a[0] - mid_lat) + abs(b[0] - mid_lat)) / 2   # общее удаление
        sign = 1.0 if a[0] > mid_lat else -1.0
        # Высоту берём НЕ по центру зуба, а по жевательной поверхности.
        # Врач: «линии не прилегают к поверхности зубов». Так и было: центр
        # коронки лежит ВНУТРИ зуба, и линия шла сквозь него — снаружи
        # выглядело, будто она висит в воздухе. Уровень окклюзии — крайние
        # точки зуба по вертикали: у нижней челюсти это максимум, у верхней
        # минимум (коронки смотрят вниз).
        up_t = frame.project(np.asarray(teeth[t].vertices))[:, 2]
        up_m = frame.project(np.asarray(teeth[m].vertices))[:, 2]
        if jaw == "upper":
            occ = (float(np.percentile(up_t, 3)) + float(np.percentile(up_m, 3))) / 2
            occ -= lift_mm          # чуть ниже коронок — линия видна снизу
        else:
            occ = (float(np.percentile(up_t, 97)) + float(np.percentile(up_m, 97))) / 2
            occ += lift_mm          # чуть выше бугров — линия видна сверху
        right.append([mid_lat + sign * d, (a[1] + b[1]) / 2, occ])
    if len(right) < 3:
        return {"points": [], "midline": [], "reliable": False,
                "note": "мало надёжных пар для линии"}

    right = np.array(right, float)
    left = right.copy()
    left[:, 0] = 2 * mid_lat - left[:, 0]           # зеркальная половина
    ordered = np.vstack([right, left[::-1]])        # обход дуги целиком

    # Сплайн по накопленной длине: параметр — путь вдоль дуги, а не индекс,
    # иначе участок с частыми зубами (передний отдел) растянется.
    t_par = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(ordered, axis=0), axis=1))])
    if t_par[-1] <= 0:
        return {"points": [], "midline": [], "reliable": False,
                "note": "точки дуги совпали"}
    tt = np.linspace(0, t_par[-1], samples)
    curve = np.stack([np.interp(tt, t_par, ordered[:, k]) for k in range(3)], 1)
    if len(ordered) >= 4:
        try:
            from scipy.interpolate import make_interp_spline
            k = min(3, len(ordered) - 1)
            curve = make_interp_spline(t_par, ordered, k=k)(tt)
        except Exception:                            # noqa: BLE001
            pass

    # Обратно в мировые координаты: оси ортонормированы, поэтому обратное
    # преобразование — просто разложение по ним.
    def to_world(p):
        p = np.atleast_2d(p)
        return (p[:, 0:1] * frame.lat + p[:, 1:2] * frame.fwd
                + p[:, 2:3] * frame.up)

    world = to_world(curve)
    # Средняя линия: отрезок в плоскости симметрии, от переднего края дуги
    # вперёд и назад, на средней высоте зубов.
    fwd_min, fwd_max = float(curve[:, 1].min()), float(curve[:, 1].max())
    up_mid = float(np.median(curve[:, 2]))   # тот же уровень, что и сама дуга
    span = (fwd_max - fwd_min) * 0.12
    mid_pts = to_world(np.array([[mid_lat, fwd_min - span, up_mid],
                                 [mid_lat, fwd_max + span, up_mid]]))
    return {
        "points": [[round(float(v), 3) for v in q] for q in world],
        "midline": [[round(float(v), 3) for v in q] for q in mid_pts],
        "pairs_used": int(len(right)),
        "reliable": bool(frame.reliable),
    }


def mesh_quality(rows: list) -> dict:
    """-> {fdi: [причины сомневаться]}. Пусто — значит зуб можно мерить."""
    by = {r["fdi"]: r for r in rows}
    bad = {}
    for r in rows:
        why = []
        if r["mesh_bloat"] is not None and r["mesh_bloat"] > BLOAT_MAX:
            why.append(f"меш длиннее зуба в {r['mesh_bloat']:.1f}× "
                       f"(посторонние куски)")
        if r["buccolingual_mm"] < MIN_BL_MM:
            why.append(f"толщина всего {r['buccolingual_mm']:.1f} мм — обрывок")
        if r["neck_verts"] < MIN_NECK_VERTS:
            why.append("шейка не найдена")
        m = by.get(mirror_fdi(r["fdi"]))
        if m and m["surface_mm2"] > 0 and \
                r["surface_mm2"] / m["surface_mm2"] < MIN_MIRROR_RATIO:
            why.append(f"площадь {r['surface_mm2']:.0f} мм² против "
                       f"{m['surface_mm2']:.0f} у парного {m['fdi']}")
        if why:
            bad[r["fdi"]] = why
    return bad


def analyze_geometry(case_id: str) -> dict:
    teeth, gums = load_case(case_id)
    if not gums:
        raise SystemExit("в кейсе нет мешей десны — шейку определить нечем")
    frame = arch_frame(teeth)
    seq = arch_sequence(teeth)
    gum_trees = {j: cKDTree(np.asarray(g.vertices)) for j, g in gums.items()}
    rows = []
    for fdi, mesh in sorted(teeth.items()):
        jaw = "upper" if is_upper(fdi) else "lower"
        if jaw not in gum_trees:
            continue
        rows.append(tooth_metrics(fdi, mesh, frame, seq, teeth, gum_trees[jaw]))
    return {
        "case_id": case_id,
        "planarity": round(frame.planarity, 4),
        "frame_reliable": frame.reliable,
        "sequence": seq,
        "teeth": rows,
        "mesh_quality": mesh_quality(rows),
    }


# ===================== анатомические ориентиры на зубе =====================
#
# Класс по Энглю определяется НЕ центрами зубов, а двумя точками: мезиально-
# щёчным бугром верхней шестёрки и щёчной бороздой нижней. В норме первый
# стоит ровно во второй, поэтому у правильного измерения база равна НУЛЮ —
# в отличие от разницы центроидов, где норма оказалась −1.9 мм.
#
# Модель предельно простая: ориентир сидит в одном и том же относительном
# месте зуба (доли габарита вдоль дуги / наружу / вверх). Таблица выучена по
# медиане на 40 сканах 3DTeethLand и проверена на других 40:
#
#     FacialPoint  0.80 мм (66% в пределах миллиметра)
#     MBCusp       1.12 мм      Mesial 1.09 мм      Distal 1.09 мм
#
# Рукописные правила («самая щёчная точка в средней полосе») давали втрое
# хуже — 1.9-2.4 мм: у экватора коронка выпирает сильнее, и крайняя точка
# уезжает туда. Две строчки статистики оказались точнее анатомических
# рассуждений, и это стоит помнить.
LANDMARK_POS = {
    (1, "Distal"): (0.103, 0.769, 0.926),
    (2, "Distal"): (0.096, 0.598, 0.874),
    (3, "Distal"): (0.122, 0.412, 0.725),
    (4, "Distal"): (0.108, 0.405, 0.566),
    (5, "Distal"): (0.112, 0.433, 0.610),
    (6, "Distal"): (0.097, 0.446, 0.577),
    (7, "Distal"): (0.086, 0.431, 0.529),
    (1, "FacialPoint"): (0.543, 0.909, 0.604),
    (2, "FacialPoint"): (0.469, 0.911, 0.575),
    (3, "FacialPoint"): (0.411, 0.854, 0.546),
    (4, "FacialPoint"): (0.496, 0.905, 0.565),
    (5, "FacialPoint"): (0.484, 0.904, 0.565),
    (6, "FacialPoint"): (0.524, 0.883, 0.583),
    (7, "FacialPoint"): (0.471, 0.878, 0.540),
    (4, "MBCusp"): (0.453, 0.630, 0.976),
    (5, "MBCusp"): (0.491, 0.663, 0.978),
    (6, "MBCusp"): (0.707, 0.658, 0.962),
    (7, "MBCusp"): (0.648, 0.651, 0.900),
    (1, "Mesial"): (0.906, 0.575, 0.949),
    (2, "Mesial"): (0.858, 0.753, 0.959),
    (3, "Mesial"): (0.875, 0.680, 0.788),
    (4, "Mesial"): (0.872, 0.454, 0.711),
    (5, "Mesial"): (0.889, 0.436, 0.742),
    (6, "Mesial"): (0.901, 0.437, 0.771),
    (7, "Mesial"): (0.877, 0.487, 0.702),
}


def _norm_coords(pts: np.ndarray, axes: tuple) -> np.ndarray:
    """Точки в долях габарита зуба по трём его осям. -> (N, 3)."""
    cols = []
    for ax in axes:
        v = pts @ ax
        cols.append((v - v.min()) / max(float(np.ptp(v)), 1e-6))
    return np.stack(cols, axis=1)


def landmark(pts: np.ndarray, axes: tuple, fdi: int, kind: str):
    """Ориентир на зубе: вершина, ближайшая к выученному положению.

    kind: FacialPoint (щёчная борозда), MBCusp (мезиально-щёчный бугор),
    Mesial / Distal (контактные пункты). Вершина, а не произвольная точка:
    ориентир обязан лежать НА поверхности зуба.
    """
    key = (fdi % 10, kind)
    if key not in LANDMARK_POS or len(pts) < 20:
        return None
    target = np.array(LANDMARK_POS[key])
    C = _norm_coords(pts, axes)
    return pts[int(np.argmin(np.linalg.norm(C - target, axis=1)))]


# ===================== измерения прикуса =====================
#
# Пороги ИЗМЕРЕНЫ по 180 пациентам Bits2Bites с врачебной разметкой
# (ios_frame_validate.py), и проверены на ОБОИХ путях расчёта — по сырой
# сегментации и по очищенному мешу кейса. Расхождение между путями в пределах
# нескольких процентов, то есть пороги переносятся.
#
# Выбраны НЕ по максимуму разделения, а на специфичность около 90%: в
# скрининге ложная тревога дороже пропуска, врач всё равно смотрит пациента
# сам. Пороги по максимуму Юдена трогали почти половину здоровых (доля
# нетронутых I класса 54%), эти — 80-85%.
#
#                        чувствительность / специфичность
#   глубокий прикус            92-93% / 89-92%
#   открытый прикус               95% / 85-86%
#   II класс                   52-62% / 90-92%
#   III класс                  44-53% / 86-95%
#   средняя линия              46-54% / 80-82%
#
# Подписывает ортодонт: выборка одна, к своей популяции надо перепроверить.
DEEP_BITE_MM = 3.8        # перекрытие больше — глубокий прикус
OPEN_BITE_MM = 1.25       # перекрытие меньше — открытый прикус
MIDLINE_MM = 1.8          # смещение средней линии больше — отклонение

# Класс по Энглю по смещению моляров. Отсчёт НЕ ОТ НУЛЯ: верхняя шестёрка
# анатомически стоит дистальнее нижней, поэтому и при I классе смещение
# отрицательное (медиана -1.9 мм). Пороги стоят широко именно потому, что
# разброс I класса велик (квартили -2.9..-1.1) — узкая полоса записывала бы
# каждого второго здорового в патологию.
# Само измерение по центроидам грубое: класс Энгля строится на конкретных
# ориентирах (мезиально-щёчный бугор против щёчной борозды), и до модели
# ориентиров (3DTeethLand) это лишь ориентир, а не диагноз. Отсюда и
# формулировка «похоже на».
CLASS_II_MM = -0.3        # выше — похоже на II класс (дистальный)
CLASS_III_MM = -5.0       # ниже — похоже на III класс (мезиальный)
CROSSBITE_MM = 0.5        # верхний зуб должен перекрывать нижний снаружи

INCISORS_UP, INCISORS_LO = (11, 21), (41, 31)
MOLAR_PAIRS = (("справа", 16, 46), ("слева", 26, 36))
LATERAL_PAIRS = ((14, 44), (15, 45), (16, 46), (17, 47),
                 (24, 34), (25, 35), (26, 36), (27, 37))


def occlusion_measures(frame: ArchFrame, pts: dict) -> dict:
    """Прикус по облакам точек зубов {fdi: Nx3}. Чистая геометрия.

    Одна реализация на два входа: CLI `ios_occlusion` кормит сюда точки прямо
    из сегментации OBJ, а слой находок — вершины мешей из scan.glb. Разводить
    эти измерения по двум местам нельзя: именно так уже разъехалась вертикаль.

    Все величины меряются в осях дуги, а их вертикаль — перпендикуляр к
    плоскости дуги. Прежний способ (разница центров верх/низ) промахивался на
    31.6° и ронял разделение диагнозов: глубокий прикус 0.573 против 0.971 AUC.
    """
    lat, fwd, up = frame.lat, frame.fwd, frame.up
    proj = lambda p: np.stack([p @ lat, p @ fwd, p @ up], axis=-1)
    out = {"measurements": {}, "findings": []}
    add = lambda name, val, who="ортодонт": out["findings"].append((name, val, who))

    ui = [pts[t] for t in INCISORS_UP if t in pts]
    li = [pts[t] for t in INCISORS_LO if t in pts]
    if ui and li:
        pu, pl = proj(np.vstack(ui)), proj(np.vstack(li))
        # режущий край верхнего резца — снизу, нижнего — сверху
        overbite = float(np.percentile(pl[:, 2], 98) - np.percentile(pu[:, 2], 2))
        # губная поверхность: самая передняя точка коронки
        overjet = float(np.percentile(pu[:, 1], 98) - np.percentile(pl[:, 1], 98))
        midline = float(pu[:, 0].mean() - pl[:, 0].mean())
        out["measurements"].update({
            "перекрытие_мм": round(overbite, 1),
            "сагиттальная_щель_мм": round(overjet, 1),
            "средняя_линия_мм": round(midline, 1)})

        if overbite > DEEP_BITE_MM:
            add("Глубокое резцовое перекрытие (deep bite)", f"{overbite:.1f} мм")
        elif overbite < OPEN_BITE_MM:
            add("Открытый прикус (open bite)", f"{overbite:.1f} мм")
        if overjet < 0:
            add("Обратное резцовое перекрытие", f"{overjet:.1f} мм")
        if abs(midline) > MIDLINE_MM:
            add(f"Смещение средней линии {'влево' if midline > 0 else 'вправо'}",
                f"{abs(midline):.1f} мм")

    for side, u6, l6 in MOLAR_PAIRS:
        if u6 not in pts or l6 not in pts:
            continue
        d = float(proj(pts[u6]).mean(0)[1] - proj(pts[l6]).mean(0)[1])
        out["measurements"][f"моляры_{side}_мм"] = round(d, 1)
        if d > CLASS_II_MM:
            cls = "похоже на II класс (дистальный)"
        elif d < CLASS_III_MM:
            cls = "похоже на III класс (мезиальный)"
        else:
            cls = "в пределах I класса"
        out["measurements"][f"класс_{side}"] = cls
        if not cls.startswith("в пределах"):
            add(f"{cls}, {side}", f"смещение {d:+.1f} мм")

    cross = []
    for u, l in LATERAL_PAIRS:
        if u not in pts or l not in pts:
            continue
        # в норме верхний зуб перекрывает нижний СНАРУЖИ (щёчнее)
        if abs(proj(pts[u]).mean(0)[0]) < abs(proj(pts[l]).mean(0)[0]) - CROSSBITE_MM:
            cross.append(f"{u}-{l}")
    out["measurements"]["перекрёстный_зубы"] = cross
    if cross:
        add("Перекрёстный прикус", ", ".join(cross))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    args = ap.parse_args()
    res = analyze_geometry(args.case_id)

    print(f"\nкейс {res['case_id']}: плоскостность дуги {res['planarity']:.3f} "
          f"({'оси надёжны' if res['frame_reliable'] else 'ОСИ НЕНАДЁЖНЫ'})")
    for jaw, row in res["sequence"].items():
        print(f"  порядок {jaw}: {row}")
    bad = res["mesh_quality"]
    print(f"\n{'зуб':>5} {'коронка':>9} {'мезио-дист.':>12} {'щёчно-язык.':>12} "
          f"{'габарит':>9} {'раздут':>8}  качество")
    for r in res["teeth"]:
        f = lambda k, w: "—".rjust(w) if r[k] is None else f"{r[k]:{w}.2f}"
        flag = "" if r["fdi"] not in bad else "  <-- " + "; ".join(bad[r["fdi"]])
        print(f"{r['fdi']:>5} {f('crown_height_mm', 9)} {f('mesiodistal_mm', 12)} "
              f"{r['buccolingual_mm']:12.2f} {r['arch_extent_mm']:9.2f} "
              f"{f('mesh_bloat', 8)}{flag}")
    if bad:
        print(f"\nзубов с сомнительным мешем: {len(bad)} — "
              f"измерения по ним отдавать с пониженным доверием")


if __name__ == "__main__":
    main()
