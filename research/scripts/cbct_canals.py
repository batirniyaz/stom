"""Корневые каналы: дотрассировка по плотности от устья до верхушки.

    .venv\\Scripts\\python.exe research\\scripts\\cbct_canals.py --validate 10
    .venv\\Scripts\\python.exe research\\scripts\\cbct_canals.py --case client5 --model sp

ЗАЧЕМ. Врач с приёма: у зубов с 3-4 каналами каналы не дорисованы — где
один, где ни одного, а смотреть их нужно целиком. Замер на client5/sp:
пульпа модели доходит лишь до 72-91% длины зуба, у 26, 27, 46 на двух
третях длины виден один канал. При этом НА СНИМКЕ каналы видны: на срезах
вдоль зуба по центру корней идут тёмные линии до самой верхушки. Модель их
недометила, а не аппарат не снял.

ПОЧЕМУ ТРАССИРОВКА, А НЕ ПОРОГ. Канал в корне — 1-2 вокселя, его контраст
на клиническом аппарате на грани: в самой тёмной точке сечения корня
«тёмность» (закрытие минус исходное) 150-356 при таком же хвосте у
обычного дентина (90-й перцентиль 200-250). Порог по вокселю либо рвёт
канал на куски (оторванное от камеры отбрасывается), либо разрастается в
шум: при пороге 0.2 от плотности дентина у 16 в одном срезе выходило пять
«каналов». Канал же — НЕПРЕРЫВНАЯ линия вдоль корня, и это знание
решает: идём от устья к верхушке срез за срезом, на каждом берём самую
тёмную точку рядом с предыдущей. Шум в дентине не складывается в линию,
канал складывается.

КАК УСТРОЕНО.
1. Зуб «выпрямляется»: выборка вдоль его собственной оси (главная ось
   вокселей), срез за срезом от коронки к апексу.
2. Затравки — концы обрывков пульпы апикальнее камеры (по одной на обрывок:
   у мезиального корня нижнего моляра их бывает два). Корень без обрывка
   (тот самый «ни одного канала») получает затравку в самой тёмной точке
   сразу под развилкой корней.
3. Шаг: самая тёмная точка (после лёгкого размытия) в радиусе
   `SEARCH_R` от предыдущей, не у поверхности корня — у поверхности
   тёмная периодонтальная щель.
4. Стоп: корень кончился, или канал в среднем за `STOP_WINDOW` срезов не
   темнее кольца дентина вокруг него больше чем на `CONTRAST_SIGMA` шума
   (облитерирован или запломбирован — пломба яркая, тёмной линии там нет).
5. Куски канала, отмеченные моделью с разрывом, соединяются отрезком
   (`_bridges`): канал обязан проходить между ними.
6. Вокруг трасс — дорастание по порогу (`complete_canals`).

ГДЕ ПРЕДЕЛ. Не в корне: маска зуба доходит до верхушки. Проверено на
эталоне DentVoxel (178 зубов, 6 томов, модель этих данных не видела):
расхождение у верхушки медиана -0.13 мм (предсказание даже чуть длиннее
эталона), 90-й перцентиль 0.26 мм, короче эталона больше чем на 1 мм — 1%
зубов. Предел в САМОМ КАНАЛЕ: у обучающих данных он размечен не до конца.
У ToothFairy3 пульпа доходит до 85% длины зуба лишь у 46% зубов (медиана
0.84) — модель повторяет своего учителя.

ЭТОТ МОДУЛЬ ТЕПЕРЬ ЗАПАСНОЙ ПУТЬ. Отдельная модель канала на вырезе зуба
(`train_canal_net.py`) обучена и сверена с трассировкой ЭТИМ ЖЕ протоколом
(`--validate N --net <веса>`, тома только из проверочной доли): ошибка
дохода 0.017 против 0.089, число каналов совпадает с эталоном в 91% против
71% при 11 лишних против 40, покрытие отрезанного канала 0.99 против 0.84,
разрыв в середине закрыт у 94 из 94 против 57 из 94. В конвейере стоит она
(`cbct_backend.canal_net`); трассировка работает, если весов нет.

ГРАБЛИ ЗАМЕРА, стоившие неверного вывода. Сначала «недоход» корня мерился
как продолжение плотной ткани за концом маски вдоль оси зуба — вышло 3.5-4
мм, и я записал это как «модель обрезает корень». Неверно: у верхушки
вплотную лежит плотная кость (lamina dura), и порог по плотности утекает в
неё даже внутри конуса вокруг оси. Разошлось с эталоном в 30 раз. Правило
прежнее: там, где есть эталон, мерить по нему, а «похоже на ткань по
плотности» — не эталон.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

SEARCH_R = 1.5        # вокселей вбок на шаг (зуб наклонён, канал изогнут)
EDGE_MIN = 1.5        # не ближе к поверхности корня, вокселей
# Когда канал «кончился». Порог — в единицах ШУМА дентина этого зуба, а не
# в HU и не в долях плотности: шум у аппаратов разный (ToothFairy3 — 108,
# client5 — 139), и абсолютный порог на клинике обрывал канал сразу под
# концом маски. Замер на ToothFairy3 (контраст точки к кольцу вокруг, в
# шумах):
#
#                        p10   p25   p50   p75   p90
#     эталонный канал    0.8   1.6   2.5   4.3   6.3     (2945 точек)
#     дентин без канала -0.5  -0.2   0.1   0.5   0.9     (6179 точек)
#
# Решение по среднему за несколько срезов: одиночный бледный срез на
# изгибе канала не должен его обрывать.
#
# Порог 0.7, а не 1.0: на ToothFairy3 разница мала (точность прироста 0.84
# против 0.86, ошибка дохода 0.077 против 0.075), а на клинике каналы
# доходят заметно дальше (client5: 26 — 0.93 против 0.88, 37 — 0.94 против
# 0.90). Врачу нужны каналы целиком.
CONTRAST_SIGMA = 0.7
STOP_WINDOW = 4
RING = (2.0, 4.0)     # кольцо «окружающего дентина», вокселей
# У самой верхушки сечение корня крошечное: канал там не различить, а самая
# тёмная точка уходит вбок, и конец трассы ложился в 3D плоской «ножкой».
APEX_AREA = 20        # вокселей в сечении корня (~1.9 мм2) — дальше не идём
# Размытие перед поиском минимума, вокселей. Проверено и НЕ помогло
# размывать сильнее вдоль оси зуба (2.0 вдоль, 0.7 поперёк): оценка шума
# растёт вместе с сигналом, отношение не меняется.
SMOOTH = 0.7

# Дорастание вокруг найденных линий: канал бывает шире вокселя, а у корня
# с двумя каналами трассировка идёт по одному. Порог — по «тёмности»
# (закрытие минус исходное) в долях плотности дентина ЭТОГО зуба; рост
# только связно с пульпой и трассами и только апикальнее камеры.
GROW_CLOSING = 5
GROW_ALPHA = 0.35     # тёмность >= alpha * K
GROW_BETA = 0.25      # и сам воксель темнее дентина на beta * K
GROW_K = 0.4          # K = 0.4 * плотность дентина
GROW_INNER = 2.0      # не ближе к поверхности зуба, вокселей


def _frame(body: np.ndarray, sp: np.ndarray, apex_sign: int):
    pts = np.argwhere(body) * sp
    ctr = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
    ax = vt[0] if np.sign(vt[0][0]) == apex_sign else -vt[0]
    ref = np.array([0.0, 1.0, 0.0]) if abs(ax[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(ax, ref); u /= np.linalg.norm(u)
    w = np.cross(ax, u)
    rel = pts - ctr
    t = rel @ ax
    r = np.linalg.norm(rel - np.outer(t, ax), axis=1)
    return ctr, ax, u, w, float(t.min()), float(t.max()), float(r.max())


def _straighten(vol, body, pulp, sp, apex_sign):
    """Выборка зуба вдоль оси: (уровень, u, w) и координаты каждой точки в мм."""
    ctr, ax, u, w, t0, t1, rmax = _frame(body, sp, apex_sign)
    s = float(min(sp))
    ta = np.arange(t0 - s, t1 + s, s)
    ru = np.arange(-rmax - 1.0, rmax + 1.0 + 1e-6, s)
    A, U, W = np.meshgrid(ta, ru, ru, indexing="ij")
    pos = ctr + A[..., None] * ax + U[..., None] * u + W[..., None] * w
    coords = np.moveaxis(pos / sp, -1, 0)
    sv = ndi.map_coordinates(vol.astype(np.float32), coords, order=1,
                             mode="constant", cval=float(vol.min()))
    sb = ndi.map_coordinates(body.astype(np.float32), coords, order=1) > 0.5
    spp = ndi.map_coordinates(pulp.astype(np.float32), coords, order=1) > 0.5
    return sv, sb, spp, pos


def _seeds(sb, spp, sm):
    """(уровень, y, x) затравок: концы обрывков пульпы и корни без каналов."""
    n = sb.shape[0]
    area = spp.sum(axis=(1, 2))
    if not area.any():
        return []
    ic = int(np.argmax(area))                      # уровень камеры
    stub = spp.copy()
    stub[:ic + 2] = False
    # развилка: первый уровень ниже камеры, где сечение тела распалось
    fork = None
    for i in range(ic + 1, n):
        lb, m = ndi.label(sb[i])
        sizes = ndi.sum(sb[i], lb, range(1, m + 1)) if m else []
        if sum(1 for z in sizes if z >= 8) >= 2:
            fork = i
            break
    seeds = []

    def deepest(comp_mask):
        pts = np.argwhere(comp_mask)
        i = int(pts[:, 0].max())
        yx = pts[pts[:, 0] == i][:, 1:].mean(0)
        seeds.append((i, float(yx[0]), float(yx[1])))

    # Затравка в КАЖДОМ корне. Каналы соседних корней сходятся в камеру
    # выше развилки, и весь обрывок пульпы — одна связная компонента: одна
    # затравка на самом глубоком её конце уводила трассу только в один
    # корень (так было у 46 client5).
    lab, k = ndi.label(stub, structure=np.ones((3, 3, 3)))
    for s in range(1, k + 1):
        comp = lab == s
        if fork is None or not comp[fork:].any():
            deepest(comp)
            continue
        below = comp.copy()
        below[:fork] = False
        bl, bk = ndi.label(below, structure=np.ones((3, 3, 3)))
        for r in range(1, bk + 1):
            deepest(bl == r)
    if not seeds and fork is None and ic + 3 < n:
        # однокорневой зуб, пульпа только в камере: канал ищем прямо под ней
        i = ic + 2
        lb, m = ndi.label(sb[i])
        if m:
            comp = lb == (1 + int(np.argmax(ndi.sum(sb[i], lb, range(1, m + 1)))))
            edt = ndi.distance_transform_edt(comp)
            cand = np.where(edt >= EDGE_MIN, sm[i], np.inf)
            if not np.isinf(cand).all():
                y, x = np.unravel_index(np.argmin(cand), cand.shape)
                seeds.append((i - 1, float(y), float(x)))
    if fork is not None and fork + 2 < n:
        i = fork + 2
        lb, m = ndi.label(sb[i])
        for r in range(1, m + 1):
            comp = lb == r
            if comp.sum() < 8:
                continue
            # есть ли в этом корне обрывок пульпы или затравка поблизости
            if (spp[i:min(i + 6, n)] & comp[None]).any():
                continue
            if any(abs(si - i) <= 6 and comp[int(round(y)), int(round(x))]
                   for si, y, x in seeds
                   if 0 <= int(round(y)) < comp.shape[0] and 0 <= int(round(x)) < comp.shape[1]):
                continue
            edt = ndi.distance_transform_edt(comp)
            cand = np.where(edt >= EDGE_MIN, sm[i], np.inf)
            if np.isinf(cand).all():
                continue
            y, x = np.unravel_index(np.argmin(cand), cand.shape)
            seeds.append((i - 1, float(y), float(x)))
    return seeds


BRIDGE_GAP = 16      # уровней (~5 мм) — дальше разрыв не перекрываем
BRIDGE_SIDE = 4.0    # вокселей вбок между концами кусков


def _bridges(sb, spp):
    """Отрезки между кусками канала, которые модель отметила с разрывом.

    Модель иногда видит канал кусками: у 35 client5 пульпа идёт до уровня
    23 и отдельно два вокселя у самой верхушки, а между ними пусто. Канал
    обязан проходить между кусками, если они лежат друг над другом почти на
    одной линии. Камера тоже участвует — как верхний «кусок».
    """
    n = sb.shape[0]
    area = spp.sum(axis=(1, 2))
    if not area.any():
        return []
    ic = int(np.argmax(area))
    stub = spp.copy()
    stub[:ic + 2] = False
    lab, k = ndi.label(stub, structure=np.ones((3, 3, 3)))
    ends = []                                   # (верх, низ) каждого куска
    for s in range(1, k + 1):
        pts = np.argwhere(lab == s)
        lo, hi = int(pts[:, 0].min()), int(pts[:, 0].max())
        top = np.r_[lo, pts[pts[:, 0] == lo][:, 1:].mean(0)]
        bot = np.r_[hi, pts[pts[:, 0] == hi][:, 1:].mean(0)]
        ends.append((top, bot))
    floor = ic + 1
    if spp[floor].any():
        yx = np.argwhere(spp[floor]).mean(0)
        chamber_bot = np.r_[floor, yx]
    else:
        chamber_bot = None
    out = []
    for i, (top_b, _bot_b) in enumerate(ends):
        if top_b[0] <= ic + 2:
            continue                            # кусок и так касается камеры
        best = None
        cands = [bot for j, (_t, bot) in enumerate(ends) if j != i]
        if chamber_bot is not None:
            cands.append(chamber_bot)
        for bot_a in cands:
            gap = top_b[0] - bot_a[0]
            side = float(np.hypot(*(top_b[1:] - bot_a[1:])))
            if 0 < gap <= BRIDGE_GAP and side <= BRIDGE_SIDE:
                if best is None or gap < best[0]:
                    best = (gap, bot_a)
        if best is None:
            continue
        a, b = best[1], top_b
        m = int(np.ceil(np.abs(b - a).max())) + 1
        q = np.round(np.linspace(a, b, m)).astype(int)
        inside = sb[q[:, 0].clip(0, n - 1), q[:, 1].clip(0, sb.shape[1] - 1),
                    q[:, 2].clip(0, sb.shape[2] - 1)].mean()
        if inside >= 0.8:                        # отрезок идёт внутри зуба
            out.append((a, b))
    return out


def _noise(sm, sb, spp) -> float:
    """Шум дентина зуба: MAD размытой плотности в глубине корней."""
    n = sb.shape[0]
    vals = []
    for i in range(int(n * 0.45), int(n * 0.9)):
        edt = ndi.distance_transform_edt(sb[i])
        m = (edt >= 2) & ~ndi.binary_dilation(spp[i], iterations=2)
        vals.append(sm[i][m])
    vals = np.concatenate(vals) if vals else np.array([])
    if len(vals) < 20:
        return 0.0
    return float(1.4826 * np.median(np.abs(vals - np.median(vals))))


def _trace(sb, sm, seed, sigma):
    n = sb.shape[0]
    i0, y, x = seed
    path, cons = [], []
    for j in range(i0 + 1, n):
        lab, _ = ndi.label(sb[j])
        yi, xi = int(round(y)), int(round(x))
        win = lab[max(yi - 2, 0):yi + 3, max(xi - 2, 0):xi + 3]
        ids = np.unique(win[win > 0])
        if not len(ids):
            break                                   # корень кончился
        comp = np.isin(lab, ids)
        if comp.sum() < APEX_AREA:
            break                                   # у верхушки канал не различить
        edt = ndi.distance_transform_edt(comp)
        yy, xx = np.nonzero(comp & (edt >= EDGE_MIN))
        if not len(yy):
            break                                   # корень тоньше допуска — апекс
        d = np.hypot(yy - y, xx - x)
        sel = d <= SEARCH_R
        if not sel.any():
            break
        k = int(np.argmin(sm[j][yy[sel], xx[sel]]))
        ny, nx = int(yy[sel][k]), int(xx[sel][k])
        gy, gx = np.ogrid[:comp.shape[0], :comp.shape[1]]
        rr = np.hypot(gy - ny, gx - nx)
        # внешний слой корня темнее из-за частичного объёма — в кольцо не берём
        ring = comp & (rr >= RING[0]) & (rr <= RING[1]) & (edt >= 1.0)
        if ring.sum() < 4:
            break
        cons.append(float(np.median(sm[j][ring]) - sm[j][ny, nx]) / sigma)
        path.append((j, ny, nx))
        if len(cons) >= STOP_WINDOW and np.mean(cons[-STOP_WINDOW:]) < CONTRAST_SIGMA:
            break
        y, x = ny, nx
    # хвост, где канал уже не отличался от дентина, отрезаем
    while cons and cons[-1] < CONTRAST_SIGMA:
        cons.pop()
        path.pop()
    # и конечные шаги вбок: канал идёт вдоль корня, и два боковых шага
    # подряд в самом конце — это уход минимума к краю, а не изгиб канала
    # (у наклонного канала боковой шаг бывает раз в два-три среза).
    def side(a, b):
        return np.hypot(a[1] - b[1], a[2] - b[2])
    while (len(path) >= 3 and side(path[-1], path[-2]) >= 1.0
           and side(path[-2], path[-3]) >= 1.0):
        path.pop()
    return path


def grow_dark(vol, body, seed, spacing, apex_sign: int, pulp=None) -> np.ndarray:
    """Нарастить `seed` по тёмным тонким структурам внутри зуба."""
    import cc3d
    sp = np.asarray(spacing, float)
    pulp = seed if pulp is None else pulp
    v = vol.astype(np.float32, copy=False)
    dentin = float(np.median(v[body & ~pulp]))
    k = GROW_K * dentin
    th = ndi.grey_closing(v, size=(GROW_CLOSING,) * 3) - v
    inner = ndi.distance_transform_edt(body) >= GROW_INNER
    cand = inner & (th >= GROW_ALPHA * k) & (v <= dentin - GROW_BETA * k)
    ctr, ax, *_r = _frame(body, sp, apex_sign)
    pc = float(((np.argwhere(pulp) * sp - ctr) @ ax).mean())
    cidx = np.argwhere(cand)
    keep = np.zeros_like(cand)
    keep[tuple(cidx[((cidx * sp - ctr) @ ax) > pc].T)] = True
    lab = cc3d.connected_components(keep | seed, connectivity=26)
    ids = np.unique(lab[seed])
    return np.isin(lab, ids[ids > 0]) & (keep | seed)


def _tube(lines: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Трубка вокруг осевых линий, с тонкими концами.

    Канал толщиной в воксель в 3D читается ниткой; врачу нужна трубка.
    Утолщаем кубом 3x3x3, а не крестом: у диагональной линии крест даёт
    переменную толщину, и после размытия перед мешем канал шёл «бусинами».
    Утолщаем только ВНУТРЕННИЕ точки линии (у них два соседа по линии):
    у верхушки модельный канал тонкий, и куб вокруг последнего шага ложился
    на его конец плоской «ножкой» шире самого канала.
    """
    cube = np.ones((3, 3, 3), bool)
    nb = ndi.convolve(lines.astype(np.uint8), cube.astype(np.uint8),
                      mode="constant")
    inner = lines & (nb >= 3)             # сам воксель + два соседа
    return (ndi.binary_dilation(inner, cube) | lines) & body


def complete_canals(vol, body, pulp, spacing, apex_sign: int) -> np.ndarray:
    """Трассировка + дорастание: воксели каналов, которых не было в `pulp`.

    Проверка на ToothFairy3 (80 зубов, эталонная пульпа обрезана на
    половине длины зуба), в порядке доработки:

        вариант                         покрытие точность ошибка  каналы  лишних
                                                          дохода  совпали
        только трассировка                0.58     0.92    0.096    0.52     17
        только порог                      0.87     0.88    0.078    0.65     69
        трассировка + порог               0.88     0.87    0.077    0.65     77
        + трубка кубом, раскрытие роста,  0.85     0.86    0.075    0.73     17
          связность, мосты, тонкие концы

    Итоговый вариант (последняя строка): число каналов в срезе совпадает с
    эталоном в 73% случаев, «лишних» каналов вчетверо меньше; пульпа,
    обрезанная у камеры, дотягивается до эталона в 31 из 37; разрыв на
    45-70% длины закрывается целиком у 49 зубов из 79 (медиана покрытия 0.94).

    Порог в одиночку на эталоне хорош, но на клиническом аппарате канал
    бледнее и рвётся: у client5 он почти не дорастал. Трассы дают ему
    непрерывные затравки до верхушки.
    """
    lines = trace_lines(vol, body, pulp, spacing, apex_sign)
    if not lines.any():
        return lines
    cross = ndi.generate_binary_structure(3, 1)
    tube = _tube(lines, body)
    grown = grow_dark(vol, body, pulp | tube, spacing, apex_sign, pulp=pulp)
    extra = grown & ~pulp & ~tube
    # Раскрытие убирает поперечные «шипики» в воксель толщиной: рост цепляет
    # тёмные воксели сбоку от трассы, и в 3D они торчат иглами (шипы уже
    # были отдельной жалобой врача). Раскрываем ТОЛЬКО приросшее по порогу:
    # трубку трассы раскрытие рвало у поверхности корня, и канал шёл
    # «бусинами». Пульпа и трубка участвуют как опора, чтобы не оторвать стык.
    extra &= ndi.binary_opening(pulp | tube | extra, cross)
    add = (tube | extra) & ~pulp
    # Всё, что не связано с пульпой, — крошки: в 3D они висят отдельно.
    import cc3d
    lab = cc3d.connected_components(pulp | add, connectivity=26)
    ids = np.unique(lab[pulp])
    return add & np.isin(lab, ids[ids > 0])


def trace_lines(vol, body, pulp, spacing, apex_sign: int) -> np.ndarray:
    """Осевые линии дотрассированных каналов (маска той же формы, что `body`).

    `body` — зуб ВМЕСТЕ с пульпой; `apex_sign` — знак оси 0 (z), в которую
    смотрит апекс этого зуба. Каждая линия начинается в пульпе: у корня без
    канала затравка лежит под развилкой, и линия тянется к ближайшей точке
    пульпы — иначе канал висел бы отдельно от камеры.
    """
    sp = np.asarray(spacing, float)
    out = np.zeros(body.shape, bool)
    if pulp.sum() < 20 or body.sum() < 500:
        return out
    dentin = float(np.median(vol[body & ~pulp]))
    if dentin <= float(np.median(vol[pulp])):
        return out          # в пульпе плотнее дентина — пломба, тёмных каналов нет
    sv, sb, spp, pos = _straighten(vol, body, pulp, sp, apex_sign)
    sm = ndi.gaussian_filter(sv, SMOOTH)
    sigma = _noise(sm, sb, spp)
    if sigma <= 0:
        return out
    shape = np.array(body.shape)
    pulp_pts = np.argwhere(pulp).astype(float)

    def to_vol(pt):
        # (уровень, y, x) выпрямленной выборки -> индексы тома
        j, y, x = (int(round(v)) for v in pt)
        return pos[j, y, x] / sp

    def draw(pts):
        # Соседние точки соединяем отрезком: после перевода на сетку тома
        # они отстоят на воксель-полтора, и без этого канал в 3D рассыпался
        # на «бусины» (было видно на 47 client5).
        for a, b in zip(pts[:-1], pts[1:]):
            n = int(np.ceil(np.abs(b - a).max())) + 1
            for q in np.round(np.linspace(a, b, n)).astype(int):
                if (q >= 0).all() and (q < shape).all() and body[tuple(q)]:
                    out[tuple(q)] = True

    for a, b in _bridges(sb, spp):
        draw([to_vol(a), to_vol(b)])
    for seed in _seeds(sb, spp, sm):
        i0, y0, x0 = seed
        start = pos[i0, int(round(y0)), int(round(x0))] / sp
        path = _trace(sb, sm, seed, sigma)
        if not path:
            continue
        near = pulp_pts[np.argmin(((pulp_pts - start) ** 2).sum(1))]
        draw([near, start] + [pos[j, y, x] / sp for j, y, x in path])
    return out & ~pulp


def trace_canals(vol, body, pulp, spacing, apex_sign: int) -> np.ndarray:
    """Только трассы, утолщённые до трубки (без дорастания) — для сравнения."""
    lines = trace_lines(vol, body, pulp, spacing, apex_sign)
    if not lines.any():
        return lines
    tube = _tube(lines, body)
    return tube & ~pulp


# ======================= проверка =======================

def _levels(body, pulp, sp, apex_sign, levels=(0.55, 0.66, 0.77, 0.88)):
    import cc3d
    ctr, ax, *_rest = _frame(body, sp, apex_sign)
    t = (np.argwhere(body) * sp - ctr) @ ax
    t0, t1 = t.min(), t.max()
    pidx = np.argwhere(pulp)
    if not len(pidx):
        return 0.0, [0] * len(levels)
    pp = (pidx * sp - ctr) @ ax
    cnt = []
    for f in levels:
        band = np.zeros_like(pulp)
        band[tuple(pidx[np.abs(pp - (t0 + (t1 - t0) * f)) < 0.5].T)] = True
        cnt.append(int(cc3d.connected_components(band, connectivity=26).max())
                   if band.any() else 0)
    return float((pp.max() - t0) / (t1 - t0)), cnt


def _net_canal(net, device, v, b, sp, sgn):
    """Канал по модели (`train_canal_net`) в координатах того же кропа."""
    import cbct_canal_apply as ca
    return ca.canal_in_tooth(net, device, v, b, np.zeros_like(b), sp, sgn)


def validate(n_vol: int, net_weights: str | None = None) -> None:
    """ToothFairy3: пульпу эталона обрезаем на половине длины зуба и
    дотрассировываем; сравниваем с отрезанным.

    `net_weights` — сверх того же протокола прогнать модель канала и союз
    «трассировка ИЛИ сеть». Одним протоколом, потому что иначе числа двух
    способов несопоставимы: у трассировки они уже посчитаны здесь.

    ВАЖНО: с моделью тома берутся ТОЛЬКО из проверочной доли нарезки
    (`cbct_canal_crops.split_of`). На обучающих томах сеть мерила бы себя
    по своему же учителю — ошибка, которая в проекте уже случалась.
    """
    import SimpleITK as sitk
    from skimage.morphology import skeletonize
    import cbct_backend as cb
    ds = Path(__file__).resolve().parents[2] / "research_3d/nnUNet_raw/Dataset113_ToothFairy3"
    stems = sorted(p.name[:-7] for p in (ds / "labelsTr_fdi").glob("*.nii.gz"))
    net = device = None
    if net_weights:
        import torch
        import cbct_canal_apply as ca
        import cbct_canal_crops as cc
        stems = [s for s in stems if cc.split_of(f"tf3/{s}") == "val"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, info = ca.load_net(Path(net_weights), device)
        print(f"модель канала: эпоха {info['epoch']}, томов проверочной доли {len(stems)}")
    stems = stems[5::41][:n_vol] if not net_weights else stems[:n_vol]
    ball = ndi.generate_binary_structure(3, 1)
    n = 0
    gt_c = np.zeros(4)

    def new_acc():
        return {"cover": [], "prec": [], "rerr": [], "exact": 0, "over": 0,
                "cnt": np.zeros(4), "miss_ok": 0, "miss_n": 0, "gap": []}

    accs = {"трассировка": new_acc()}
    if net is not None:
        accs["сеть"] = new_acc()
        accs["союз"] = new_acc()
    for stem in stems:
        img = sitk.ReadImage(str(ds / "imagesTr" / f"{stem}_0000.nii.gz"))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(ds / "labelsTr_fdi" / f"{stem}.nii.gz")))
        sp = np.array(img.GetSpacing(), float)[::-1]
        zup = cb.occlusal_up(mask)
        for fdi in [f for f in np.unique(mask) if 11 <= f <= 48 and f % 10 >= 4]:
            pg, tooth = mask == fdi + 100, mask == fdi
            if pg.sum() < 150 or tooth.sum() < 2000:
                continue
            sgn = 1 if (fdi // 10 in (1, 2)) == zup else -1
            body = tooth | pg
            idx = np.argwhere(body)
            box = tuple(slice(a, b) for a, b in zip(np.maximum(idx.min(0) - 4, 0), idx.max(0) + 5))
            b, g, v = body[box], pg[box], vol[box]
            gr, gc = _levels(b, g, sp, sgn)
            if gr < 0.8:
                continue
            ctr, ax, *_r, t0, t1, _rm = _frame(b, sp, sgn)
            pidx = np.argwhere(g)
            pp = (pidx * sp - ctr) @ ax
            tr = np.zeros_like(g)
            tr[tuple(pidx[pp <= t0 + (t1 - t0) * 0.5].T)] = True
            add = complete_canals(v, b, tr, sp, sgn)
            # Сеть пульпу на вход НЕ получает (только плотность и тело зуба),
            # поэтому на все три опыта у неё одно предсказание.
            pred = _net_canal(net, device, v, b, sp, sgn) if net is not None else None
            adds = {"трассировка": add}
            if pred is not None:
                adds["сеть"] = pred & ~tr
                adds["союз"] = (add | pred) & ~tr
            n += 1
            gt_c += gc
            skel = skeletonize(g & ~tr)

            # корень без канала: обрезаем пульпу у самой камеры — затравки
            # тогда только от развилки
            tr2 = np.zeros_like(g)
            tr2[tuple(pidx[pp <= t0 + (t1 - t0) * 0.30].T)] = True
            add2 = complete_canals(v, b, tr2, sp, sgn) if tr2.sum() >= 20 else None
            # разрыв посередине: модель видит канал кусками
            band = (pp > t0 + (t1 - t0) * 0.45) & (pp < t0 + (t1 - t0) * 0.70)
            tr3 = add3 = sk = None
            if band.sum() >= 20:
                tr3 = g.copy()
                tr3[tuple(pidx[band].T)] = False
                add3 = complete_canals(v, b, tr3, sp, sgn)
                gap = np.zeros_like(g)
                gap[tuple(pidx[band].T)] = True
                sk = skeletonize(gap)

            for name, ad in adds.items():
                acc = accs[name]
                nr, nc = _levels(b, tr | ad, sp, sgn)
                acc["rerr"].append(abs(nr - gr))
                if skel.sum():
                    acc["cover"].append(float(
                        (skel & ndi.binary_dilation(ad, ball, iterations=2)).sum() / skel.sum()))
                if ad.sum():
                    acc["prec"].append(float(
                        (ad & ndi.binary_dilation(g, ball, iterations=1)).sum() / ad.sum()))
                acc["exact"] += sum(int(x == y) for x, y in zip(nc[:3], gc[:3]))
                acc["over"] += sum(max(x - y, 0) for x, y in zip(nc[:3], gc[:3]))
                acc["cnt"] += nc
                if add2 is not None:
                    a2 = {"трассировка": add2, "сеть": pred,
                          "союз": (add2 | pred) if pred is not None else add2}[name]
                    r2, _ = _levels(b, tr2 | a2, sp, sgn)
                    acc["miss_n"] += 1
                    acc["miss_ok"] += int(r2 >= gr - 0.15)
                if sk is not None and sk.sum():
                    a3 = {"трассировка": add3, "сеть": pred,
                          "союз": (add3 | pred) if pred is not None else add3}[name]
                    acc["gap"].append(float(
                        (sk & ndi.binary_dilation(a3, ball, iterations=2)).sum() / sk.sum()))
        print(f"  {stem}: зубов {n}", flush=True)

    print(f"\nзубов {n}")
    print(f"каналов 55/66/77/88 у эталона: {gt_c.astype(int)}")
    for name, acc in accs.items():
        g = np.array(acc["gap"])
        print(f"\n--- {name}")
        print(f"доход: ошибка {np.mean(acc['rerr']):.3f}")
        print(f"каналов 55/66/77/88: {acc['cnt'].astype(int)}; "
              f"совпало {acc['exact'] / (3 * max(n, 1)):.2f}, лишних {acc['over']}")
        print(f"покрытие отрезанного канала {np.mean(acc['cover']):.2f} "
              f"(медиана {np.median(acc['cover']):.2f})")
        print(f"точность прироста {np.mean(acc['prec']):.2f} "
              f"(медиана {np.median(acc['prec']):.2f})")
        print(f"пульпа обрезана у камеры: дотянуто до эталона (±0.15) "
              f"{acc['miss_ok']} из {acc['miss_n']}")
        if len(g):
            print(f"разрыв на 45-70% длины: покрытие {g.mean():.2f} "
                  f"(медиана {np.median(g):.2f}), закрыт полностью (>=0.8) у "
                  f"{int((g >= 0.8).sum())} из {len(g)}")


def report_case(case: str, model: str) -> None:
    import SimpleITK as sitk
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cbct_backend as cb
    c = cb.get_case(case)
    sp = np.array(c["spacing"], float)
    vol = sitk.GetArrayFromImage(sitk.ReadImage(c["volume"]))
    mimg = sitk.ReadImage(c["masks"][model])
    raw = sitk.GetArrayFromImage(mimg)
    vol, raw = cb.apply_scanner_mirror(vol, raw, c)
    raw = cb.orient_sides(raw, mimg.GetDirection(), c)
    filt = cb.filter_mask(raw, vol.astype(np.float32), spacing=tuple(sp), canals=False)
    zup = cb.occlusal_up(filt)
    print(f"{case}/{model}: зуб | доход до -> после | каналы 55/66/77/88 до -> после | +вокс")
    for fdi in sorted(set(np.unique(filt)) & set(cb.TEETH_CLASSES)):
        pulp = filt == fdi + 100
        if fdi % 10 < 4 or not pulp.any():
            continue
        sgn = 1 if (fdi // 10 in (1, 2)) == zup else -1
        idx = np.argwhere((filt == fdi) | pulp)
        box = tuple(slice(a, b) for a, b in zip(np.maximum(idx.min(0) - 4, 0), idx.max(0) + 5))
        body = cb.canal_body((filt == fdi)[box], pulp[box])
        add = complete_canals(vol[box].astype(np.float32), body, pulp[box], sp, sgn)
        r0, c0 = _levels(body, pulp[box], sp, sgn)
        r1, c1 = _levels(body, pulp[box] | add, sp, sgn)
        print(f"  {fdi}: {r0:.2f} -> {r1:.2f} | {c0} -> {c1} | +{int(add.sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", type=int, default=0, help="томов ToothFairy3")
    ap.add_argument("--net", help="веса модели канала: сверить её тем же протоколом")
    ap.add_argument("--case")
    ap.add_argument("--model", default="sp")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if a.validate:
        validate(a.validate, a.net)
    if a.case:
        report_case(a.case, a.model)


if __name__ == "__main__":
    main()
