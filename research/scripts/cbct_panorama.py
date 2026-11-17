"""Панорамная реконструкция (ОПТГ) из КЛКТ.

Врач читает панораму быстрее, чем крутит 3D: на ней сразу весь зубной ряд.
У Diagnocat это основной экран, и строится он из той же томограммы, что у нас.

Как устроено. Зубной ряд — подкова, поэтому плоским срезом его не показать:
любая плоскость пересекает дугу в двух точках и режет остальное вкось. Поэтому
разворачиваем ИЗОГНУТУЮ поверхность: ведём кривую по дуге, в каждой её точке
берём столбик вокселей поперёк дуги толщиной с фокальный слой и усредняем.
Получается изображение «длина вдоль дуги × высота», то есть ровно панорама.

Ключевые решения и почему так:

* Дуга строится ПО ЦЕНТРАМ ЗУБОВ, а не по плотности. Плотность в аксиальном
  срезе даёт и позвоночник, и ветви челюсти; центры зубов уже есть в маске и
  лежат ровно там, где нужно.
* Порядок точек берётся из НУМЕРАЦИИ (FDI), а не из геометрии. Сортировать по
  углу или по координате нельзя: у подковы это даёт перехлёсты на молярах.
  Та же причина, по которой на сканах дуга строится сплайном по порядку зубов,
  а не параболой (см. `ios_geometry.fit_arch` и CLAUDE.md).
* Верх и низ объединяются в ОДНУ кривую (среднее одноимённых позиций): у
  настоящей ОПТГ фокальный слой тоже один на обе челюсти.
* Усреднение вдоль столбика, а не максимум. Рентген — это сумма ослабления по
  лучу, и среднее к ней ближе; максимум даёт неестественно яркие пломбы и
  теряет корни.
* Направление «вверх» берётся из данных (`cb.occlusal_up`). Жёстко задавать
  ось нельзя: ориентация тома у разных аппаратов разная, на этом уже
  разваливалось измерение уровня кости.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_panorama.py client4
    .venv/Scripts/python.exe research/scripts/cbct_panorama.py client2 --out pano.png
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402

MIN_TOOTH_VOX = 300
MIN_ARCH_VOX = 2000     # для ПОСТРОЕНИЯ дуги: огрызок даёт бессмысленный центр
MAX_Z_DEV_MM = 15.0     # промах по высоте относительно своей челюсти
MIDLINE_TOL_MM = 6.0    # запас у средней линии: резцы стоят вплотную к ней
ARC_MIN_MM, ARC_MAX_MM = 80.0, 220.0   # вменяемая длина зубного ряда
CROP_PAD_MM = 22.0      # запас по высоте над и под зубами
THICKNESS_MM = 24.0     # фокальный слой; у плёночной ОПТГ примерно столько же
DEPTH_STEPS = 25        # столбиков поперёк дуги на один пиксель ширины
STEP_MM = 0.35          # шаг вдоль дуги: чуть мельче вокселя, чтобы не рвать
SMOOTH = 8.0            # сглаживание сплайна дуги
MARGIN_MM = 8.0         # запас дуги за последним зубом с каждой стороны


def _order_index(fdi: int) -> int:
    """Позиция зуба вдоль дуги: справа налево, независимо от челюсти.

    Квадранты 1 и 4 — правая сторона пациента, 2 и 3 — левая (ISO 3950).
    Восьмёрка правая -> -8, центральный резец правый -> -1, левый -> +1.
    """
    q, n = fdi // 10, fdi % 10
    return -n if q in (1, 4) else n


def arch_points(mask: np.ndarray, spacing: np.ndarray):
    """Точки дуги в мм (плоскость y, x), центры зубов и список отвергнутых.

    Строить дугу прямо по центрам зубов НЕЛЬЗЯ: сегментация даёт грубые
    промахи, и один такой промах утаскивает всю кривую. Замер на client4:
    зубы 43, 46, 47 стоят на z 87-90 при остальных 29-63 (выше верхней
    челюсти — анатомически невозможно, у 47 при этом 11806 вокселей), а зубы
    25, 33, 37 оказались на ПРОТИВОПОЛОЖНОЙ стороне (25 на x=58 при соседях
    24 и 26 на 99 и 103). Дуга из-за них металась через весь череп, и
    панорама выходила задвоенной.

    Поэтому три отсева, каждый по своему признаку:
      * мелкие огрызки (< MIN_ARCH_VOX) — их центр не значит ничего;
      * промах по высоте: зуб далеко от медианы СВОЕЙ челюсти;
      * промах по стороне: правые зубы обязаны лежать по одну сторону от
        средней линии, левые по другую.
    """
    tooth_pt, kept = {}, {}
    for t in cb.TEETH_CLASSES:
        bm = mask == t
        n = int(bm.sum())
        if n < MIN_TOOTH_VOX:
            continue
        idx = np.array(np.where(bm))
        tooth_pt[t] = (idx.T * spacing).mean(0)      # (z, y, x) в мм
        if n >= MIN_ARCH_VOX:
            kept[t] = tooth_pt[t]

    rejected = []
    # --- отсев по высоте, внутри своей челюсти ---
    for jaw in ((1, 2), (3, 4)):
        grp = {t: c for t, c in kept.items() if t // 10 in jaw}
        if len(grp) < 3:
            continue
        med = float(np.median([c[0] for c in grp.values()]))
        for t, c in grp.items():
            if abs(c[0] - med) > MAX_Z_DEV_MM:
                rejected.append((t, f"высота {c[0]:.0f} мм при медиане "
                                    f"{med:.0f} по своей челюсти"))
                kept.pop(t, None)

    # --- отсев по стороне: средняя линия как медиана x всех зубов ---
    if len(kept) >= 6:
        mid = float(np.median([c[2] for c in kept.values()]))
        for t, c in list(kept.items()):
            right = t // 10 in (1, 4)
            # запас MIDLINE_TOL_MM: резцы стоят вплотную к средней линии
            wrong = ((c[2] > mid + MIDLINE_TOL_MM) if right
                     else (c[2] < mid - MIDLINE_TOL_MM))
            if wrong:
                rejected.append((t, f"сторона не та: x={c[2]:.0f} при средней "
                                    f"линии {mid:.0f}"))
                kept.pop(t, None)

    by_pos: dict[int, list] = {}
    for t, c in kept.items():
        by_pos.setdefault(_order_index(t), []).append(c[1:])
    if len(by_pos) < 4:
        return None, tooth_pt, rejected
    order = sorted(by_pos)
    pts = np.array([np.mean(by_pos[k], axis=0) for k in order])  # (N, 2) y, x

    # Фильтра «по соседям» здесь БЫТЬ НЕ ДОЛЖНО. Пробовали: у client4
    # отсутствуют 15 и 16, а 14 и 17 — огрызки, поэтому между 18 и 13 лежит
    # законный разрыв в три позиции. Правило приняло его за выброс и дальше
    # рубило по цепочке, отвергнув все двенадцать точек. Разрыв от удалённых
    # зубов — норма, его и должен перекрывать сплайн.
    return pts, tooth_pt, rejected


def arch_curve(pts: np.ndarray):
    """Гладкая кривая по точкам дуги, равномерно по длине. -> (точки, нормали)."""
    from scipy.interpolate import splprep, splev
    # k=3 требует минимум 4 точки; при малом числе зубов падаем до линейной
    k = 3 if len(pts) >= 4 else max(len(pts) - 1, 1)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=SMOOTH * len(pts), k=k)
    u = np.linspace(0, 1, 2000)
    cy, cx = splev(u, tck)
    curve = np.stack([cy, cx], 1)
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    # Продлеваем прямыми за крайние зубы: иначе восьмёрки обрезаются по центру.
    total = arc[-1]
    n_out = int((total + 2 * MARGIN_MM) / STEP_MM)
    want = np.linspace(-MARGIN_MM, total + MARGIN_MM, n_out)
    inside = np.clip(want, 0, total)
    samp = np.stack([np.interp(inside, arc, curve[:, 0]),
                     np.interp(inside, arc, curve[:, 1])], 1)
    # касательная -> нормаль в плоскости
    tang = np.gradient(samp, axis=0)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    # за краями идём по прямой вдоль касательной крайней точки
    out_lo, out_hi = want < 0, want > total
    if out_lo.any():
        samp[out_lo] = samp[out_lo.sum()] + \
            np.outer(want[out_lo] - 0, tang[out_lo.sum()])
    if out_hi.any():
        j = len(want) - out_hi.sum() - 1
        samp[out_hi] = samp[j] + np.outer(want[out_hi] - total, tang[j])
    return samp, norm, want


def render(vol: np.ndarray, mask: np.ndarray, spacing: np.ndarray,
           thickness_mm: float = THICKNESS_MM) -> tuple:
    """Панорама и разметка: где на ней какой зуб.

    -> (изображение float32 (H, W), meta dict)
    """
    pts, tooth_pt, rejected = arch_points(mask, spacing)
    if pts is None:
        raise ValueError("зубов слишком мало, дугу не построить")
    samp, norm, along = arch_curve(pts)
    # Длина дуги — дешёвая проверка вменяемости. У взрослого зубной ряд
    # 110-150 мм; 486 мм мы уже видели, когда кривую утаскивали промахи
    # сегментации, и панорама выходила задвоенной. Лучше отказ, чем картинка,
    # по которой врач будет ставить диагноз.
    arc_len = float(np.linalg.norm(np.diff(samp, axis=0), axis=1).sum())
    if not (ARC_MIN_MM <= arc_len <= ARC_MAX_MM):
        raise ValueError(
            f"дуга длиной {arc_len:.0f} мм — не похоже на зубной ряд "
            f"(ждём {ARC_MIN_MM:.0f}-{ARC_MAX_MM:.0f} мм). "
            f"Скорее всего сегментация промахнулась: {rejected}")

    D = vol.shape[0]
    zs = np.arange(D, dtype=np.float32)
    depth = np.linspace(-thickness_mm / 2, thickness_mm / 2, DEPTH_STEPS)

    acc = np.zeros((D, len(samp)), np.float32)
    # Идём по глубине и копим сумму: держать все координаты разом — это
    # сотни мегабайт на кейс, а кэш массивов и так рассчитан на один том.
    zz = np.repeat(zs[:, None], len(samp), axis=1)
    for d in depth:
        p = samp + norm * d                      # (W, 2) в мм
        yy = np.repeat((p[:, 0] / spacing[1])[None, :], D, axis=0)
        xx = np.repeat((p[:, 1] / spacing[2])[None, :], D, axis=0)
        acc += ndi.map_coordinates(vol, [zz, yy, xx], order=1,
                                   mode="constant", cval=cb.HU_MIN)
    img = acc / len(depth)

    # Переворот. Строка 0 массива — это z=0, а при показе она оказывается
    # СВЕРХУ. Значит когда z растёт вверх (occlusal_up), верхняя челюсть уедет
    # вниз кадра, и панорама выйдет вверх ногами: пазухи под зубами вместо
    # над ними. Поэтому переворачиваем именно в этом случае.
    flip = cb.occlusal_up(mask)
    if flip:
        img = img[::-1]

    # Обрезка по высоте. В томе 180 мм, из них челюсти занимают около 90;
    # остальное — свод черепа и шея, на ОПТГ их нет и врачу они мешают.
    zt = [c[0] / spacing[0] for c in tooth_pt.values()]
    if zt:
        lo_v, hi_v = min(zt), max(zt)
        pad = CROP_PAD_MM / spacing[0]
        r0, r1 = int(max(lo_v - pad, 0)), int(min(hi_v + pad, D))
        if flip:
            r0, r1 = D - int(min(hi_v + pad, D)), D - int(max(lo_v - pad, 0))
        if r1 - r0 > 50:
            img = img[r0:r1]
        else:
            r0 = 0
    else:
        r0 = 0

    # где какой зуб: ближайшая точка дуги к его центру
    cols = {}
    for t, c in tooth_pt.items():
        j = int(np.argmin(np.linalg.norm(samp - c[1:], axis=1)))
        row = c[0] / spacing[0]
        if flip:
            row = D - 1 - row
        cols[t] = {"x": j, "y": float(row - r0)}   # r0 — сдвиг от обрезки
    meta = {"rejected": [{"tooth": t, "why": w} for t, w in rejected],
            "width": int(img.shape[1]), "height": int(img.shape[0]),
            "teeth": cols, "thickness_mm": thickness_mm,
            "mm_per_px_x": STEP_MM, "mm_per_px_y": float(spacing[0])}
    return img, meta


def to_png(img: np.ndarray, lo: float = -200.0, hi: float = 2200.0) -> bytes:
    """Серое изображение с окном по плотности.

    Окно уже, чем у срезов (HU_MIN..HU_MAX): после усреднения по толщине
    контраст падает, и широкое окно даёт равномерно серую кашу.
    """
    from PIL import Image
    g = np.clip((img - lo) / (hi - lo), 0, 1)
    a = (g * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


def build(c: dict, model: str, thickness_mm: float = THICKNESS_MM) -> tuple:
    """Панорама кейса. -> (png bytes, meta)"""
    vol, mask = cb.arrays_for(c, model)
    img, meta = render(vol, mask, np.array(c["spacing"], float), thickness_mm)
    return to_png(img), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case")
    ap.add_argument("--model", default="tf3")
    ap.add_argument("--thickness", type=float, default=THICKNESS_MM)
    ap.add_argument("--out", type=Path,
                    default=Path("research/output/panorama.png"))
    a = ap.parse_args()

    c = cb.get_case(a.case)
    png, meta = build(c, a.model, a.thickness)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(png)
    print(f"панорама: {a.out} ({len(png)/1024:.0f} КБ), "
          f"{meta['width']}x{meta['height']} px")
    print(f"зубов размечено: {len(meta['teeth'])}, "
          f"толщина слоя {meta['thickness_mm']:.0f} мм")
    if meta["rejected"]:
        print("не участвовали в построении дуги (промах сегментации):")
        for r in meta["rejected"]:
            print(f"   зуб {r['tooth']}: {r['why']}")


if __name__ == "__main__":
    main()
