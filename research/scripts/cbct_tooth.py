"""Карточка зуба: срезы вдоль СОБСТВЕННОЙ оси зуба, а не осей аппарата.

Зачем отдельный модуль. Обычные срезы (z / y / x) идут по сетке тома, а зуб
стоит наклонно: на аксиальном срезе моляр попадает косо, корни и канал
расползаются по соседним срезам, и по такой картинке нельзя ни оценить длину
корня, ни проследить канал. Тот же приём, что в панораме: плоскость строится
по анатомии, а не по сетке.

Три плоскости на зуб:

  * `md` — вдоль оси зуба и мезио-дистального направления. Это вид прицельного
    снимка: оба контактных пункта и межзубная кость с обеих сторон.
  * `bl` — вдоль оси и вестибуло-язычного направления: толщина стенки, наклон
    корня, отношение к дну пазухи или каналу.
  * `ax` — поперечный на уровне шейки: сколько корней в сечении, форма канала.

Направления берутся из уже проверенных функций `cbct_perio`: ось зуба — главная
ось его вокселей (`tooth_axis`), мезиальное направление — сосед по номеру
(`site_dirs`). Дублировать их здесь нельзя: разъедутся с измерением уровня
кости, и врач увидит срез не в той плоскости, в которой посчитано число.

Запуск как CLI (кладёт PNG рядом, для глазной проверки):
    .venv/Scripts/python.exe research/scripts/cbct_tooth.py client4 --tooth 13
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402
import cbct_perio as P  # noqa: E402

PLANES = ("md", "bl", "ax")
PLANE_RU = {
    "md": "Мезио-дистальный",
    "bl": "Вестибуло-язычный",
    "ax": "Поперечный (шейка)",
}
# Что где на картинке — подпись обязана быть, иначе «слева» ничего не значит.
PLANE_HINT = {
    "md": "← мезиально · дистально →",
    "bl": "← язычно · вестибулярно →",
    "ax": "← мезиально · дистально →, вестибулярно вниз",
}

PAD_MM = 3.5        # запас поля вокруг габарита зуба: нужна кость вокруг
STEP_MM = 0.06      # шаг выборки, впятеро мельче вокселя (0.31 мм)
# 0.12 давало картинку 113x222 px на поле 13.6x26.6 мм, а вёрстка растягивает
# её втрое — врач видел крупные квадратные пиксели. Информации мельче вокселя
# в данных нет, но при 0.06 контур маски и градиент не рвутся на ступеньки.
MAX_PX = 1100       # потолок стороны картинки
AX_FROM_CROWN = 0.28   # где резать поперёк: доля длины зуба от коронки (шейка)
# Окно плотности. Было -200..2600, и это ВЫЖИГАЛО зуб: медиана плотности зуба
# у client5 равна 1900, то есть попадала в верхнюю четверть окна, а 15%
# вокселей зуба уходили выше потолка в чистый белый. На срезе зуб выглядел
# сплошным пятном — ни канала, ни границы эмаль/дентин, ни кариеса под
# пломбой. Замер по этому кейсу: зубы 1%..99% = 1002..3822, кость челюсти
# 330..2686, пульпа 5..2089. Окно 0..3400 покрывает все три: зуб не выжжен, губчатая кость ещё видна
# (при нижней границе 200 она уходила в чёрное на поперечном срезе).
HU_LO, HU_HI = 0.0, 3400.0

MIN_TOOTH_VOX = 300     # столько же, сколько в perio: меньше — огрызок маски


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


_frames = {"key": None, "data": None}


def frames_for(mask: np.ndarray, spacing: np.ndarray, key: str) -> dict:
    """Оси всех зубов кейса. Считается один раз: проход по маске дорогой.

    Центроиды берём ОДНИМ проходом по всем зубам сразу — 32 отдельных
    сравнения с полным томом (577x516x516) занимают десятки секунд.
    """
    if _frames["key"] == key and _frames["data"] is not None:
        return _frames["data"]
    z_up = cb.occlusal_up(mask)
    teeth_lo, teeth_hi = min(cb.TEETH_CLASSES), max(cb.TEETH_CLASSES)
    tm = (mask >= teeth_lo) & (mask <= teeth_hi)
    idx_all = np.array(np.nonzero(tm))          # (3, N) в вокселях
    lab = np.asarray(mask)[tm]
    out = {}
    centroids = {}
    for t in cb.TEETH_CLASSES:
        sel = lab == t
        if int(sel.sum()) < MIN_TOOTH_VOX:
            continue
        out[t] = {"idx": idx_all[:, sel]}
        centroids[t] = (out[t]["idx"].T * spacing).mean(0)
    # Вестибулярное направление — наружу от центра дуги. Центр берём по всем
    # зубам своей челюсти: у половины дуги он уехал бы вбок.
    arch_c = {}
    for jaw, quads in (("upper", (1, 2)), ("lower", (3, 4))):
        pts = [centroids[t] for t in centroids if t // 10 in quads]
        if pts:
            arch_c[jaw] = np.mean(pts, axis=0)
    for t, row in list(out.items()):
        upper = t // 10 in (1, 2)
        axis, origin = P.tooth_axis(row["idx"], spacing, upper, z_up)
        mesial = P.site_dirs(centroids, t, axis)
        if mesial is None:
            # Одинокий зуб в квадранте: мезиальное направление не построить.
            # Берём любую ось поперёк — карточка всё равно полезна, но
            # подписи «мезиально/дистально» тогда врать нельзя.
            mesial = _unit(np.cross(axis, [1.0, 0.0, 0.0]))
            row["dirs_known"] = False
        else:
            row["dirs_known"] = True
        c = centroids[t]
        ref = arch_c.get("upper" if upper else "lower", c)
        out_dir = c - ref
        out_dir = out_dir - axis * (out_dir @ axis)
        if np.linalg.norm(out_dir) < 1e-6:
            out_dir = np.cross(axis, mesial)
        buccal = _unit(out_dir)
        # Ортогонализуем мезиальное относительно оси и щёчного, чтобы тройка
        # была правой и плоскости не «поехали».
        mesial = _unit(np.cross(buccal, axis))
        row.update(axis=axis, origin=origin, center=c,
                   mesial=mesial, buccal=buccal, upper=upper, z_up=z_up)
    _frames.update(key=key, data=out)
    return out


def _plane_axes(fr: dict, plane: str):
    """(вправо, вниз, центр) для плоскости, в мм-координатах (z, y, x).

    Зуб рисуем АНАТОМИЧЕСКИ: у верхнего коронка внизу, у нижнего вверху —
    так же, как на панораме. Иначе врач сверяет карточку с панорамой и видит
    зуб вверх ногами.
    """
    axis, mesial, buccal = fr["axis"], fr["mesial"], fr["buccal"]
    down = axis if fr["upper"] else -axis     # axis смотрит К КОРОНКЕ
    if plane == "md":
        return -mesial, down, fr["center"]
    if plane == "bl":
        return buccal, down, fr["center"]
    # поперечный: на уровне шейки, «вестибулярно» вниз по экрану
    pts = fr["idx"].T * fr["spacing"]
    proj = (pts - fr["origin"]) @ axis
    lo, hi = float(proj.min()), float(proj.max())
    lvl = hi - AX_FROM_CROWN * (hi - lo)      # hi — коронковый конец
    off = fr["center"] - fr["origin"]
    ctr = fr["origin"] + axis * lvl + (off - axis * (off @ axis))
    return -mesial, buccal, ctr


def _extent(fr: dict, e_u: np.ndarray, e_v: np.ndarray, ctr: np.ndarray):
    """Габарит зуба вдоль пары направлений, мм, с запасом."""
    pts = fr["idx"].T * fr["spacing"] - ctr
    u, v = pts @ e_u, pts @ e_v
    w = float(u.max() - u.min()) + 2 * PAD_MM
    h = float(v.max() - v.min()) + 2 * PAD_MM
    return max(w, 8.0), max(h, 8.0)


def _sample(vol, mask, spacing, ctr, e_u, e_v, w_mm, h_mm, step):
    """Плоскость из тома: серый + метки, размер (nv, nu)."""
    nu = min(int(round(w_mm / step)), MAX_PX)
    nv = min(int(round(h_mm / step)), MAX_PX)
    u = (np.arange(nu) - (nu - 1) / 2) * (w_mm / max(nu - 1, 1))
    v = (np.arange(nv) - (nv - 1) / 2) * (h_mm / max(nv - 1, 1))
    pos = (ctr[None, None, :]
           + u[None, :, None] * e_u[None, None, :]
           + v[:, None, None] * e_v[None, None, :])
    coords = (pos / spacing).transpose(2, 0, 1)     # (3, nv, nu)
    g = ndi.map_coordinates(vol, coords, order=1, mode="constant",
                            cval=HU_LO, output=np.float32)
    m = ndi.map_coordinates(mask, coords, order=0, mode="constant", cval=0)
    return g, m.astype(np.int16)


def render(vol, mask, spacing, fdi: int, plane: str, key: str,
           contour: bool = True, offset_mm: float = 0.0) -> tuple:
    """PNG одной плоскости зуба + метаданные (мм на сторону).

    `offset_mm` — сдвиг плоскости поперёк себя. Одна плоскость через центр
    зуба показывает не всё: щёчный канал моляра лежит в 2-3 мм от нёбного, и
    на срединном срезе его просто нет. Врач ведёт ползунок, как на MPR.
    """
    from PIL import Image
    fr = frames_for(mask, spacing, key).get(int(fdi))
    if fr is None:
        return None, None
    fr = dict(fr, spacing=spacing)
    e_u, e_v, ctr = _plane_axes(fr, plane)
    # Размер поля считаем ДО сдвига: сдвиг идёт по нормали, на проекции зуба
    # на e_u/e_v он не влияет, зато кадр не должен «дышать» при прокрутке.
    w_mm, h_mm = _extent(fr, e_u, e_v, ctr)
    if offset_mm:
        normal = np.cross(e_u, e_v)     # оба вектора единичные и ортогональны
        ctr = ctr + normal * float(offset_mm)
    g, m = _sample(vol, mask, spacing, ctr, e_u, e_v, w_mm, h_mm, STEP_MM)

    gray = np.clip((g - HU_LO) / (HU_HI - HU_LO), 0, 1)
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=-1)
    focus = (int(fdi), int(fdi) + 100)
    rgb = (cb.overlay_contour(rgb, m, focus=focus) if contour
           else cb.overlay_fill(rgb, m))

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    meta = {"tooth": int(fdi), "plane": plane, "title": PLANE_RU[plane],
            "hint": (PLANE_HINT[plane] if fr["dirs_known"]
                     else "стороны не определены: у зуба нет соседа по номеру"),
            "w_mm": round(w_mm, 1), "h_mm": round(h_mm, 1),
            "px": [int(rgb.shape[1]), int(rgb.shape[0])]}
    return buf.getvalue(), meta


def card(vol, mask, spacing, fdi: int, key: str) -> dict | None:
    """Геометрия зуба для карточки. Клинику добавляет слой находок."""
    fr = frames_for(mask, spacing, key).get(int(fdi))
    if fr is None:
        return None
    fr = dict(fr, spacing=spacing)
    axis = fr["axis"]
    pts = fr["idx"].T * spacing
    proj = (pts - fr["origin"]) @ axis
    length = float(proj.max() - proj.min())
    vox = float(np.prod(spacing))
    n_pulp = int((np.asarray(mask) == int(fdi) + 100).sum())
    planes = []
    for p in PLANES:
        e_u, e_v, ctr = _plane_axes(fr, p)
        w_mm, h_mm = _extent(fr, e_u, e_v, ctr)
        # Насколько далеко имеет смысл уводить плоскость поперёк себя: до края
        # самого зуба плюс небольшой запас на кость вокруг.
        normal = np.cross(e_u, e_v)
        proj = (pts - ctr) @ normal
        rng = max(float(proj.max()), -float(proj.min())) + 2.0
        planes.append({
            "key": p, "title": PLANE_RU[p],
            "hint": (PLANE_HINT[p] if fr["dirs_known"] else
                     "стороны не определены: нет соседа по номеру"),
            "w_mm": round(w_mm, 1), "h_mm": round(h_mm, 1),
            "range_mm": round(min(rng, 12.0), 1),
        })
    # Центр зуба в вокселях — чтобы MPR встал перекрестием на выбранный зуб.
    center_vox = [int(round(v)) for v in (fr["center"] / spacing)]
    return {
        "tooth": int(fdi),
        "center_vox": center_vox,
        "length_mm": round(length, 1),
        "volume_mm3": round(fr["idx"].shape[1] * vox, 0),
        "pulp_mm3": round(n_pulp * vox, 1) if n_pulp else None,
        "dirs_known": bool(fr["dirs_known"]),
        "planes": planes,
    }


def present(mask: np.ndarray, spacing: np.ndarray, key: str) -> list:
    """Номера зубов, для которых карточка строится."""
    return sorted(frames_for(mask, spacing, key).keys())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case")
    ap.add_argument("--tooth", type=int, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=Path("."))
    a = ap.parse_args()
    c = cb.get_case(a.case)
    if c is None:
        raise SystemExit(f"нет кейса {a.case}")
    model = a.model or cb.mask_model_for(c)
    vol, mask = cb.arrays_for(c, model)
    spacing = np.array(c["spacing"], float)
    key = f"{a.case}:{model}"
    info = card(vol, mask, spacing, a.tooth, key)
    if info is None:
        raise SystemExit(f"зуб {a.tooth} не найден в маске ({model})")
    print(f"зуб {a.tooth}: длина {info['length_mm']} мм, "
          f"объём {info['volume_mm3']:.0f} мм3, пульпа {info['pulp_mm3']}")
    for p in PLANES:
        png, meta = render(vol, mask, spacing, a.tooth, p, key)
        out = a.out / f"tooth_{a.tooth}_{p}.png"
        out.write_bytes(png)
        print(f"  {p}: {meta['w_mm']}x{meta['h_mm']} мм -> {out}")


if __name__ == "__main__":
    main()
