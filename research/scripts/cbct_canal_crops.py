"""Вырезы зубов для модели канала: ToothFairy3 и Pulpy3D -> обучающий набор.

    .venv\\Scripts\\python.exe research\\scripts\\cbct_canal_crops.py --source tf3 --limit 20
    .venv\\Scripts\\python.exe research\\scripts\\cbct_canal_crops.py --source pulpy3d

ЗАЧЕМ. Пульпа нашей модели не доходит до верхушки (72-91% длины зуба), и это
НЕ ошибка сегментации корня: на эталоне DentVoxel корень кончается там же,
где у разметчика (расхождение 0.26 мм по 90-му перцентилю). Не дорисован сам
канал, потому что он не дорисован у учителя: у ToothFairy3 пульпа доходит до
85% длины зуба лишь у 46% зубов. Значит нужна вторая модель — на ВЫРЕЗЕ
ЗУБА, обученная там, где канал размечен до конца.

ВЫРЕЗ ВЫПРЯМЛЕН И НОРМИРОВАН ПО ЗУБУ. Зуб выбирается вдоль собственной оси в
сетку фиксированного размера: уровни от коронки к верхушке, поперёк — радиус
зуба с запасом. Три следствия:
  * масштаб перестаёт зависеть от вокселя аппарата (у Pulpy3D он вообще
    потерян в метаданных — везде записан 1.0);
  * наклон зуба не мешает: канал всегда идёт «вниз» по сетке;
  * сеть видит зуб целиком в одном разрешении, а не кусок большого тома.

ГОЧА, СТОИВШАЯ ОБУЧЕНИЯ. Вторым каналом сеть получает не зуб, а ТЕЛО
зуба — зуб вместе с пульпой, замкнутый и с заполненными полостями. Сначала
я подавал маску зуба как есть, а она и у эталона, и у нашей модели пульпу
НЕ включает: на её месте дырка. Сеть на второй же эпохе выдала Dice 0.996 —
она просто обводила дырку во входе, то есть читала ответ. На приёме такой
подсказки нет: там тело строится из зуба и НЕДОРИСОВАННОЙ пульпы модели
(`cbct_backend.canal_body`), и никакой дырки в нужном месте там не будет.

ОТКУДА ЗУБЫ. У ToothFairy3 — из эталона. У Pulpy3D зубов нет вовсе (классы
только пульпа и нерв), поэтому их даёт наша модель `sp`. Проверка, что это
законно: эталонная пульпа Pulpy3D попадает внутрь наших зубов на 98-100%.

МАСШТАБ Pulpy3D. В файлах воксель 1.0, реальный ~0.31 мм. Определён двумя
линейками: длина зуба нашей же меркой (медиана 20.1 мм на клинике при
известном вокселе против 25.0 у Pulpy3D при пробном 0.38 -> 0.31) и
нижнечелюстной канал (дал 0.38, но канал в этом эталоне размечен щедро —
линейка завышает). Для нарезки масштаб не важен (вырез нормирован), важен
для инференса зубов: nnU-Net ресемплит по spacing.

ОТБОР. В обучение идут только зубы, где канал размечен ДО ВЕРХУШКИ
(`MIN_REACH`): остальные учили бы сеть останавливаться на середине — ровно
тому, от чего уходим. Отсев записывается в `meta.jsonl`, чтобы потом можно
было пересобрать с другим порогом, не гоняя инференс заново.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

TF3 = ROOT / "research_3d/nnUNet_raw/Dataset113_ToothFairy3"
PULPY = (ROOT / "research_3d/datasets/kagglehub_cache/datasets/gingeryy/pulpy3d"
         / "versions/1/Dataset007_Pulpy3D_Full")
OUT = ROOT / "research_3d/datasets/canal_crops"

PULPY_SPACING = 0.31          # см. «МАСШТАБ Pulpy3D» выше
TOOTH_MODEL = "sp"

GRID = (128, 64, 64)          # уровни вдоль оси, поперёк, поперёк
MARGIN_TOP = 0.04             # запас над коронкой, долей длины зуба
MARGIN_APEX = 0.10            # запас за верхушкой: там и живёт «недоход»
RADIUS_K = 1.35               # запас вбок от радиуса зуба
MIN_TOOTH_VOX = 2000
MIN_PULP_VOX = 100
MIN_REACH = 0.85              # канал считается доведённым до верхушки
VAL_PERCENT = 15              # разбиение по ТОМУ, а не по зубу


def _axis(pts: np.ndarray, apex_sign: int):
    ctr = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
    ax = vt[0] if np.sign(vt[0][0]) == apex_sign else -vt[0]
    ref = np.array([0.0, 1.0, 0.0]) if abs(ax[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(ax, ref); u /= np.linalg.norm(u)
    return ctr, ax, u, np.cross(ax, u)


def grid_of(body, spacing, apex_sign: int):
    """Координаты сетки выреза (в вокселях кропа) и её описание.

    Вынесено отдельно, потому что этой же сеткой предсказание возвращается из
    выреза в том на приёме (`cbct_canal_apply.py`). Считать её там второй
    раз нельзя: разойдётся геометрия обучения и применения — в проекте такое
    уже было с `wbf` в трёх копиях.
    """
    sp = np.asarray(spacing, float)
    pts = np.argwhere(body) * sp
    ctr, ax, u, w = _axis(pts, apex_sign)
    rel = pts - ctr
    t = rel @ ax
    radial = np.linalg.norm(rel - np.outer(t, ax), axis=1)
    length = float(t.max() - t.min())
    lo = float(t.min()) - MARGIN_TOP * length
    hi = float(t.max()) + MARGIN_APEX * length
    rad = float(np.percentile(radial, 99)) * RADIUS_K

    a = np.linspace(lo, hi, GRID[0])
    b = np.linspace(-rad, rad, GRID[1])
    c = np.linspace(-rad, rad, GRID[2])
    A, B, C = np.meshgrid(a, b, c, indexing="ij")
    pos = ctr + A[..., None] * ax + B[..., None] * u + C[..., None] * w
    info = {"length_mm": round(length, 2),
            "radius_mm": round(rad, 2),
            "step_axial_mm": round((hi - lo) / (GRID[0] - 1), 4),
            "step_lateral_mm": round(2 * rad / (GRID[1] - 1), 4),
            # Полное описание рамки: без него точку из сетки не вернуть в
            # воксели тома, а разметка врача обязана пережить смену модели.
            "frame": {"ctr": ctr.tolist(), "ax": ax.tolist(),
                      "u": u.tolist(), "w": w.tolist(),
                      "a0": float(lo), "da": float((hi - lo) / (GRID[0] - 1)),
                      "b0": float(-rad), "db": float(2 * rad / (GRID[1] - 1)),
                      "c0": float(-rad), "dc": float(2 * rad / (GRID[2] - 1))}}
    return np.moveaxis(pos / sp, -1, 0), info


def normalize(img: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Плотность в единицах дентина ЭТОГО зуба: (img - медиана) / разброс.

    Шкала КЛКТ не количественная, и абсолютные значения между аппаратами не
    переносятся. Формула одна на обучение (`train_canal_net.Crops`) и на
    приём — разъедутся, и сеть на приёме получит другой вход.
    """
    m = body > 0.5
    if m.sum() > 100:
        med = float(np.median(img[m]))
        mad = float(np.median(np.abs(img[m] - med))) * 1.4826
    else:
        med, mad = 0.0, 500.0
    return (img.astype(np.float32) - med) / max(mad, 50.0)


def straighten(vol, body, pulp, spacing, apex_sign: int):
    """Вырез вдоль оси зуба в сетку GRID. Возвращает (img, body, pulp, инфо)."""
    coords, info = grid_of(body, spacing, apex_sign)
    img = ndi.map_coordinates(vol.astype(np.float32), coords, order=1,
                              mode="constant", cval=float(vol.min()))
    tt = ndi.map_coordinates(body.astype(np.float32), coords, order=1) > 0.5
    pp = ndi.map_coordinates(pulp.astype(np.float32), coords, order=1) > 0.5
    return img, tt, pp, info


def reach_of(tooth, pulp, spacing, apex_sign: int) -> float:
    """Докуда доходит пульпа, в долях длины зуба (0 — нет, 1 — до верхушки)."""
    sp = np.asarray(spacing, float)
    pts = np.argwhere(tooth) * sp
    if not len(pts) or not pulp.any():
        return 0.0
    ctr, ax, *_ = _axis(pts, apex_sign)
    t = (pts - ctr) @ ax
    p = (np.argwhere(pulp) * sp - ctr) @ ax
    return float((p.max() - t.min()) / (t.max() - t.min()))


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return "val" if h % 100 < VAL_PERCENT else "train"


def body_of(tooth: np.ndarray, pulp: np.ndarray) -> np.ndarray:
    """Тело зуба: зуб вместе с пульпой, замкнутое, с заполненными полостями.

    Ровно то же, что строит `cbct_backend.canal_body` на приёме. Подавать
    сети «зуб без пульпы» нельзя — см. ГОЧУ в докстринге модуля."""
    return ndi.binary_fill_holes(
        ndi.binary_closing(tooth | pulp, iterations=2)) | tooth | pulp


def save_crop(out: Path, name: str, img, body, pulp) -> Path:
    p = out / f"{name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        img=np.clip(img, -1000, 6000).astype(np.int16),
        body=body.astype(np.uint8),
        pulp=pulp.astype(np.uint8))
    return p


def _teeth_of_volume(mask, classes):
    return [t for t in classes if (mask == t).sum() >= MIN_TOOTH_VOX]


def run_tf3(limit: int, out: Path, meta) -> int:
    import cbct_backend as cb
    stems = sorted(p.name[:-7] for p in (TF3 / "labelsTr_fdi").glob("*.nii.gz"))
    if limit:
        stems = stems[:limit]
    n = 0
    for i, stem in enumerate(stems, 1):
        img = sitk.ReadImage(str(TF3 / "imagesTr" / f"{stem}_0000.nii.gz"))
        vol = sitk.GetArrayFromImage(img)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(
            str(TF3 / "labelsTr_fdi" / f"{stem}.nii.gz")))
        sp = np.array(img.GetSpacing(), float)[::-1]
        zup = cb.occlusal_up(mask)
        split = split_of(f"tf3/{stem}")
        for fdi in _teeth_of_volume(mask, cb.TEETH_CLASSES):
            tooth, pulp = mask == fdi, mask == fdi + 100
            if pulp.sum() < MIN_PULP_VOX:
                continue
            sgn = 1 if (fdi // 10 in (1, 2)) == zup else -1
            r = reach_of(tooth, pulp, sp, sgn)
            rec = {"source": "tf3", "volume": stem, "fdi": int(fdi),
                   "reach": round(r, 3), "split": split,
                   "used": bool(r >= MIN_REACH)}
            if rec["used"]:
                idx = np.argwhere(tooth | pulp)
                box = tuple(slice(a, b) for a, b in
                            zip(np.maximum(idx.min(0) - 6, 0),
                                np.minimum(idx.max(0) + 7, mask.shape)))
                im, bb, pp, info = straighten(vol[box], body_of(tooth[box], pulp[box]),
                                             pulp[box], sp, sgn)
                name = f"{split}/tf3_{stem}_{fdi}"
                save_crop(out, name, im, bb, pp)
                rec.update(info, file=f"{name}.npz")
                n += 1
            meta.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta.flush()
        print(f"  tf3 {i}/{len(stems)} {stem}: вырезов всего {n}", flush=True)
    return n


def run_pulpy(limit: int, out: Path, meta) -> int:
    import cbct_backend as cb
    files = sorted((PULPY / "imagesTr").glob("*.nii"))
    if limit:
        files = files[:limit]
    n = 0
    for i, f in enumerate(files, 1):
        stem = f.stem.replace("_0000", "")
        img = sitk.ReadImage(str(f))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(PULPY / "labelsTr" / f"{stem}.nii")))
        sp = (PULPY_SPACING,) * 3
        props = {"spacing": sp, "sitk_stuff": {"spacing": sp, "origin": img.GetOrigin(),
                                               "direction": img.GetDirection()}}
        pred = cb.get_predictor(TOOTH_MODEL, cb._fits_on_gpu(TOOTH_MODEL, vol.shape)
                                ).predict_single_npy_array(vol[None], props, None, None, False)
        pred = cb.remap_prediction(pred, TOOTH_MODEL, spacing=sp)
        zup = cb.occlusal_up(pred)
        split = split_of(f"pulpy3d/{stem}")
        gt_pulp = gt == 1
        for fdi in _teeth_of_volume(pred, cb.TEETH_CLASSES):
            tooth = pred == fdi
            # Пульпа в этом наборе размечена одним классом на весь снимок:
            # берём ту её часть, что лежит в теле ЭТОГО зуба (зуб плюс щели
            # внутри него — недорисованный канал модель метит фоном).
            body = ndi.binary_fill_holes(ndi.binary_closing(tooth, iterations=2)) | tooth
            pulp = gt_pulp & body
            if pulp.sum() < MIN_PULP_VOX:
                continue
            sgn = 1 if (fdi // 10 in (1, 2)) == zup else -1
            r = reach_of(tooth, pulp, sp, sgn)
            rec = {"source": "pulpy3d", "volume": stem, "fdi": int(fdi),
                   "reach": round(r, 3), "split": split,
                   "used": bool(r >= MIN_REACH)}
            if rec["used"]:
                idx = np.argwhere(body)
                box = tuple(slice(a, b) for a, b in
                            zip(np.maximum(idx.min(0) - 6, 0),
                                np.minimum(idx.max(0) + 7, pred.shape)))
                im, bb, pp, info = straighten(vol[box], body[box], pulp[box], sp, sgn)
                name = f"{split}/pulpy3d_{stem}_{fdi}"
                save_crop(out, name, im, bb, pp)
                rec.update(info, file=f"{name}.npz")
                n += 1
            meta.write(json.dumps(rec, ensure_ascii=False) + "\n")
        meta.flush()
        print(f"  pulpy3d {i}/{len(files)} {stem}: зубов {len(_teeth_of_volume(pred, cb.TEETH_CLASSES))}, "
              f"вырезов всего {n}", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("tf3", "pulpy3d"), required=True)
    ap.add_argument("--limit", type=int, default=0, help="сколько томов (0 — все)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"meta_{a.source}.jsonl", "a", encoding="utf-8") as meta:
        n = run_tf3(a.limit, out, meta) if a.source == "tf3" \
            else run_pulpy(a.limit, out, meta)
    print(f"\nготово: вырезов {n}; метки {out / f'meta_{a.source}.jsonl'}")


if __name__ == "__main__":
    main()
