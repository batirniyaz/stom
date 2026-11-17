"""Модель канала на приёме: канал дорисовывает СЕТЬ, а не порог плотности.

    .venv\\Scripts\\python.exe research\\scripts\\cbct_canal_apply.py --case client5
    .venv\\Scripts\\python.exe research\\scripts\\cbct_canal_apply.py --case client5 --save

ЗАЧЕМ. Пульпа основной модели не доходит до верхушки (72-91% длины зуба), и
сейчас канал дотягивается трассировкой по плотности (`cbct_canals.py`). У неё
есть предел: порог живёт в единицах шума дентина, и там, где канал бледнее
шума, трасса либо рвётся, либо разрастается. Модель канала
(`train_canal_net.py`) учится на вырезах, где канал размечен до верхушки, и
должна брать те же зубы без порога.

ЧТО СЧИТАЕТСЯ. Для каждого зуба три варианта канала на ОДНОЙ маске:
маска модели как есть, плюс трассировка по плотности, плюс сеть. Сравниваются
доходом канала (доля длины зуба, до которой дотянулась пульпа) и объёмом.
Числа печатаются по зубам, потому что средним такое сравнение не читается:
врачу важен КОНКРЕТНЫЙ зуб с четырьмя каналами.

ГЕОМЕТРИЯ ВЫРЕЗА БЕРЁТСЯ ИЗ НАРЕЗКИ (`cbct_canal_crops.grid_of`), а не
считается здесь заново: разойдись она с обучением — сеть получит другой вход,
и это не упадёт, а тихо ухудшит результат. По той же причине нормировка
плотности одна на оба места (`cbct_canal_crops.normalize`).

ПРЕДСКАЗАНИЕ ОБРЕЗАЕТСЯ ТЕЛОМ ЗУБА. Сеть видит вырез с запасом за верхушкой
(`MARGIN_APEX`), и канал, «продолженный» в периодонтальную щель, был бы
находкой о том, чего в зубе нет.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import cbct_backend as cb          # noqa: E402
import cbct_canal_crops as cc      # noqa: E402
from train_canal_net import UNet3D  # noqa: E402

WEIGHTS = ROOT / "research/output/canal_net/best.pt"
THR = 0.5
MIN_PULP_VOX = 30        # меньше — не пульпа, а крошка сегментации
MIN_ADD_VOX = 20         # приращение меньше этого в 3D не видно


def load_net(weights: Path, device) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(weights, map_location=device, weights_only=False)
    net = UNet3D().to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, {"epoch": ck.get("epoch"), "metrics": ck.get("metrics", {})}


@torch.no_grad()
def canal_in_tooth(net, device, vol_box, body, pulp, spacing, apex_sign: int,
                   thr: float = THR, want_grid: bool = False):
    """Канал зуба по сети, в координатах кропа (той же формы, что body).

    При `want_grid` вторым значением отдаётся сам вырез (плотность, тело,
    предсказание и координаты сетки) — для картинки на глаз.
    """
    from scipy import ndimage as ndi

    coords, _ = cc.grid_of(body, spacing, apex_sign)
    img = ndi.map_coordinates(vol_box.astype(np.float32), coords, order=1,
                              mode="constant", cval=float(vol_box.min()))
    bg = ndi.map_coordinates(body.astype(np.float32), coords, order=1) > 0.5
    x = np.stack([cc.normalize(img, bg), bg.astype(np.float32)])
    prob = torch.sigmoid(net(torch.from_numpy(x)[None].to(device)))[0, 0].cpu().numpy()
    pred = (prob > thr) & bg

    out = np.zeros(body.shape, bool)
    if pred.any():
        # Обратное отображение: шаг сетки (0.15-0.19 мм) мельче вокселя (0.31),
        # поэтому округления хватает — дырок от разреженности не остаётся.
        idx = np.rint(coords[:, pred]).astype(int)
        for k in range(3):
            np.clip(idx[k], 0, body.shape[k] - 1, out=idx[k])
        out[idx[0], idx[1], idx[2]] = True
    out &= body
    if want_grid:
        return out, {"img": img, "body": bg, "pred": pred, "coords": coords}
    return out


def reach(tooth: np.ndarray, pulp: np.ndarray, spacing, apex_sign: int) -> float:
    return cc.reach_of(tooth, pulp, spacing, apex_sign)


# Окно плотности выреза: то же, что у карточки зуба — при широком окне зуб
# выгорает в белое и канал на картинке не виден (замер в CLAUDE.md).
PNG_HU = (0, 3400)
SCALE = 3


def _panel(img2d, inside, layers: list[tuple[np.ndarray, tuple]]):
    """Серый кадр плотности с цветными слоями поверх.

    Вне зуба плотность не показываем: проекция минимума по всему вырезу берёт
    окружающий воздух (-1000) и кадр выходит чёрным — первая отрисовка именно
    так и вышла.
    """
    g = np.clip((img2d - PNG_HU[0]) / (PNG_HU[1] - PNG_HU[0]), 0, 1)
    rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb[~inside] = 18
    for m, color in layers:
        rgb[m] = (0.35 * rgb[m] + 0.65 * np.array(color)).astype(np.uint8)
    return rgb


def _font(size: int = 13):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "tahoma.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()      # без кириллицы, но лучше, чем падение


def tooth_png(grid, pulp_g, den_g, fdi: int, rows: dict) -> "Image.Image":
    """Две проекции выреза: канал сети, трассировки и модели в одном кадре.

    Проекция, а не срез: у моляра каналы лежат в разных плоскостях, и один
    срез показал бы один канал из четырёх. По плотности берётся МИНИМУМ —
    канал темнее дентина, и минимум его как раз проявляет.
    """
    from PIL import Image, ImageDraw

    from scipy import ndimage as ndi

    body = grid["body"]
    # Проекция берётся по ВНУТРЕННОСТИ зуба, отступив от поверхности: у края
    # плотность падает (частичный объём), и минимум по всей толщине давал
    # ровное серое поле, в котором канал не виден вовсе — первая отрисовка
    # чистой панели именно так и вышла.
    core = ndi.binary_erosion(body, iterations=2)
    if not core.any():
        core = body
    panels = []
    for ax in (2, 1):
        inside = body.any(axis=ax)
        img2d = np.where(core, grid["img"], np.inf).min(axis=ax)
        img2d = np.where(np.isfinite(img2d), img2d,
                         np.where(body, grid["img"], np.inf).min(axis=ax))
        img2d = np.where(inside, img2d, PNG_HU[0])
        layers = [(den_g.any(axis=ax) & ~pulp_g.any(axis=ax), (80, 220, 80)),
                  (grid["pred"].any(axis=ax) & ~pulp_g.any(axis=ax), (240, 70, 70)),
                  (pulp_g.any(axis=ax), (90, 150, 255))]
        # Первым — тот же кадр БЕЗ разметки: иначе по картинке нельзя
        # проверить, идёт ли нарисованный канал по тёмной линии.
        panels.append(_panel(img2d, inside, []))
        panels.append(_panel(img2d, inside, layers))
    pad, top = 8, 40
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + pad * (len(panels) - 1)
    sheet = np.zeros((h, w, 3), np.uint8)
    x = 0
    for p in panels:
        sheet[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + pad
    im = Image.new("RGB", (w * SCALE, h * SCALE + top), (12, 12, 12))
    im.paste(Image.fromarray(sheet).resize((w * SCALE, h * SCALE), Image.NEAREST),
             (0, top))
    d = ImageDraw.Draw(im)
    d.text((8, 6), f"зуб {fdi}   доход: модель {rows['reach_base']:.2f} -> сеть "
                   f"{rows['reach_net']:.2f} / плотность {rows['reach_den']:.2f}",
           fill=(255, 255, 255), font=_font(15))
    d.text((8, 24), "слева от каждой пары — снимок как есть; синее — пульпа модели, "
                    "красное — сеть, зелёное — трассировка",
           fill=(190, 190, 190), font=_font(12))
    return im


def apply_case(case_id: str, model: str, net, device, thr: float = THR,
               save: bool = False, png: Path | None = None,
               png_teeth: set | None = None) -> list[dict]:
    c = cb.get_case(case_id)
    if not c or not c["masks"].get(model):
        raise SystemExit(f"нет кейса {case_id} с моделью {model}")
    # Без дотрассировки: сеть должна работать с тем, что дала сегментация.
    vol, base = cb._arrays_for(c, model, canals=False)
    sp = np.asarray(c["spacing"], float)
    zup = cb.occlusal_up(base)
    out = base.copy()
    rows = []
    for fdi in cb.TEETH_CLASSES:
        tooth_all, pulp_all = base == fdi, base == fdi + 100
        if pulp_all.sum() < MIN_PULP_VOX or not tooth_all.any():
            continue
        idx = np.argwhere(tooth_all | pulp_all)
        box = tuple(slice(a, b) for a, b in
                    zip(np.maximum(idx.min(0) - 4, 0),
                        np.minimum(idx.max(0) + 5, base.shape)))
        sub = base[box]
        tooth, pulp = sub == fdi, sub == fdi + 100
        body = cb.canal_body(tooth, pulp)
        sgn = 1 if (fdi // 10 in (1, 2)) == zup else -1

        want = png is not None and (not png_teeth or fdi in png_teeth)
        res = canal_in_tooth(net, device, vol[box], body, pulp, sp, sgn, thr,
                             want_grid=want)
        add_net, grid = res if want else (res, None)
        add_net &= (sub == fdi) | (sub == 0)      # чужие метки не трогаем
        import cbct_canals
        add_den = cbct_canals.complete_canals(vol[box], body, pulp, sp, sgn)
        add_den &= (sub == fdi) | (sub == 0)

        r0 = reach(tooth, pulp, sp, sgn)
        rn = reach(tooth, pulp | add_net, sp, sgn) if add_net.any() else r0
        rd = reach(tooth, pulp | add_den, sp, sgn) if add_den.any() else r0
        # Врач жаловался именно на ЧИСЛО каналов у многокорневых («где один,
        # где ни одного»), поэтому считаем его на тех же уровнях корня,
        # что и `cbct_canals.report_case`.
        cnt = {}
        if fdi % 10 >= 4:
            cnt = {k: cbct_canals._levels(body, m, sp, sgn)[1] for k, m in
                   (("base", pulp), ("net", pulp | add_net), ("den", pulp | add_den))}
        row = {"fdi": int(fdi), "reach_base": r0, "reach_net": rn,
               "reach_den": rd, "pulp_vox": int(pulp.sum()),
               "add_net": int(add_net.sum()), "add_den": int(add_den.sum()),
               "both": int((add_net & add_den).sum()), "canals": cnt}
        rows.append(row)
        if want:
            from scipy import ndimage as ndi
            g = grid["coords"]
            pulp_g = ndi.map_coordinates(pulp.astype(np.float32), g, order=1) > 0.5
            den_g = ndi.map_coordinates(add_den.astype(np.float32), g, order=1) > 0.5
            png.mkdir(parents=True, exist_ok=True)
            tooth_png(grid, pulp_g, den_g, fdi, row).save(
                png / f"{case_id}_{model}_{fdi}.png")
        if save and add_net.sum() >= MIN_ADD_VOX:
            out[box][add_net] = fdi + 100
    if save:
        import SimpleITK as sitk
        ref = sitk.ReadImage(c["masks"][model])
        img = sitk.GetImageFromArray(out.astype(np.uint8))
        img.CopyInformation(ref)
        p = Path(c["masks"][model]).with_name(f"mask_{model}_canalnet.mha")
        sitk.WriteImage(img, str(p), useCompression=True)
        print(f"маска с каналами сети: {p}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", default="sp")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--thr", type=float, default=THR)
    ap.add_argument("--save", action="store_true", help="записать маску рядом с кейсом")
    ap.add_argument("--png", help="папка для картинок выреза (на глаз)")
    ap.add_argument("--teeth", help="номера зубов для картинок через запятую")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, info = load_net(Path(a.weights), device)
    m = info["metrics"]
    print(f"веса: {a.weights} (эпоха {info['epoch']}, на проверке Dice "
          f"{m.get('dice', float('nan')):.3f}, ошибка дохода "
          f"{m.get('reach_err', float('nan')):.3f})")

    teeth = {int(t) for t in a.teeth.split(",")} if a.teeth else None
    rows = apply_case(a.case, a.model, net, device, a.thr, a.save,
                      Path(a.png) if a.png else None, teeth)
    print(f"\n{a.case}/{a.model}: зубов с пульпой {len(rows)}")
    print("зуб   доход: модель -> сеть / плотность    прирост вокселей: сеть / плотность / общих")
    for r in rows:
        print(f"{r['fdi']:3d}   {r['reach_base']:.2f} -> {r['reach_net']:.2f} / "
              f"{r['reach_den']:.2f}                {r['add_net']:6d} / "
              f"{r['add_den']:6d} / {r['both']:6d}")
    if rows:
        b = np.array([r["reach_base"] for r in rows])
        n = np.array([r["reach_net"] for r in rows])
        d = np.array([r["reach_den"] for r in rows])
        print(f"\nмедиана дохода: модель {np.median(b):.2f}, сеть {np.median(n):.2f}, "
              f"плотность {np.median(d):.2f}")
        print(f"доведено до 0.95 длины зуба: модель {(b >= 0.95).sum()}, "
              f"сеть {(n >= 0.95).sum()}, плотность {(d >= 0.95).sum()} из {len(rows)}")
        print(f"сеть дальше плотности у {(n > d + 0.02).sum()} зубов, "
              f"короче у {(d > n + 0.02).sum()}")

        multi = [r for r in rows if r["canals"]]
        if multi:
            print("\nчисло каналов в срезе на 55/66/77/88% длины корня "
                  "(премоляры и моляры):")
            print("зуб    модель          сеть            плотность")
            for r in multi:
                c = r["canals"]
                print(f"{r['fdi']:3d}    {str(c['base']):15s} {str(c['net']):15s} "
                      f"{str(c['den'])}")
            deep = lambda key: np.array([max(r["canals"][key][2:]) for r in multi])
            print(f"каналов в апикальной трети (максимум по 77/88%), медиана: "
                  f"модель {np.median(deep('base')):.0f}, сеть {np.median(deep('net')):.0f}, "
                  f"плотность {np.median(deep('den')):.0f}")


if __name__ == "__main__":
    main()
