"""Подготовка зубов к разметке каналов врачом: вырезы, черновики, очередь.

    .venv\\Scripts\\python.exe research\\scripts\\cbct_canal_prepare.py            # все клинические кейсы
    .venv\\Scripts\\python.exe research\\scripts\\cbct_canal_prepare.py --cases client5
    .venv\\Scripts\\python.exe research\\scripts\\cbct_canal_prepare.py --force    # пересобрать готовые

ЗАЧЕМ. Модель канала доводит его до верхушки на эталонных данных (ошибка
дохода 0.017), но на нашем Dentium прибавка частичная: медиана дохода 0.82 у
сегментации, 0.86 у модели при «полном» уровне около 0.91. Проверить, кто
прав, нечем: разметки каналов на нашем аппарате не существует. Её и собираем.

ЧТО ОТДАЁТСЯ ВРАЧУ — РОДНЫЕ СРЕЗЫ ТОМА, а не наш выпрямленный вырез. Врач
читает канал на аксиальном срезе как тёмную точку в кольце дентина, ровно как
в Rainbow у аппарата. Наш вырез для этого не годится: он пересчитан вдоль оси
зуба с шагом 0.16 мм из вокселей 0.31 мм, каждый пиксель — смесь восьми
соседей, и картинка расплывается («фарш» — прямые слова врача). Выпрямленная
сетка нужна МОДЕЛИ, и она остаётся внутри: точки врача переводятся в неё при
обучении, а не наоборот.

Плюс к срезам отдаются ЧЕРНОВЫЕ линии каналов от модели: врач не рисует с
нуля, а подтверждает или дотягивает. Иначе зуб стоит две-три минуты вместо
двадцати секунд, а нам нужны сотни зубов.

ОЧЕРЕДЬ ПО РАСХОЖДЕНИЮ. Зуб, где модель и трассировка по плотности согласны,
разметкой почти ничего не добавляет. Поэтому считаем оба способа и сортируем
очередь по расхождению — так те же часы врача дают в разы больше сигнала.
Но каждый пятый зуб идёт СЛУЧАЙНЫМ (`random_every`): набор только из трудных
случаев занизил бы измеренное качество, и честного числа мы бы не получили.

ТОЧКИ ХРАНЯТСЯ В КООРДИНАТАХ ТОМА. Вырез строится по маске зуба ТЕКУЩЕЙ
модели. Станет сегментация лучше — рамка сдвинется, и разметка, записанная в
сетке выреза, тихо уедет вместе с ней. Поэтому здесь же сохраняются параметры
выреза (центр, оси, шаг, смещение бокса), а сервер разметки переводит каждую
точку в воксели исходного тома. Разметка стоит дороже любой модели, и пережить
смену модели она обязана.

СТОРОНЫ. Кейс с аппарата, для которого конвенция сторон не подтверждена, в
разметку НЕ идёт: номера зубов на нём могут быть зеркальны, и врач разметит
канал «зуба 16», который на самом деле 26.

ДУБЛИ ОТСЕИВАЮТСЯ ПО СОДЕРЖИМОМУ ТОМА, а не по имени кейса. На первом же
прогоне выяснилось, что из восьми клинических КЛКТ разных пациентов всего
четыре: `client2` и `case_1788704679` — один том, а `client5` лежит в
ЧЕТЫРЁХ копиях (загружался в демо повторно). Без отсева врач размечал бы
один и тот же зуб по четыре раза, а при обучении один пациент попал бы и в
обучение, и в проверку. То же правило, что в подаче панорам, где дубль
поймался с первого прогона.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import cbct_backend as cb          # noqa: E402
import cbct_canal_crops as cc      # noqa: E402
import cbct_canal_apply as ca      # noqa: E402

OUT = ROOT / "research_3d/canal_annotation"
# Окно плотности считается ПО САМОМУ ЗУБУ, а не берётся общим. Общее окно
# 0..3400 (от карточки зуба) на client2 упёрлось в потолок у ЧЕТВЕРТИ вокселей
# зуба: и канал, и дентин вокруг него сохранялись как 255, разница между ними
# в файле становилась нулевой, и врач видел белое пятно. Замер по зубу 22:
# внутри зуба p50 = 2477, p75 = 3659, p99 = 6842 — то есть потолок 3400 срезал
# верхнюю четверть. Шкала КЛКТ не количественная, поэтому окно только своё.
HU_PCT = (0.5, 99.5)      # перцентили внутри тела зуба
HU_MIN_RANGE = 500.0      # уже — значит зуб почти однороден, растягивать нечего
SHEET_COLS = 16           # лист срезов 16x8 при GRID[0] = 128
MIN_PULP_VOX = 30
MIN_CANAL_VOX = 8         # компонента в срезе меньше — шум, не канал
RANDOM_EVERY = 5          # каждый пятый зуб в очереди — случайный
# Зуб короче этого — огрызок маски, а не зуб: врачу показывать нечего, а канал
# в нём обрезан вместе с корнем. На наших кейсах таких 15 из 109, самый мелкий
# 6.8 мм при медиане 19.5. Порог мягкий: третьи моляры законно короче прочих.
MIN_TOOTH_LEN_MM = 13.0


def _lines(mask: np.ndarray) -> list[list[list[int]]]:
    """Линии каналов из объёмной маски: по срезам центры компонент, связанные
    по ближайшему. Для черновика этого достаточно — врач всё равно правит."""
    chains: list[list[list[int]]] = []
    tails: list[tuple[int, np.ndarray]] = []      # (индекс цепочки, последняя точка)
    for lvl in range(mask.shape[0]):
        sl = mask[lvl]
        if not sl.any():
            continue
        lab, n = ndi.label(sl)
        for k in range(1, n + 1):
            pts = np.argwhere(lab == k)
            if len(pts) < MIN_CANAL_VOX:
                continue
            c = pts.mean(0)
            best, bd = -1, 1e9
            for i, (ci, tail) in enumerate(tails):
                d = float(np.hypot(*(tail - c)))
                if d < bd:
                    best, bd = i, d
            # 6 вокселей сетки это около миллиметра: дальше — другой канал
            if best >= 0 and bd < 6.0:
                ci = tails[best][0]
                chains[ci].append([lvl, int(round(c[0])), int(round(c[1]))])
                tails[best] = (ci, c)
            else:
                chains.append([[lvl, int(round(c[0])), int(round(c[1]))]])
                tails.append((len(chains) - 1, c))
    return [ch for ch in chains if len(ch) >= 3]


def hu_window(img: np.ndarray, body: np.ndarray) -> tuple[float, float]:
    """Окно плотности этого зуба: медиана дентина в середине шкалы.

    Правило выбрано СРАВНЕНИЕМ ТРЁХ на отрисовке (зуб 36 client2):
      p0.5-p99.5      дентин уходит на 0.22 шкалы — зуб тёмный, канала не видно;
      медиана +-1.8 IQR   дентин 0.50, канал тёмной точкой — как во вьюере аппарата;
      p2-p85          дентин 0.66 — зуб выжжен в белое.
    Крайние перцентили не годятся: одна плотная пломба (у зуба 22 плотность
    доходит до 8329 при дентине 2477) утягивает верхнюю границу, и весь зуб
    проваливается в темноту. Межквартильный размах к таким выбросам глух.
    """
    v = img[body]
    if v.size < 50:
        return (0.0, 3400.0)
    q25, med, q75 = (float(x) for x in np.percentile(v, (25, 50, 75)))
    half = max(1.8 * (q75 - q25), HU_MIN_RANGE / 2)
    return med - half, med + half


def _sheet_cols(vol3d: np.ndarray, lo: float, hi: float, cols: int) -> Image.Image:
    """Лист срезов: срез за срезом в `cols` столбцов."""
    n, h, w = vol3d.shape
    rows = int(np.ceil(n / cols))
    g = np.clip((vol3d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    sheet = np.zeros((rows * h, cols * w), np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = (g[i] * 255).astype(np.uint8)
    return Image.fromarray(sheet, mode="L")


def _sheet(vol3d: np.ndarray, lo: float, hi: float) -> Image.Image:
    return _sheet_cols(vol3d, lo, hi, SHEET_COLS)


AXIAL_PAD = 7             # вокселей вокруг зуба в родном срезе
AXIAL_COLS = 10


def axial_stack(vol: np.ndarray, tooth: np.ndarray, pulp: np.ndarray):
    """Родные аксиальные срезы вокруг зуба: тот же вид, что во вьюере аппарата.

    Ничего не пересчитываем — берём воксели тома как есть, иначе канал
    расплывается интерполяцией. Возвращаем (стек, начало бокса в томе).
    """
    idx = np.argwhere(tooth | pulp)
    lo = np.maximum(idx.min(0) - np.array([1, AXIAL_PAD, AXIAL_PAD]), 0)
    hi = np.minimum(idx.max(0) + np.array([2, AXIAL_PAD + 1, AXIAL_PAD + 1]), vol.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    return vol[box].astype(np.float32), [int(v) for v in lo]


def volume_hash(path: str) -> str:
    """Отпечаток тома по прореженным вокселям: полный том 300 МБ, а разные
    пациенты расходятся и на каждом четвёртом вокселе."""
    import hashlib
    a = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return hashlib.md5(np.ascontiguousarray(a[::4, ::4, ::4]).tobytes()).hexdigest()[:12]


def unique_cases(ids: list[str]) -> list[str]:
    """Оставить по одному кейсу на пациента; предпочитаем именованный."""
    groups: dict[str, list[str]] = {}
    for cid in ids:
        c = cb.get_case(cid)
        if not c or not c.get("volume"):
            continue
        groups.setdefault(volume_hash(c["volume"]), []).append(cid)
    out = []
    for h, same in sorted(groups.items()):
        # «client5» понятнее, чем «case_1788759007», поэтому именованный первым
        same.sort(key=lambda x: (x.startswith("case_"), x))
        out.append(same[0])
        if len(same) > 1:
            print(f"  дубль тома {h}: беру {same[0]}, пропускаю {', '.join(same[1:])}")
    return sorted(out)


def drop_teeth_of(case_id: str) -> int:
    """Убрать подготовленные зубы кейса (например, оказавшегося дублем)."""
    import shutil
    n = 0
    for d in (OUT / "teeth").glob(f"{case_id}_*"):
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    return n


def prepare_case(case_id: str, model: str, net, device, force: bool) -> list[dict]:
    c = cb.get_case(case_id)
    if not c or not c["masks"].get(model):
        print(f"  {case_id}: нет маски модели {model}")
        return []
    state = cb.scanner_sides_state(c) if hasattr(cb, "scanner_sides_state") else {}
    if state and not state.get("confirmed", True):
        print(f"  {case_id}: стороны аппарата НЕ подтверждены — пропускаю "
              f"(номера зубов могут быть зеркальны)")
        return []

    vol, base = cb._arrays_for(c, model, canals=False)
    sp = np.asarray(c["spacing"], float)
    zup = cb.occlusal_up(base)
    rows = []
    for fdi in cb.TEETH_CLASSES:
        tooth_all, pulp_all = base == fdi, base == fdi + 100
        if pulp_all.sum() < MIN_PULP_VOX or not tooth_all.any():
            continue
        d = OUT / "teeth" / f"{case_id}_{model}_{fdi}"
        if (d / "tooth.json").is_file() and not force:
            rows.append(json.loads((d / "tooth.json").read_text(encoding="utf-8")))
            continue
        idx = np.argwhere(tooth_all | pulp_all)
        lo_i = np.maximum(idx.min(0) - 4, 0)
        hi_i = np.minimum(idx.max(0) + 5, base.shape)
        box = tuple(slice(a, b) for a, b in zip(lo_i, hi_i))
        sub = base[box]
        tooth, pulp = sub == fdi, sub == fdi + 100
        body = cb.canal_body(tooth, pulp)
        sgn = 1 if (fdi // 10 in (1, 2)) == zup else -1

        coords, info = cc.grid_of(body, sp, sgn)
        if info["length_mm"] < MIN_TOOTH_LEN_MM:
            print(f"  {case_id} зуб {fdi}: огрызок маски ({info['length_mm']} мм) — пропускаю",
                  flush=True)
            continue
        img = ndi.map_coordinates(vol[box].astype(np.float32), coords, order=1,
                                  mode="constant", cval=float(vol[box].min()))
        bg = ndi.map_coordinates(body.astype(np.float32), coords, order=1) > 0.5
        pg = ndi.map_coordinates(pulp.astype(np.float32), coords, order=1) > 0.5

        # Черновики: модель канала и трассировка по плотности, обе в сетке.
        x = np.stack([cc.normalize(img, bg), bg.astype(np.float32)])
        import torch
        with torch.no_grad():
            prob = torch.sigmoid(net(torch.from_numpy(x)[None].to(device)))[0, 0]
        pred = (prob.cpu().numpy() > ca.THR) & bg
        import cbct_canals
        add_den = cbct_canals.complete_canals(vol[box], body, pulp, sp, sgn)
        den_g = ndi.map_coordinates(add_den.astype(np.float32), coords, order=1) > 0.5

        d.mkdir(parents=True, exist_ok=True)
        hu = hu_window(img, bg)
        _sheet(img, *hu).save(d / "slices.png")
        _sheet(bg.astype(np.uint8) * 255, 0, 255).save(d / "body.png")
        # родные срезы тома — то, что видит врач
        ax_vol, ax_off = axial_stack(vol, tooth_all, pulp_all)
        ax_tooth, _ = axial_stack((tooth_all | pulp_all).astype(np.float32),
                                  tooth_all, pulp_all)
        ax_hu = hu_window(ax_vol, ax_tooth > 0.5)
        _sheet_cols(ax_vol, *ax_hu, AXIAL_COLS).save(d / "axial.png")
        _sheet_cols(ax_tooth * 255, 0, 255, AXIAL_COLS).save(d / "axial_tooth.png")

        rec = {
            "case": case_id, "model": model, "fdi": int(fdi),
            "grid": list(cc.GRID), "sheet_cols": SHEET_COLS,
            "hu": [round(v, 1) for v in hu],
            "axial": {"shape": [int(v) for v in ax_vol.shape], "cols": AXIAL_COLS,
                      "offset": ax_off, "hu": [round(v, 1) for v in ax_hu],
                      "spacing": [float(v) for v in sp]},
            "spacing": [float(v) for v in sp],
            "box_offset": [int(v) for v in lo_i],
            "frame": info["frame"],
            "length_mm": info["length_mm"], "radius_mm": info["radius_mm"],
            "step_axial_mm": info["step_axial_mm"],
            "step_lateral_mm": info["step_lateral_mm"],
            "draft_net": _lines(pred & ~pg), "draft_den": _lines(den_g & ~pg),
            "pulp_lines": _lines(pg),
            "reach_base": round(ca.reach(tooth, pulp, sp, sgn), 3),
            "reach_net": round(ca.reach(tooth, pulp | (
                _to_vol(pred, coords, body.shape) & ~pulp), sp, sgn), 3),
            "reach_den": round(ca.reach(tooth, pulp | add_den, sp, sgn), 3),
        }
        rec["disagree"] = round(abs(rec["reach_net"] - rec["reach_den"]), 3)
        (d / "tooth.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        rows.append(rec)
        print(f"  {case_id} зуб {fdi}: доход {rec['reach_base']:.2f} -> "
              f"сеть {rec['reach_net']:.2f} / плотность {rec['reach_den']:.2f}", flush=True)
    return rows


def _to_vol(grid_mask: np.ndarray, coords: np.ndarray, shape) -> np.ndarray:
    out = np.zeros(shape, bool)
    if not grid_mask.any():
        return out
    idx = np.rint(coords[:, grid_mask]).astype(int)
    for k in range(3):
        np.clip(idx[k], 0, shape[k] - 1, out=idx[k])
    out[idx[0], idx[1], idx[2]] = True
    return out


def build_queue(rows: list[dict]) -> list[dict]:
    """Очередь: сначала спорные зубы, но каждый пятый — случайный.

    Только из трудных случаев набор дал бы заниженное качество и нечестную
    метрику, поэтому случайные вкраплены с самого начала, а не «потом».
    Выпасть из очереди не должен НИ ОДИН зуб: случайный не заменяет спорный,
    а вклинивается перед ним (первая версия молча теряла каждый пятый).
    """
    rng = np.random.default_rng(0)
    hard = sorted(range(len(rows)), key=lambda i: -rows[i]["disagree"])
    rand = [int(i) for i in rng.permutation(len(rows))]
    used: set[int] = set()
    hi = ri = 0
    order: list[dict] = []

    def take(pool: list[int], pos: int):
        while pos < len(pool) and pool[pos] in used:
            pos += 1
        return (pool[pos], pos + 1) if pos < len(pool) else (None, pos)

    while len(order) < len(rows):
        if (len(order) + 1) % RANDOM_EVERY == 0:
            i, ri = take(rand, ri)
            if i is None:
                i, hi = take(hard, hi)
        else:
            i, hi = take(hard, hi)
            if i is None:
                i, ri = take(rand, ri)
        if i is None:
            break
        used.add(i)
        r = rows[i]
        order.append({"id": f"{r['case']}_{r['model']}_{r['fdi']}", "case": r["case"],
                      "fdi": r["fdi"], "disagree": r["disagree"],
                      "reach_base": r["reach_base"], "reach_net": r["reach_net"],
                      "reach_den": r["reach_den"]})
    return order


def redraw_images(model: str) -> list[dict]:
    """Перерисовать листы срезов у готовых зубов (черновики не трогаем)."""
    rows = rows_on_disk()
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    for cid, items in sorted(by_case.items()):
        c = cb.get_case(cid)
        vol, base = cb._arrays_for(c, model, canals=False)
        sp = np.asarray(c["spacing"], float)
        zup = cb.occlusal_up(base)
        for r in items:
            fdi = r["fdi"]
            tooth, pulp = base == fdi, base == fdi + 100
            if not tooth.any():
                continue
            idx = np.argwhere(tooth | pulp)
            lo_i = np.maximum(idx.min(0) - 4, 0)
            box = tuple(slice(a, b) for a, b in
                        zip(lo_i, np.minimum(idx.max(0) + 5, base.shape)))
            sub = base[box]
            body = cb.canal_body(sub == fdi, sub == fdi + 100)
            sgn = 1 if (fdi // 10 in (1, 2)) == zup else -1
            coords, info = cc.grid_of(body, sp, sgn)
            img = ndi.map_coordinates(vol[box].astype(np.float32), coords, order=1,
                                      mode="constant", cval=float(vol[box].min()))
            bg = ndi.map_coordinates(body.astype(np.float32), coords, order=1) > 0.5
            d = OUT / "teeth" / f"{cid}_{model}_{fdi}"
            hu = hu_window(img, bg)
            _sheet(img, *hu).save(d / "slices.png")
            _sheet(bg.astype(np.uint8) * 255, 0, 255).save(d / "body.png")
            ax_vol, ax_off = axial_stack(vol, tooth, pulp)
            ax_tooth, _ = axial_stack((tooth | pulp).astype(np.float32), tooth, pulp)
            ax_hu = hu_window(ax_vol, ax_tooth > 0.5)
            _sheet_cols(ax_vol, *ax_hu, AXIAL_COLS).save(d / "axial.png")
            _sheet_cols(ax_tooth * 255, 0, 255, AXIAL_COLS).save(d / "axial_tooth.png")
            r["hu"] = [round(v, 1) for v in hu]
            r["axial"] = {"shape": [int(v) for v in ax_vol.shape], "cols": AXIAL_COLS,
                          "offset": ax_off, "hu": [round(v, 1) for v in ax_hu],
                          "spacing": [float(v) for v in sp]}
            (d / "tooth.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"  {cid}: перерисовано {len(items)}", flush=True)
    return rows


def rows_on_disk() -> list[dict]:
    """Все подготовленные зубы — чтобы пересобрать очередь без пересчёта."""
    out = []
    for p in sorted((OUT / "teeth").glob("*/tooth.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"  повреждён {p}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", help="кейсы; по умолчанию все клинические")
    ap.add_argument("--model", default="sp")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--queue-only", action="store_true",
                    help="пересобрать очередь по готовым зубам, ничего не считая")
    ap.add_argument("--images-only", action="store_true",
                    help="перерисовать картинки зубов, не пересчитывая черновики")
    a = ap.parse_args()

    if a.images_only:
        rows = redraw_images(a.model)
    elif a.queue_only:
        rows = rows_on_disk()
    else:
        ids = a.cases or [cid for cid, c in sorted(cb.cases.items())
                          if "Dentium" in ((c.get("scanner") or {}).get("manufacturer") or "")]
        keep = unique_cases(ids)
        for cid in ids:
            if cid not in keep and drop_teeth_of(cid):
                print(f"  убраны зубы дубля {cid}")
        ids = keep
        net, device = cb.canal_net()
        if net is None:
            raise SystemExit("нет весов модели канала — сначала train_canal_net.py")
        print(f"кейсов: {len(ids)}")
        rows = []
        for cid in ids:
            print(f"-> {cid}")
            rows += prepare_case(cid, a.model, net, device, a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    # Огрызки, подготовленные прошлыми версиями, из очереди убираем вместе с
    # файлами: врач на них только потратит время.
    import shutil
    keep = []
    for r in rows:
        if r["length_mm"] < MIN_TOOTH_LEN_MM:
            shutil.rmtree(OUT / "teeth" / f"{r['case']}_{r['model']}_{r['fdi']}",
                          ignore_errors=True)
            print(f"  убран огрызок {r['case']} зуб {r['fdi']} ({r['length_mm']} мм)")
        else:
            keep.append(r)
    rows = keep
    queue = build_queue(rows)
    (OUT / "queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(f"\nзубов готово: {len(rows)}, очередь: {OUT / 'queue.json'}")
    if rows:
        d = np.array([r["disagree"] for r in rows])
        print(f"расхождение сети и плотности: медиана {np.median(d):.3f}, "
              f"больше 0.05 у {(d > 0.05).sum()} зубов")


if __name__ == "__main__":
    main()
