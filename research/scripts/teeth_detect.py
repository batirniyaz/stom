"""Детекция и нумерация зубов на панораме — первый этап двухэтапной схемы.

    .venv\\Scripts\\python.exe research\\scripts\\teeth_detect.py снимок.jpg

Сеть находит зубы и называет каждый номером FDI, а этот модуль приводит её
ответ в вид, которому можно верить: у одного зуба не бывает двух номеров, а
номера внутри квадранта идут по порядку от средней линии.

ЗАЧЕМ ПОСТ-ОБРАБОТКА. Замер на проверочной выборке (95 снимков, 2819
предсказаний против 2722 в разметке):

    дублей номера                       61 (0.64 на снимок, в 37 снимках)
    квадрантов с нарушенным порядком    57
    разрыв уверенности у дублей         медиана 0.20; >0.15 у 62%

То есть ошибки не случайны и лечатся правилами, а не переобучением. Разрыв
уверенности показывает, что в большинстве дублей одна рамка заметно лучше
другой — этого достаточно, чтобы выбрать.

ЧЕГО ЗДЕСЬ НАМЕРЕННО НЕТ: восстановления пропущенных зубов «по порядку».
Соблазн есть — увидев 24 и 26, дописать между ними 25. Но зуб может быть
удалён, и тогда мы придумаем врачу зуб, которого нет. Пропуск остаётся
пропуском; отсутствие зуба — это находка, а не ошибка детектора.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "research/output/runs/teeth_fdi_960-2/weights/best.pt"

FDI = [q * 10 + i for q in (1, 2, 3, 4) for i in range(1, 9)]

# Порог уверенности. Ниже 0.30 идут в основном ложные рамки на фоне, выше
# 0.45 начинают теряться восьмёрки — они мельче и хуже размечены (341 пример
# против 900 у центральных резцов).
CONF = 0.30
IMGSZ = 960

# Насколько рамка должна быть увереннее соперника, чтобы дубль решался
# уверенностью, а не геометрией. Медиана разрыва у дублей 0.20, поэтому 0.15
# отделяет «явно лучшую» рамку от спорной пары.
DUP_CONF_GAP = 0.15


def _load(weights: Path | str | None = None):
    from ultralytics import YOLO
    w = Path(weights or WEIGHTS)
    if not w.is_file():
        raise SystemExit(f"нет весов: {w}")
    return YOLO(str(w))


def detect(image, model=None, conf: float = CONF, imgsz: int = IMGSZ,
           device: int | str = 0) -> list[dict]:
    """Сырые детекции: [{fdi, conf, xywh}] без правил."""
    model = model or _load()
    r = model.predict(image, imgsz=imgsz, conf=conf, verbose=False,
                      device=device)[0]
    cls = r.boxes.cls.cpu().numpy().astype(int)
    cnf = r.boxes.conf.cpu().numpy()
    xywh = r.boxes.xywh.cpu().numpy()
    return [{"fdi": FDI[c], "conf": float(k), "xywh": box.tolist()}
            for c, k, box in zip(cls, cnf, xywh)]


def _midline(dets: list[dict]) -> float:
    """Средняя линия — медиана центров по горизонтали.

    Медиана, а не среднее: у пациента с удалёнными зубами на одной стороне
    среднее уезжает в сторону сохранившихся, и весь разбор по квадрантам
    ломается. Тот же выбор, что в `orient_sides` на КЛКТ, где центр берётся
    по структуре, а не по счёту объектов.
    """
    if not dets:
        return 0.0
    return float(np.median([d["xywh"][0] for d in dets]))


def resolve(dets: list[dict]) -> tuple[list[dict], dict]:
    """Убрать дубли номеров и пометить нарушения порядка.

    Возвращает очищенный список и сводку: что именно правилось. Сводка нужна
    не для отладки, а для врача — по ней видно, где модель сомневалась.
    """
    out = {d["fdi"]: d for d in []}          # fdi -> лучшая рамка
    dropped, ambiguous = [], []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        cur = out.get(d["fdi"])
        if cur is None:
            out[d["fdi"]] = d
            continue
        # Дубль: рамка с меньшей уверенностью отбрасывается. Если разрыв мал,
        # выбор считаем спорным и говорим об этом, а не молчим.
        if cur["conf"] - d["conf"] < DUP_CONF_GAP:
            ambiguous.append(d["fdi"])
        dropped.append({"fdi": d["fdi"], "conf": d["conf"]})

    kept = list(out.values())
    mid = _midline(kept)
    disorder = []
    for q in (1, 2, 3, 4):
        sel = [d for d in kept if d["fdi"] // 10 == q]
        if len(sel) < 3:
            continue
        sel.sort(key=lambda d: abs(d["xywh"][0] - mid))
        nums = [d["fdi"] % 10 for d in sel]
        if any(b <= a for a, b in zip(nums, nums[1:])):
            disorder.append(q)

    kept.sort(key=lambda d: d["fdi"])
    return kept, {"dropped": dropped, "ambiguous": sorted(set(ambiguous)),
                  "disorder_quadrants": disorder}


def crop(image, det: dict, pad: float = 0.25):
    """Вырез зуба для второго этапа.

    Запас по краям обязателен: рамки у нас не идеальны (mAP50-95 = 0.56), а
    кариес и убыль кости живут ровно на границе зуба — по контактной
    поверхности и у шейки. Урезанный кроп отрезал бы то, ради чего он берётся.
    """
    import numpy as np
    from PIL import Image
    im = Image.open(image) if isinstance(image, (str, Path)) else image
    cx, cy, w, h = det["xywh"]
    dx, dy = w * pad, h * pad
    box = (max(cx - w / 2 - dx, 0), max(cy - h / 2 - dy, 0),
           min(cx + w / 2 + dx, im.width), min(cy + h / 2 + dy, im.height))
    return im.crop(tuple(int(v) for v in box))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--raw", action="store_true", help="без пост-обработки")
    args = ap.parse_args()

    dets = detect(args.image, model=_load(args.weights), conf=args.conf)
    if args.raw:
        kept, info = dets, {}
    else:
        kept, info = resolve(dets)

    print(f"зубов: {len(kept)}")
    print("номера:", [d["fdi"] for d in kept])
    if info.get("dropped"):
        print(f"снято дублей: {len(info['dropped'])} "
              f"({', '.join(str(x['fdi']) for x in info['dropped'])})")
    if info.get("ambiguous"):
        print("спорный выбор (разрыв уверенности мал):", info["ambiguous"])
    if info.get("disorder_quadrants"):
        print("порядок номеров нарушен в квадрантах:",
              info["disorder_quadrants"])


if __name__ == "__main__":
    main()
