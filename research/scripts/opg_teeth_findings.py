"""Кариес на панораме ДВУМЯ схемами сразу, с уровнем достоверности.

    .venv\\Scripts\\python.exe research\\scripts\\opg_teeth_findings.py снимок.jpg

ЗАЧЕМ ДВЕ СХЕМЫ. У нас есть одноэтапный детектор патологий (ищет очаг прямо
на снимке) и каскад (находит зуб, потом смотрит на его вырез). Замер на
официальной валидации DENTEX, 1395 зубов, 147 из них с кариесом:

    режим                 точность  полнота     F1   ложных  пропущено
    только детектор         0.567    0.633    0.598     71       54
    только каскад           0.427    0.694    0.528    137       45
    обе согласны            0.658    0.524    0.583     40       70
    любая из двух           0.413    0.803    0.545    168       29

Ни один режим не лучший по всем показателям — и это не беда, а то, чем можно
распорядиться, потому что ОШИБКИ У НИХ РАЗНЫЕ:

    верно нашли обе   77
    только детектор   16
    только каскад     25

Каскад видит 25 зубов, которые детектор пропускает, и наоборот. Значит
объединять есть смысл — а выбирать один режим значило бы выбросить эту
разницу.

ПОЭТОМУ ТРИ УРОВНЯ, А НЕ ОДНО ЧИСЛО. Врач сначала смотрит уверенные находки,
потом спорные:

    confident   обе схемы согласны       ложных 40 вместо 71
    possible    сказала только одна      здесь лежит половина настоящих находок
    (нет)       не сказал никто

Так пропущенных 29 вместо 54 у нынешней схемы, и при этом чистый список не
тонет в ложных тревогах: они уходят в «под вопросом», а не подаются наравне.

СПИСОК «ПОД ВОПРОСОМ» С ПОРОГАМИ (17.09). Врачи жаловались на лишнее.
Весь шум был там: 128 ложных против 41 верного. Список «обе согласны» не
трогаем, а в «под вопросом» одна схема попадает, только если уверена —
детектор от 0.40, классификатор выреза с вероятностью кариеса от 0.70:

    вопрос: детектор / классификатор   верных  ложных  пропущено всего
    0.25 / наиболее вероятный            41     128         29
    0.25 / 0.70                          30      95         40
    0.40 / 0.70  (выбрано)               23      82         47
    0.40 / 0.80                          21      62         49

Пропусков стало больше (47), но всё ещё меньше, чем у одного детектора (54),
а ложных в спорном списке на треть меньше. Поднимать сам детектор до 0.40
для «обе согласны» невыгодно: у него полнота падает с 0.633 до 0.483 —
этот детектор откалиброван иначе, чем ансамбль.

ЧЕГО ЗДЕСЬ НЕТ. Ни глубины кариеса как диагноза, ни «кариеса нет». Полнота
0.803 означает, что каждый пятый кариес мы не видим — отрицательное
заключение по панораме недопустимо (до четверти проксимальных поражений не
проецируется вовсе, это предел модальности, а не модели).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DETECTOR = REPO / "research/output/runs/yolo12x_aug960/weights/best.pt"
CLASSIFIER = REPO / "research/output/tooth_cls/best.pt"
# Модель на 14 состояний (Zenodo 15487430, CC BY 4.0). Даёт то, чего у нас не
# было вовсе: эндодонтическое лечение, обтурацию, фрагменты корня.
STATES = REPO / "research/output/runs/panoramic14/weights/best.pt"

# Берём из неё ТОЛЬКО те классы, где она сильна. Замер на её валидации
# (153 снимка, mAP50):
#
#   END эндо-лечение    0.804   берём — лучший класс набора
#   IMP имплант         0.692   берём
#   IMT ретенция        0.659   берём
#   ROT фрагмент корня  0.637   берём
#   PRR протез          0.526   берём
#   OBT пломба          0.474   берём — это ПЛОМБА НА КОРОНКЕ, не канал
#
# ВНИМАНИЕ К НАЗВАНИЮ. Набор румынский, и `OBT` там — «obturație», то есть
# обычная пломба (реставрация коронки), а не пломбировка канала. Сначала я
# подписал её врачу как «пломбировка канала» — неверно. Проверено геометрией
# рамок на обучающей выборке и глазами на снимке:
#
#     класс   высота/ширина   высота рамки   что это
#     OBT         1.73           0.055       коронка, размер как у кариеса
#     END         4.03           0.103       вдоль корня — пломбировка канала
#     CAR         1.92           0.045       кариес, для сравнения
#
# Канал запломбирован — это `END`.
#   CAR кариес          0.430   НЕ берём — наш ансамбль даёт 0.665
#   BON резорбция кости 0.079   НЕ берём — см. ниже
#   APS, ROR            0.04    НЕ берём — 4 и 5 меток, статистики нет
#
# Почему резорбция кости провалилась: убыль — это не объект с границами, а
# УРОВЕНЬ вдоль всего края челюсти. Рамкой такое не ловится: полнота 0.017
# при 3178 обучающих метках. Пародонт придёт переносом измерения по ключевым
# точкам с КЛКТ, а не детекцией.
STATE_NAMES = ["IMP", "PRR", "OBT", "END", "CAR", "BON", "IMT",
               "API", "ROT", "FUR", "APS", "ROR", "ORD", "SRD"]
STATE_TAKE = {
    "END": ("endo_treated", "канал запломбирован (эндо-лечение)"),
    "OBT": ("restoration", "пломба (реставрация)"),
    "ROT": ("root_fragment", "фрагмент корня"),
    "IMT": ("impacted", "ретенция"),
    "IMP": ("implant", "имплант"),
    "PRR": ("prosthesis", "протез или коронка"),
}

# Классы одноэтапного детектора: 0 ретенция, 1 кариес, 2 очаг, 3 глубокий.
CARIES_CLASSES = (1, 3)
DEEP_CLASS = 3
APICAL_CLASS = 2

# Перекрытие, при котором детекция патологии считается относящейся к зубу.
# 0.3, а не 0.5: рамка патологии у детектора смещена относительно зуба, и
# строгий порог терял бы верные совпадения.
IOU_HIT = 0.3

# Одна схема без поддержки второй попадает в «под вопросом» только от этих
# порогов (замер — в докстринге). Версия правил пишется в ответ: кэш зубов
# у исследований (`demo_server.teeth2d_for`) по ней понимает, что устарел.
POSSIBLE_DET_CONF = 0.40
POSSIBLE_CASC_P = 0.70
RULES_VERSION = 2

_cache: dict = {}


def _models():
    if "yolo" not in _cache:
        import torch
        from ultralytics import YOLO
        import teeth_detect as td
        from train_tooth_cls import Net
        _cache["yolo"] = YOLO(str(DETECTOR))
        _cache["states"] = YOLO(str(STATES)) if STATES.is_file() else None
        _cache["teeth"] = td._load()
        ck = torch.load(CLASSIFIER, map_location="cpu", weights_only=False)
        net = Net().eval()
        net.load_state_dict(ck["model"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _cache["cls"] = net.to(dev)
        _cache["dev"] = dev
    return _cache


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / (aw * ah + bw * bh - i)


def analyze(image_path, conf: float = 0.25) -> dict:
    """Находки по панораме на уровне зуба."""
    import torch
    from PIL import Image
    import teeth_detect as td
    from train_tooth_cls import IN_W, IN_H

    m = _models()
    teeth, info = td.resolve(td.detect(str(image_path), model=m["teeth"]))

    r = m["yolo"].predict(str(image_path), imgsz=960, conf=conf,
                          verbose=False, device=0)[0]
    boxes = r.boxes.xywh.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()
    det = {"caries": [], "apical": []}
    for c, (x, y, w, h), sc in zip(clss, boxes, scores):
        box = (x - w / 2, y - h / 2, w, h)
        if c in CARIES_CLASSES:
            det["caries"].append((box, c == DEEP_CLASS, float(sc)))
        elif c == APICAL_CLASS:
            det["apical"].append(box)

    # ---- третий источник: состояния зуба (эндо, обтурация, фрагменты) ----
    states: dict[str, list] = {}
    if m.get("states") is not None:
        rs = m["states"].predict(str(image_path), imgsz=960, conf=conf,
                                 verbose=False, device=0)[0]
        for c, (x, y, w, h) in zip(rs.boxes.cls.cpu().numpy().astype(int),
                                   rs.boxes.xywh.cpu().numpy()):
            nm = STATE_NAMES[c] if c < len(STATE_NAMES) else ""
            if nm in STATE_TAKE:
                states.setdefault(nm, []).append((x - w / 2, y - h / 2, w, h))

    im = Image.open(image_path).convert("L")
    out = []
    for t in teeth:
        cx, cy, w, h = t["xywh"]
        box = (cx - w / 2, cy - h / 2, w, h)
        dx, dy = w * 0.25, h * 0.25
        crop = im.crop((int(max(cx - w / 2 - dx, 0)), int(max(cy - h / 2 - dy, 0)),
                        int(min(cx + w / 2 + dx, im.width)),
                        int(min(cy + h / 2 + dy, im.height))))
        a = np.asarray(crop.resize((IN_W, IN_H), Image.BILINEAR),
                       np.float32) / 255.0
        x = torch.from_numpy(a)[None].repeat(3, 1, 1)[None].to(m["dev"])
        with torch.no_grad():
            oc, oa = m["cls"](x)
            pc = torch.softmax(oc, 1)[0].cpu().numpy()
            pa = torch.softmax(oa, 1)[0].cpu().numpy()

        casc_caries = int(pc.argmax()) > 0
        casc_deep = int(pc.argmax()) == 2
        casc_p = float(1 - pc[0])
        det_hits = [(deep, sc) for bb, deep, sc in det["caries"]
                    if _iou(box, bb) >= IOU_HIT]
        det_caries = bool(det_hits)
        det_deep = any(d for d, _ in det_hits)
        det_score = max((sc for _, sc in det_hits), default=0.0)

        if det_caries and casc_caries:
            level = "confident"
        elif det_score >= POSSIBLE_DET_CONF or casc_p >= POSSIBLE_CASC_P:
            level = "possible"
        else:
            level = None

        # Состояния привязываем к зубу тем же перекрытием, что и патологии.
        tooth_states = [STATE_TAKE[nm][0] for nm, boxes in states.items()
                        if any(_iou(box, bb) >= IOU_HIT for bb in boxes)]

        # Периапикальный очаг — ОТДЕЛЬНО от кариеса и тем же правилом двух
        # схем. Раньше очаг попадал в ответ только вместе с кариесом, и очаг
        # на пролеченном или внешне целом зубе терялся — а это ровно тот
        # случай, где он клинически важнее всего (ре-эндо).
        #
        # Уровней два, как у кариеса: голова очага у классификатора выреза
        # слабая (F1 0.31 на 19 примерах проверки), и выдавать её сигнал
        # наравне с ансамблем значило бы подмешать ложные тревоги.
        det_apical = any(_iou(box, bb) >= IOU_HIT for bb in det["apical"])
        casc_apical = int(pa.argmax()) == 1
        if det_apical and casc_apical:
            apical_level = "confident"
        elif det_apical or casc_apical:
            apical_level = "possible"
        else:
            apical_level = None

        common = {
            "tooth": t["fdi"],
            "apical": apical_level is not None,
            "apical_level": apical_level,
            "apical_by_detector": det_apical,
            "states": tooth_states,
        }
        if level:
            out.append({
                **common,
                "code": "caries",
                "level": level,
                "deep": bool(det_deep or casc_deep),
                # Что именно сказала каждая схема — врач должен видеть, на
                # чём основано «под вопросом», а не только итог.
                # В «под вопросом» схема считается сказавшей, только если
                # прошла свой порог, — иначе подпись врачу соврала бы.
                "by_detector": (det_caries if level == "confident"
                                else det_score >= POSSIBLE_DET_CONF),
                "by_cascade": (casc_caries if level == "confident"
                               else casc_p >= POSSIBLE_CASC_P),
                "detector_score": round(det_score, 3),
                "cascade_score": round(casc_p, 3),
            })
        elif tooth_states or apical_level:
            # Зуб без кариеса, но с состоянием (пролечен, протез, фрагмент)
            # или с очагом — тоже находка: врачу важно видеть и то, что уже
            # сделано, и очаг у зуба, где кариеса нет.
            out.append({**common, "code": "state", "level": "state"})

    # Все найденные зубы с рамками — для зубной карты и подписей номеров на
    # снимке. Рамка в координатах ИСХОДНОГО снимка (x1, y1, x2, y2); масштаб
    # под показ применяет вызывающий.
    teeth_out = [{"fdi": t["fdi"], "conf": round(t["conf"], 3),
                  "box": [round(t["xywh"][0] - t["xywh"][2] / 2, 1),
                          round(t["xywh"][1] - t["xywh"][3] / 2, 1),
                          round(t["xywh"][0] + t["xywh"][2] / 2, 1),
                          round(t["xywh"][1] + t["xywh"][3] / 2, 1)]}
                 for t in teeth]
    return {"teeth_found": len(teeth), "teeth": teeth_out, "findings": out,
            "detector_notes": info, "width": im.width, "height": im.height,
            "rules": RULES_VERSION}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    res = analyze(args.image, conf=args.conf)
    print(f"зубов найдено: {res['teeth_found']}")
    conf_list = [f for f in res["findings"] if f["level"] == "confident"]
    poss = [f for f in res["findings"] if f["level"] == "possible"]

    print(f"\nкариес, обе схемы согласны ({len(conf_list)}):")
    for f in conf_list:
        print(f"  зуб {f['tooth']}" + ("  (глубокий)" if f["deep"] else ""))
    print(f"\nпод вопросом, сказала одна схема ({len(poss)}):")
    for f in poss:
        who = "детектор" if f["by_detector"] else "каскад"
        print(f"  зуб {f['tooth']}  ({who}, оценка {f['cascade_score']:.2f})")
    ap = [f for f in res["findings"] if f.get("apical_level")]
    if ap:
        print("")
        print("периапикальный очаг (" + str(len(ap)) + "):")
        for f in ap:
            lvl = "обе схемы" if f["apical_level"] == "confident" else "под вопросом"
            print("  зуб " + str(f["tooth"]) + "  (" + lvl + ")")
    st = [f for f in res["findings"] if f.get("states")]
    if st:
        names = {v[0]: v[1] for v in STATE_TAKE.values()}
        print("")
        print("состояния зубов (" + str(len(st)) + "):")
        for f in st:
            print("  зуб " + str(f["tooth"]) + ": "
                  + ", ".join(names.get(x, x) for x in f["states"]))
    print("\nОтрицательного заключения по панораме не даём: каждый пятый "
          "кариес не виден на этом типе снимка.")


if __name__ == "__main__":
    main()
