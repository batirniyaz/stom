"""Карта окклюзионных контактов по интраоральному скану: где зубы смыкаются.

Новых моделей здесь нет — как и в находках по КЛКТ, это геометрия поверх уже
готовой сегментации. Для каждой вершины зуба считается расстояние до
поверхности зубов ПРОТИВОПОЛОЖНОЙ челюсти; из него получаются и заливка во
вьюере, и таблица «зуб — площадь контакта — минимальный зазор».

Обязательное условие: обе челюсти в одной системе координат со смыканием.
Exocad так и отдаёт (см. ios_build_mesh), Teeth3DS — нет, там каждый скан
отцентрован отдельно и прикус уничтожен.

Расстояние берётся до плотной выборки точек противоположной дуги, а не точным
поиском по треугольникам: точный расчёт на 70 тысяч вершин в trimesh занимает
минуты, а выборка в 600 тысяч точек даёт шаг около 0.07 мм. Значения из-за
этого чуть завышены (примерно на половину шага), что для полос шириной 0.1 мм
несущественно, но помнить об этом надо.

ВАЖНО: пороги ниже — ЧЕРНОВИК. Их подписывает врач. До подписи вывод помечен
draft и назначением не является.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_occlusion_map.py batir
    .venv/Scripts/python.exe research/scripts/ios_occlusion_map.py batir --json out.json
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clinic_routes as cr  # noqa: E402

CASES = Path(__file__).resolve().parents[2] / "research_ios" / "web_cases"

# --- пороги (ЧЕРНОВИК, на подпись врачу) ---
TIGHT_MM = 0.10      # плотный контакт
CONTACT_MM = 0.30    # контакт
NEAR_MM = 0.60       # близко, но не смыкается; дальше — вне окклюзии
MAX_MM = 2.55        # верх шкалы: в байт кладём сотые доли миллиметра

N_SAMPLE = 600_000   # точек на противоположной дуге (см. докстринг)
MIN_SPOT_MM2 = 0.10  # пятно контакта мельче — шум сегментации, не точка контакта
MIN_TOOTH_FACES = 120

# Роли общие для всех источников находок (см. clinic_routes).
ROUTE = cr.ROUTES


def antagonist(tooth: int) -> int:
    """Зуб-антагонист по FDI: верхний 1X смыкается с нижним 4X, 2X — с 3X.

    Это упрощение: в норме зуб контактирует с двумя антагонистами со сдвигом
    на ползуба. Но для вопроса «есть ли вообще чему смыкаться» одноимённого
    зуба достаточно, а именно на этот вопрос нужно ответить, прежде чем
    называть отсутствие контакта нарушением прикуса.
    """
    return {1: 4, 4: 1, 2: 3, 3: 2}[tooth // 10] * 10 + tooth % 10


def load_teeth(case_dir: Path) -> dict:
    scene = trimesh.load(str(case_dir / "scan.glb"), process=False)
    teeth = {}
    for name, geom in scene.geometry.items():
        short = name.split("/")[-1]
        if short.startswith("tooth_") and len(geom.faces) >= MIN_TOOTH_FACES:
            teeth[int(short.split("_")[1])] = geom
    return teeth


def contact_spots(mesh, face_dist: np.ndarray) -> tuple[float, int]:
    """Площадь контакта в мм² и число ОТДЕЛЬНЫХ пятен контакта.

    Число пятен диагностично само по себе: у здорового моляра контактов
    несколько (по бугоркам), а единственное пятно на весь зуб — признак того,
    что зуб принимает нагрузку одной точкой.
    """
    sel = face_dist <= CONTACT_MM
    if not sel.any():
        return 0.0, 0
    areas = mesh.area_faces
    total = float(areas[sel].sum())
    sub = mesh.submesh([np.flatnonzero(sel)], append=True, repair=False)
    if sub is None or len(sub.faces) == 0:
        return total, 1
    labels = trimesh.graph.connected_component_labels(
        sub.face_adjacency, node_count=len(sub.faces))
    spots = 0
    for lab in range(labels.max() + 1 if len(labels) else 0):
        if float(sub.area_faces[labels == lab].sum()) >= MIN_SPOT_MM2:
            spots += 1
    return total, spots


def analyze(case_id: str) -> dict:
    # Выборка точек по поверхности случайна, и без сида зазоры плавали в
    # сотых миллиметра между запусками. Карта контактов идёт врачу — она не
    # имеет права меняться от повторного счёта.
    np.random.seed(0)
    case_dir = CASES / case_id
    if not (case_dir / "scan.glb").is_file():
        raise SystemExit(f"нет {case_dir / 'scan.glb'}")
    teeth = load_teeth(case_dir)
    upper = {t: m for t, m in teeth.items() if t // 10 in (1, 2)}
    lower = {t: m for t, m in teeth.items() if t // 10 in (3, 4)}
    if not upper or not lower:
        raise SystemExit("в кейсе только одна челюсть — прикус не посчитать")

    trees = {}
    for key, arch in (("upper", lower), ("lower", upper)):
        merged = trimesh.util.concatenate(list(arch.values()))
        pts, _ = trimesh.sample.sample_surface(merged, N_SAMPLE)
        trees[key] = cKDTree(pts)

    per_tooth, vertex_bytes = [], {}
    for t, mesh in sorted(teeth.items()):
        tree = trees["upper" if t // 10 in (1, 2) else "lower"]
        dist, _ = tree.query(np.asarray(mesh.vertices), workers=-1)
        # в байт — сотые доли миллиметра; вьюеру больше 2.55 мм и не нужно
        vertex_bytes[f"tooth_{t}"] = base64.b64encode(
            np.clip(dist * 100, 0, 255).astype(np.uint8).tobytes()).decode()
        face_dist = dist[mesh.faces].mean(axis=1)
        area, spots = contact_spots(mesh, face_dist)
        per_tooth.append({
            "fdi": t,
            "min_gap_mm": round(float(dist.min()), 2),
            "contact_area_mm2": round(area, 1),
            "contact_spots": spots,
            "surface_mm2": round(float(mesh.area), 1),
            "has_antagonist": antagonist(t) in teeth,
        })

    findings = build_findings(per_tooth)
    return {
        "case_id": case_id,
        "draft": True,
        "thresholds_mm": {"tight": TIGHT_MM, "contact": CONTACT_MM,
                          "near": NEAR_MM, "scale_max": MAX_MM},
        "teeth": per_tooth,
        "findings": findings,
        "routes": cr.route_list({f["route"] for f in findings}),
        "vertex_gap_b64": vertex_bytes,
    }


def _finding(code, text, route, severity="info", teeth=None, value=None,
             unit=None, action=None, confidence="medium"):
    return {"code": code, "teeth": sorted(teeth or []), "value": value,
            "unit": unit, "severity": severity, "text": text, "action": action,
            "route": route, "confidence": confidence}


def build_findings(rows: list) -> list:
    out = []
    by_fdi = {r["fdi"]: r for r in rows}

    # 1. зубы без одноимённого антагониста: это не нарушение смыкания
    lonely = [r["fdi"] for r in rows if not r["has_antagonist"]]
    if lonely:
        out.append(_finding(
            "no_antagonist",
            f"Нет одноимённого антагониста: "
            f"{', '.join(map(str, sorted(lonely)))} — слабый контакт или его "
            f"отсутствие здесь ожидаемы и нарушением прикуса не считаются",
            "prostho", teeth=lonely, confidence="high",
            action="ортопед: оценить, нужно ли восстанавливать пару"))

    # 2. зубы вне окклюзии (антагонист есть, а контакта нет)
    out_occ = [r["fdi"] for r in rows
               if r["has_antagonist"] and r["min_gap_mm"] > NEAR_MM]
    if out_occ:
        out.append(_finding(
            "out_of_occlusion",
            f"Зубы вне окклюзии: {', '.join(map(str, sorted(out_occ)))} — "
            f"антагонист есть, но зазор больше {NEAR_MM} мм",
            "gnatho", severity="warning", teeth=out_occ,
            action="гнатолог: проверить, участвуют ли эти зубы в жевании"))

    # 3. касаются, но площади контакта нет. Признак берём по площади, а не по
    # минимальному зазору: одна вершина может подойти к антагонисту на 0.29 мм,
    # и по зазору это «контакт», хотя не соприкасается ни один участок
    # поверхности. Для врача важно именно второе.
    weak = [r["fdi"] for r in rows
            if r["has_antagonist"] and r["contact_area_mm2"] == 0
            and r["min_gap_mm"] <= NEAR_MM]
    if weak:
        out.append(_finding(
            "weak_contact",
            f"Смыкание неполное: {', '.join(map(str, sorted(weak)))} — "
            f"поверхности не соприкасаются, зазор до {NEAR_MM} мм",
            "gnatho", severity="warning", teeth=weak,
            action="гнатолог: уточнить в артикуляторе или копиркой"))

    # 4. перекос жевательной нагрузки между сторонами
    def side_area(quads):
        return sum(r["contact_area_mm2"] for r in rows if r["fdi"] // 10 in quads)
    right, left = side_area((1, 4)), side_area((2, 3))
    total = right + left
    if total > 1.0:
        share = max(right, left) / total
        if share >= 0.65:
            heavy, heavy_acc, light = (("правая", "правую", "левая") if right > left
                                       else ("левая", "левую", "правая"))
            out.append(_finding(
                "side_imbalance",
                f"Площадь контактов распределена неравномерно: "
                f"{share*100:.0f}% приходится на {heavy_acc} сторону "
                f"({right:.0f} мм² справа против {left:.0f} мм² слева)",
                "gnatho", severity="warning", value=round(share, 3), unit="доля",
                action=f"гнатолог: проверить, не перегружена ли {heavy} сторона "
                       f"и почему не работает {light}"))

    # 5. первый контакт: самый плотный, если он заметно опережает остальные
    contacting = sorted((r for r in rows if r["has_antagonist"]),
                        key=lambda r: r["min_gap_mm"])
    if len(contacting) >= 3:
        first = contacting[0]
        others = [r["min_gap_mm"] for r in contacting[1:]]
        if first["min_gap_mm"] + 0.15 < float(np.median(others)):
            out.append(_finding(
                "first_contact",
                f"Первым смыкается зуб {first['fdi']} (зазор "
                f"{first['min_gap_mm']:.2f} мм при медиане "
                f"{np.median(others):.2f} мм по остальным)",
                "gnatho", severity="warning", teeth=[first["fdi"]],
                value=first["min_gap_mm"], unit="мм", confidence="low",
                action="гнатолог: проверить на супраконтакт копиркой — "
                       "статичный скан отличить преждевременный контакт "
                       "от нормального первого не может"))

    # 6. зуб держит нагрузку одной точкой
    single = [r["fdi"] for r in rows
              if r["contact_area_mm2"] > 0 and r["contact_spots"] == 1
              and r["fdi"] % 10 >= 4           # только жевательные зубы
              and r["has_antagonist"]]         # без пары одна точка — норма
    if single:
        out.append(_finding(
            "single_spot",
            f"Контакт одной точкой на жевательных зубах: "
            f"{', '.join(map(str, sorted(single)))} — в норме бугорков несколько",
            "gnatho", teeth=single, confidence="low",
            action="гнатолог: возможна точечная перегрузка"))
    return out


def print_report(res: dict) -> None:
    print(f"\n{'='*70}\nПРИКУС: карта контактов — кейс {res['case_id']}\n{'='*70}")
    print(f"{'зуб':>5} {'зазор,мм':>9} {'контакт,мм²':>12} {'пятен':>6} "
          f"{'доля пов.':>10}  состояние")
    for r in res["teeth"]:
        share = r["contact_area_mm2"] / max(r["surface_mm2"], 1e-6) * 100
        # «контакт» — только если есть соприкасающаяся площадь: см. weak_contact
        if not r["has_antagonist"]:
            state = "нет антагониста"
        elif r["contact_area_mm2"] > 0:
            state = ("плотный контакт" if r["min_gap_mm"] <= TIGHT_MM
                     else "контакт")
        elif r["min_gap_mm"] <= NEAR_MM:
            state = "неполное смыкание"
        else:
            state = "ВНЕ ОККЛЮЗИИ"
        print(f"{r['fdi']:>5} {r['min_gap_mm']:9.2f} {r['contact_area_mm2']:12.1f} "
              f"{r['contact_spots']:>6} {share:9.1f}%  {state}")

    print(f"\n{'-'*70}\nНАХОДКИ")
    if not res["findings"]:
        print("   отклонений не выявлено")
    for f in res["findings"]:
        mark = "!" if f["severity"] == "warning" else " "
        print(f"  {mark} {f['text']}")
        if f["action"]:
            print(f"      -> {f['action']}")
    print(f"\n{'-'*70}")
    print("ЧЕРНОВИК. Пороги не подтверждены врачом; не является назначением.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--json", default=None,
                    help="куда положить результат (по умолчанию — в кейс)")
    args = ap.parse_args()

    res = analyze(args.case_id)
    print_report(res)
    out = Path(args.json) if args.json else CASES / args.case_id / "occlusion.json"
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"сохранено: {out}  ({out.stat().st_size/1024:.0f} КБ)")


if __name__ == "__main__":
    main()
