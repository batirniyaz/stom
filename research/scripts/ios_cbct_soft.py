"""Толщина мягких тканей: расстояние от поверхности десны до кости.

Зачем. Ни одно исследование по отдельности этого не даёт: скан видит десну и
не видит кость, КЛКТ видит кость, а мягкие ткани на нём почти не отличаются
от воздуха по плотности. Разница между двумя поверхностями и есть толщина —
она появляется только после совмещения.

Кому нужно. Имплантологу и ортопеду: при тонкой десне металл импланта
просвечивает сквозь неё в переднем отделе, а край коронки со временем
оголяется. Толщину принято смотреть до операции, чтобы решить, нужна ли
подсадка соединительной ткани.

Как считаем.

  1. Берём меш десны из скана (`gum_upper` / `gum_lower`).
  2. Переводим его матрицей совмещения в систему координат КЛКТ — своей для
     КАЖДОЙ челюсти: они подвижны друг относительно друга, и общая матрица
     даёт втрое худшее совпадение (см. ios_cbct_register).
  3. Строим поверхность кости из маски челюсти КЛКТ.
  4. Из каждой точки десны идём ВНУТРЬ по нормали и ищем, где встретим кость.

Почему по нормали, а не «до ближайшей точки кости». Ближайшая точка может
лежать сбоку — например, у соседнего зуба или на другой стенке лунки, — и
тогда «толщина» окажется меньше настоящей просто из-за геометрии. Толщина
ткани определена вдоль нормали к поверхности, и мерить надо так же.

Запуск:
    .venv/Scripts/python.exe research/scripts/ios_cbct_soft.py \
        --cbct case_1788759007 --ios s_52532e543d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb            # noqa: E402
import ios_cbct_register as reg      # noqa: E402

IOS_CASES = Path(__file__).resolve().parents[2] / "research_ios" / "web_cases"

# Луч из точки десны внутрь: дальше этого кость искать бессмысленно —
# столько мягкой ткани над костью не бывает нигде, кроме языка и щеки,
# а они в измерение попадать не должны.
MAX_DEPTH_MM = 8.0
STEP_MM = 0.1                  # шаг вдоль нормали; воксель КЛКТ 0.2–0.3 мм

# Порог «это кость» — по СВОЕЙ базе, а не из литературы. Шкала КЛКТ не
# количественная (в проекте это уже проверялось на эндодонтии), поэтому
# литературные 300 HU здесь ничего не значат. Замер на кейсе клиники:
#
#                              5%     медиана    95%
#   кость глубоко в челюсти    880      1770     2470
#   мягкие ткани и воздух    -1049      -325      180
#
# Порог 300 попадал в шум мягких тканей: на поверхности десны ему
# соответствовало 45% точек. Ставим в разрыв между группами.
BONE_HU = 500.0

# Точки, где толщина мерится неправдоподобно: либо десна оказалась ВНУТРИ
# кости (ошибка совмещения), либо кость не встретилась вовсе (там её нет —
# например, нёбо или переходная складка).
MIN_OK_MM = 0.1


def gum_meshes(ios_case: str) -> dict:
    path = IOS_CASES / ios_case / "scan.glb"
    if not path.is_file():
        raise SystemExit(f"нет скана {path}")
    scene = trimesh.load(str(path), process=False)
    out = {}
    for name, geom in scene.geometry.items():
        short = name.split("/")[-1]
        if short.startswith("gum_"):
            out[short.split("_")[1]] = geom
    return out


def bone_field(cbct_case: str, model: str | None = None):
    """Плотность КЛКТ как функция точки в экранных мм.

    Возвращает (проверка «это кость?», размеры). Работаем прямо по объёму, а
    не по мешу: меш строится с размытием и сглаживанием, а нам нужна честная
    граница плотности.
    """
    c = cb.get_case(cbct_case)
    if c is None:
        raise SystemExit(f"нет кейса КЛКТ {cbct_case}")
    model = model or cb.mask_model_for(c)
    vol, mask = cb.arrays_for(c, model)
    spacing = np.array(c["spacing"], float)      # (z, y, x)
    jaw = np.isin(mask, cb.JAW_FOR_MIDLINE)
    # Зубы, пульпа и реставрации: плотнее кости, и луч не должен считать их
    # костью. Пульпа = класс зуба + 100 (соглашение ToothFairy).
    tooth = np.isin(mask, list(cb.TEETH_CLASSES)
                    + [t + 100 for t in cb.TEETH_CLASSES])
    if hasattr(cb, "PROSTHETICS_CLASSES"):
        tooth |= np.isin(mask, list(cb.PROSTHETICS_CLASSES))

    def is_bone(pts_viewer: np.ndarray) -> np.ndarray:
        """pts в экранных мм -> True там, где кость."""
        # Экранные оси -> оси тома. reg._to_viewer переставляет (z,y,x) в
        # экранный порядок; здесь нужен обратный ход.
        zyx = reg._from_viewer(pts_viewer) / spacing
        idx = np.round(zyx).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
        res = np.zeros(len(pts_viewer), bool)
        if not ok.any():
            return res
        i = idx[ok]
        dense = vol[i[:, 0], i[:, 1], i[:, 2]] >= BONE_HU
        # Маску челюсти НЕ требуем. Замер: в зоне шеек она не покрывает
        # альвеолярный гребень вовсе — на глубине 0–3 мм от десны в неё
        # попадает 0% точек, кость по маске начинается только с 5 мм.
        # Тонкая вестибулярная пластинка в сегментацию не попала, и требование
        # «внутри маски» отодвигало границу вглубь на несколько миллиметров.
        # Вместо этого исключаем зубы: они плотнее кости, и упираться в них
        # луч не должен.
        not_tooth = ~tooth[i[:, 0], i[:, 1], i[:, 2]]
        res[ok] = dense & not_tooth
        return res

    return is_bone, spacing


def thickness(gum: trimesh.Trimesh, matrix: np.ndarray, is_bone) -> dict:
    """Толщина ткани в каждой вершине десны, мм. None там, где не измерено."""
    v = reg._apply(matrix, np.asarray(gum.vertices, dtype=float))
    n = np.asarray(gum.vertex_normals, dtype=float)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    # Нормаль меша смотрит наружу — значит внутрь идём против неё. Знак
    # проверяем по данным: если снаружи кости больше, чем внутри, значит
    # нормали вывернуты, и направление надо развернуть.
    probe = 1.0
    out_hits = is_bone(v + n * probe).mean()
    in_hits = is_bone(v - n * probe).mean()
    direction = -n if in_hits >= out_hits else n

    steps = np.arange(STEP_MM, MAX_DEPTH_MM + STEP_MM, STEP_MM)
    found = np.full(len(v), np.nan)
    todo = np.ones(len(v), bool)
    for s in steps:
        if not todo.any():
            break
        pts = v[todo] + direction[todo] * s
        hit = is_bone(pts)
        if hit.any():
            idx = np.where(todo)[0][hit]
            found[idx] = s
            todo[idx] = False
    # Возвращаем и исходные вершины. Зона измерения (шейка зуба, вестибулярная
    # сторона) определяется по скану, а толщина считается по преобразованным
    # точкам — если перепутать, зуб и десна окажутся в разных системах
    # координат. Первый прогон так и вышел: до «ближайшей» точки десны
    # получалось 70 мм.
    return {"thickness": found, "points": v,
            "source": np.asarray(gum.vertices, dtype=float)}


def gingival_zone(ios_case: str, jaw: str) -> tuple:
    """Точки десны, ГДЕ ЕЁ ИЗМЕРЯЮТ: узкая полоса у шейки, вестибулярно.

    Это главное в измерении, и первая версия здесь ошиблась: она мерила всю
    десну подряд и получила медиану 5.8 мм на верхней челюсти. Столько десны
    не бывает — в выборку попали нёбо (слизистая там законно 3–7 мм), щёчная
    слизистая и переходная складка, где до кости далеко по анатомии.

    Клинически биотип смотрят в одном месте: прикреплённая десна на
    ВЕСТИБУЛЯРНОЙ стороне, на 1–3 мм апикальнее края коронки. Там она тонкая,
    там просвечивает имплант и там же оголяется край коронки.

    Возвращает (индексы вершин, {fdi: индексы этого зуба}).
    """
    import ios_geometry as G
    teeth, _gums = G.load_case(ios_case)
    quads = (1, 2) if jaw == "upper" else (3, 4)
    mine = {t: m for t, m in teeth.items() if t // 10 in quads}
    if len(mine) < 4:
        return None, {}
    frame = G.arch_frame(teeth)

    # Наружу от центра дуги СВОЕЙ челюсти: по всем зубам сразу центр уехал бы
    # вбок (та же оговорка, что в карточке зуба на КЛКТ).
    centres = np.array([np.asarray(m.vertices).mean(0) for m in mine.values()])
    arch_centre = centres.mean(0)
    return frame, {"teeth": mine, "arch_centre": arch_centre}


def by_tooth(ios_case: str, jaw: str, v_scan: np.ndarray, th: np.ndarray) -> dict:
    """Толщина у каждого зуба в зоне измерения биотипа.

    Врачу нужна толщина в конкретном месте («у 13-го тонко»), а не среднее по
    челюсти: биотип меняется от зуба к зубу.
    """
    frame, data = gingival_zone(ios_case, jaw)
    if frame is None:
        return {}
    ok = ~np.isnan(th) & (th >= MIN_OK_MM)
    if ok.sum() < 20:
        return {}
    gum_pts = v_scan[ok]          # координаты СКАНА: в них же лежат зубы
    vals = th[ok]
    tree = cKDTree(gum_pts)

    out = {}
    for fdi, mesh in data["teeth"].items():
        tv = np.asarray(mesh.vertices, dtype=float)
        # Шейка — ТАМ, ГДЕ К ЗУБУ ПРИЛЕГАЕТ ДЕСНА, а не там, где её ждёт
        # формула. Предыдущая версия искала её по оси дуги (перцентиль высоты
        # с поправкой на челюсть) и промахивалась: замер показал, что у зуба
        # 41 десна оказалась у верхнего края зуба, а у 46 — у нижнего, то есть
        # правило «у верхних сверху, у нижних снизу» не выполняется. Признак,
        # взятый из самих данных, надёжнее допущения об ориентации — в этом
        # проекте на таких допущениях уже несколько раз теряли смысл измерения.
        d_gum, _ = tree.query(tv, k=1)
        neck = tv[d_gum <= 1.5]
        if len(neck) < 20:
            continue
        # Вестибулярно: наружу от центра дуги. Язычную и нёбную сторону не
        # берём — там другая ткань и другая норма.
        out_dir = neck - data["arch_centre"]
        out_dir[:, 1] = 0.0                     # вдоль дуги, а не вверх
        centre_dir = np.asarray(mesh.vertices).mean(0) - data["arch_centre"]
        centre_dir[1] = 0.0
        buccal = neck[(out_dir @ centre_dir) > 0]
        if len(buccal) < 10:
            buccal = neck

        near = tree.query_ball_point(buccal[::3], r=2.0)
        got = sorted({i for lst in near for i in lst})
        if len(got) < 15:
            continue
        out[int(fdi)] = round(float(np.median(vals[got])), 2)
    return out


def analyze(cbct_case: str, ios_case: str, model: str | None = None) -> dict:
    reg_file = IOS_CASES / ios_case / f"register_{cbct_case}.json"
    if not reg_file.is_file():
        raise SystemExit(
            f"нет совмещения {reg_file.name} — сначала ios_cbct_register.py")
    R = json.loads(reg_file.read_text(encoding="utf-8"))
    is_bone, _sp = bone_field(cbct_case, model)
    gums = gum_meshes(ios_case)

    jaws = {}
    for jaw, gum in gums.items():
        # Матрица СВОЯ у каждой челюсти. Взять общую нельзя: нижняя подвижна
        # относительно верхней, и в КЛКТ рот стоит не так, как при сканировании.
        j = (R.get("jaws") or {}).get(jaw)
        if j is None:
            continue
        M = np.array(j["matrix"], float).reshape(4, 4)
        res = thickness(gum, M, is_bone)
        th, v_scan = res["thickness"], res["source"]
        measured = th[~np.isnan(th)]
        measured = measured[measured >= MIN_OK_MM]
        per_tooth = by_tooth(ios_case, jaw, v_scan, th)
        # ГЛАВНАЯ ОГОВОРКА МЕТОДА. Толщина десны — 0.5–1.5 мм, а погрешность
        # совмещения на этом кейсе 0.5–0.7 мм по зубам. Величины одного
        # порядка, значит отдельное число «0.3 мм» не измерено, а получено в
        # пределах собственной ошибки. Поэтому всё, что тоньше погрешности,
        # отдаём как «тоньше порога», а не как значение: точное число здесь
        # было бы выдумкой, как процент уверенности на находках КЛКТ.
        limit = float(j.get("tooth_median_mm") or 0.5)
        below = sorted(t for t, val in per_tooth.items() if val <= limit)
        # Сводка — по зоне измерения, а не по всей десне: медиана «по всему
        # мешу» включает нёбо и щёку и к биотипу отношения не имеет.
        zone_vals = list(per_tooth.values())
        jaws[jaw] = {
            "vertices": int(len(th)),
            "measured": int(len(measured)),
            "median_mm": round(float(np.median(zone_vals)), 2) if zone_vals else None,
            "p10_mm": round(float(np.percentile(zone_vals, 10)), 2) if zone_vals else None,
            "p90_mm": round(float(np.percentile(zone_vals, 90)), 2) if zone_vals else None,
            "by_tooth": per_tooth,
            "below_resolution": below,
            "resolution_mm": round(limit, 2),
            "registration_tooth_median_mm": j.get("tooth_median_mm"),
            "registration_good": j.get("good"),
            "note": (f"Погрешность совмещения на этой челюсти {limit} мм. "
                     f"Толщина десны того же порядка, поэтому значения ниже "
                     f"этого порога означают «тоньше {limit} мм», а не точную "
                     f"величину. Метод различает толстую и тонкую десну, но не "
                     f"даёт десятых долей миллиметра."),
        }
    return {"cbct_case": cbct_case, "ios_case": ios_case, "jaws": jaws,
            "draft": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cbct", required=True)
    ap.add_argument("--ios", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    res = analyze(a.cbct, a.ios, a.model)
    for jaw, j in res["jaws"].items():
        ru = "верхняя" if jaw == "upper" else "нижняя"
        print(f"\n{ru} челюсть")
        print(f"  точек десны: {j['vertices']}, измерено: {j['measured']} "
              f"({100 * j['measured'] / max(j['vertices'], 1):.0f}%)")
        if j["median_mm"] is None:
            print("  толщина не измерена")
            continue
        print(f"  толщина: медиана {j['median_mm']} мм, "
              f"10-й перцентиль {j['p10_mm']}, 90-й {j['p90_mm']}")
        print(f"  качество совмещения этой челюсти: "
              f"{j['registration_tooth_median_mm']} мм по зубам")
        print(f"  ВНИМАНИЕ: погрешность совмещения {j['resolution_mm']} мм — "
              f"того же порядка, что измеряемая толщина")
        if j["by_tooth"]:
            print("  по зубам (медиана, мм):")
            for t, val in sorted(j["by_tooth"].items()):
                mark = f"тоньше {j['resolution_mm']}" if t in j["below_resolution"]                     else f"{val}"
                print(f"    {t}: {mark}")
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"\nсохранено: {a.json}")


if __name__ == "__main__":
    main()
