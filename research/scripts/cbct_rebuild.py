"""Пересборка мешей и находок КЛКТ-кейсов после смены POSTPROC_VERSION.

Зачем нужен отдельный скрипт. При старте сервера `_load_persisted` сверяет
`info.postproc` с текущей версией и, если та изменилась, УДАЛЯЕТ mesh_*.glb
прямо при импорте модуля. Дальше меш пересобирается при первом открытии кейса,
и врач ждёт минуту на ровном месте. Плюс ловушка из CLAUDE.md: скрипт,
пересобирающий меши, обязан записать `info` через `_save_meta`, иначе
следующий импорт снесёт их снова.

Здесь пересобирается всё, для чего уже есть маска: меш, info и находки, после
чего meta.json пишется на диск. Сегментация заново НЕ гоняется — маски на
месте, работа чисто вычислительная (marching cubes), GPU не нужен.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_rebuild.py            # все кейсы
    .venv/Scripts/python.exe research/scripts/cbct_rebuild.py client4    # выборочно
    .venv/Scripts/python.exe research/scripts/cbct_rebuild.py --list     # что есть
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402


def pairs(case_ids: list | None, models: list | None = None) -> list:
    """Пары (кейс, модель), для которых есть маска."""
    out = []
    for cid, c in sorted(cb.cases.items()):
        if case_ids and cid not in case_ids:
            continue
        for model, path in c["masks"].items():
            if models and model not in models:
                continue
            if path and Path(path).is_file():
                out.append((cid, model))
    return out


MIN_GLB_BYTES = 10_000    # меньше — меш не записался, что бы ни вернул экспорт


def rebuild(case_id: str, model: str, attempts: int = 2) -> dict:
    """Пересборка одной пары. Записанный файл ПРОВЕРЯЕТСЯ.

    Экспорт trimesh может завершиться без исключения, не оставив файла: в
    прогоне 25.08 так вышло у case_1787223539, повтор прошёл с первого раза
    (разовый сбой файловой системы). Молчаливое «готово» при отсутствующем
    меше опаснее падения — врач открыл бы кейс и ждал минуту на пустом месте.
    """
    c = cb.get_case(case_id)
    glb = cb.CASES_DIR / case_id / f"mesh_{model}.glb"
    t0 = time.perf_counter()
    for attempt in range(1, attempts + 1):
        cb.build_mesh(c, model)
        size = glb.stat().st_size if glb.is_file() else 0
        if size >= MIN_GLB_BYTES:
            break
        print(f"   меш не записался ({size} б), попытка {attempt} из {attempts}",
              flush=True)
    else:
        raise RuntimeError(f"меш так и не записан после {attempts} попыток")

    c["model"] = model
    c["info"] = cb.compute_info(c, model)
    c["findings"] = cb.compute_findings(c, model)
    cb._save_meta(c)          # без этого следующий импорт снесёт меш
    return {"seconds": time.perf_counter() - t0,
            "mb": glb.stat().st_size / 1e6,
            "teeth": len(c["info"].get("teeth_present", [])),
            "findings": len((c["findings"] or {}).get("findings", []))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", nargs="*", help="пусто — все кейсы с маской")
    ap.add_argument("--list", action="store_true", help="показать и выйти")
    # Пересборка одной модели: кейс держит меши всех трёх, а врач смотрит одну.
    # Ждать лишние 8 минут ради невидимых мешей незачем.
    ap.add_argument("--model", action="append",
                    help="только эта модель (можно повторять)")
    # Актуальные меши по умолчанию пропускаются: пересборка всех 53 пар идёт
    # часами, и после сбоя или частичного прогона начинать с нуля незачем.
    ap.add_argument("--force", action="store_true",
                    help="пересобрать и актуальные меши")
    args = ap.parse_args()

    todo = pairs(args.case_id or None, args.model)
    print(f"версия постпроцессинга: {cb.POSTPROC_VERSION}")
    print(f"пар кейс/модель с готовой маской: {len(todo)}")
    for cid, model in todo:
        c = cb.get_case(cid)
        info = c.get("info") or {}
        # Состояние ПАРЫ, а не кейса: `info` в кейсе один, а мешей несколько.
        # Раньше строка говорила «актуален» про модель, чей меш уже удалён как
        # устаревший, — список врал ровно там, где на него и смотрят.
        mesh_ok = (c.get("meshes") or {}).get(model) == cb.POSTPROC_VERSION
        info_ok = (info.get("model") == model
                   and info.get("postproc") == cb.POSTPROC_VERSION)
        state = ("актуален" if mesh_ok and info_ok
                 else "нет меша" if not mesh_ok
                 else "меш есть, находки от другой модели")
        print(f"   {cid:<18} {model:<12} {state}")
    if args.list:
        return

    ok, bad = 0, []
    for cid, model in todo:
        c = cb.get_case(cid)
        if (not args.force
                and (c.get("meshes") or {}).get(model) == cb.POSTPROC_VERSION):
            print(f"\n-- {cid} / {model}: меш актуален, пропуск", flush=True)
            ok += 1
            continue
        print(f"\n-> {cid} / {model}", flush=True)
        try:
            r = rebuild(cid, model)
            ok += 1
            print(f"   готово за {r['seconds']:.0f} с, меш {r['mb']:.2f} МБ, "
                  f"зубов {r['teeth']}, находок {r['findings']}", flush=True)
        except Exception as exc:  # noqa: BLE001 — один кейс не должен ронять остальные
            bad.append((cid, model, str(exc)))
            # Печатаем стек, а не одну строку: «не найден mesh_ours.glb» без
            # него читается как отсутствие меша, хотя падение может быть на
            # любом шаге после сборки — и место сбоя приходится угадывать.
            import traceback
            print(f"   ОШИБКА: {exc}", flush=True)
            traceback.print_exc()
    print(f"\nпересобрано: {ok} из {len(todo)}")
    for cid, model, err in bad:
        print(f"   не вышло {cid}/{model}: {err}")


if __name__ == "__main__":
    main()
