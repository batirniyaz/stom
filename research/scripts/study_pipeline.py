"""Обработка загруженного исследования: КЛКТ, скан, панорама.

Каждый тип считается своим конвейером, но снаружи они выглядят одинаково:
очередь (`jobs.py`) даёт `study_id` и функцию прогресса, обработчик кладёт
результат и сообщает стадии. Всё, что требует видеокарты, идёт СТРОГО по
одному — очередь за этим и следит.

Где лежат результаты. Пока конвейеры пишут туда же, где живут демо-кейсы
(`research_3d/web_cases`, `research_ios/web_cases`), а исследование хранит
имя кейса в `legacy_case`. Так просмотр работает существующими вьюерами без
единой правки — переезд данных внутрь пациента это отдельный этап, и делать
его одновременно с загрузкой значило бы менять две вещи разом и не понять,
что сломалось.

Определение челюсти у скана. Сканер отдаёт два файла, и какой из них верхний
— написано в имени, но у каждой программы по-своему (`UpperJaw`, `maxilla`,
`_upper_`, «верх»). Если имя молчит, спрашиваем геометрию: верхняя челюсть
своей жевательной поверхностью смотрит ВНИЗ, нижняя — вверх, и это видно по
тому, с какой стороны от центра масс лежит выпуклая часть подковы.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from pathlib import Path as pathlib_Path

import patients as pat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
IOS_CASES = REPO / "research_ios" / "web_cases"

MESH_SUFFIXES = {".stl", ".ply", ".obj", ".off"}
# Порядок предпочтения при дублях: OBJ несёт текстуру, PLY — цвет
# вершин, STL — только геометрию.
MESH_PRIORITY = [".obj", ".ply", ".stl", ".off"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def source_files(sid: str) -> list[Path]:
    d = pat.study_dir(sid) / "source"
    return sorted(p for p in d.rglob("*") if p.is_file())


def guess_kind(names: list[str]) -> str:
    """Тип исследования по составу файлов. Врач подтверждает, а не вводит.

    Он приносит «панораму» и «скан», а не «одиночный DICOM» и «набор STL» —
    угадывать формат его дело не должно.
    """
    suf = [Path(n).suffix.lower() for n in names]
    if any(s in MESH_SUFFIXES for s in suf):
        return "ios"
    if any(s == ".zip" for s in suf):
        return "cbct"
    dicoms = [s for s in suf if s in (".dcm", "")]
    if len(dicoms) > 5:                     # серия срезов — это объём
        return "cbct"
    if any(s in IMAGE_SUFFIXES for s in suf) or len(dicoms) == 1:
        return "opg"
    return "cbct"


# --------------------------------------------------------- скан (IOS)

def _jaw_of(path: Path) -> str | None:
    """Верх или низ по имени файла. None — имя молчит, спросим геометрию."""
    n = path.stem.lower()
    if re.search(r"upper|maxill|верх|_up\b|-up\b|^up", n):
        return "upper"
    if re.search(r"lower|mandib|нижн|низ|_low\b|-low\b|^low", n):
        return "lower"
    return None


def _jaw_by_geometry(path: Path) -> str:
    """Челюсть по форме: зубы верхней смотрят вниз, нижней — вверх.

    Берём ось наименьшего разброса (вертикаль подковы, см. ios_canonical) и
    сравниваем, с какой стороны от середины лежит масса: у верхней челюсти
    нёбо широкое и тяжёлое сверху, коронки внизу.
    """
    import numpy as np
    import trimesh
    m = trimesh.load(str(path), process=False, force="mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    v = v - v.mean(0)
    # вертикаль — направление наименьшего размаха (челюсть плоская)
    spread = v.max(0) - v.min(0)
    axis = int(np.argmin(spread))
    z = v[:, axis]
    # Тонкий край (режущие края и бугры) против массивного основания.
    return "upper" if np.median(z) < 0 else "lower"


def _unpack_archives(sid: str, progress) -> None:
    """Распаковать ZIP рядом с собой.

    Сканер отдаёт папку с десятком файлов, и врач присылает её архивом —
    это естественный для него способ, а не исключение. Раньше архив
    принимался, но конвейер искал в нём .stl и не находил ничего.
    """
    src = pat.study_dir(sid) / "source"
    for z in list(src.glob("*.zip")):
        progress("распаковка архива")
        import zipfile
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Имя из архива в файловую систему не пускаем как есть:
                # путь приходит извне и может содержать ../
                name = re.sub(r"[^\w.\-]+", "_", Path(info.filename).name)[:120]
                if not name:
                    continue
                with zf.open(info) as fin, open(src / name, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        z.unlink()                    # архив больше не нужен, место дорого


def handle_ios(sid: str, progress) -> None:
    _unpack_archives(sid, progress)
    files = [p for p in source_files(sid) if p.suffix.lower() in MESH_SUFFIXES]
    if not files:
        raise ValueError("в загруженном нет файлов скана (.stl, .ply, .obj)")

    # У Exocad в папке лежат и STL, и PLY, и OBJ одной челюсти. Берём по
    # одному формату, иначе одна и та же челюсть попадёт дважды и вторая
    # перезапишет первую как «другая».
    by_stem: dict[str, pathlib_Path] = {}
    for f in files:
        prev = by_stem.get(f.stem.lower())
        if prev is None or MESH_PRIORITY.index(f.suffix.lower()) <                 MESH_PRIORITY.index(prev.suffix.lower()):
            by_stem[f.stem.lower()] = f
    files = sorted(by_stem.values())

    jaws: dict[str, Path] = {}
    unknown: list[Path] = []
    for f in files:
        j = _jaw_of(f)
        if j and j not in jaws:
            jaws[j] = f
        else:
            unknown.append(f)
    for f in unknown:
        progress("определяем челюсть по геометрии")
        j = _jaw_by_geometry(f)
        jaws.setdefault(j, f)

    if not jaws:
        raise ValueError("не удалось определить, где верхняя и нижняя челюсть")

    case_id = sid                      # имя кейса = id исследования: связь явная
    out_dir = IOS_CASES / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    args = []
    for j, f in jaws.items():
        args += [f"--{j}", str(f)]

    progress("сегментация скана нейросетью")
    _run([str(PY), str(HERE / "ios_build_mesh.py"), *args,
          "--out", str(out_dir / "scan.glb")])

    progress("исходный скан")
    _run([str(PY), str(HERE / "ios_build_raw.py"), *args,
          "--out", str(out_dir / "raw.glb")])

    # Карта контактов и находки считаются только при двух челюстях: без
    # антагониста смыкания не существует, и считать там нечего.
    if len(jaws) == 2:
        progress("карта окклюзионных контактов")
        _run([str(PY), str(HERE / "ios_occlusion_map.py"), case_id])

    progress("находки по скану")
    _run([str(PY), str(HERE / "ios_findings.py"), case_id])

    pat.update_study(sid, legacy_case=case_id, legacy_kind="ios")


# ------------------------------------------------------------- КЛКТ

def handle_cbct(sid: str, progress) -> None:
    import time
    import cbct_backend as cbct

    files = source_files(sid)
    if not files:
        raise ValueError("нет файлов исследования")
    progress("чтение DICOM-серии")
    payload = [(f.name, f.read_bytes()) for f in files]
    c = cbct.import_upload(payload)
    case_id = c["case_id"]
    scanner = " ".join(x for x in (
        (c.get("scanner") or {}).get("manufacturer", ""),
        (c.get("scanner") or {}).get("model", "")) if x)
    pat.update_study(sid, legacy_case=case_id, legacy_kind="cbct",
                     scanner=scanner)

    model = "sp"                    # модель по умолчанию, как в демо
    pat.update_study(sid, model=model)   # чтобы ссылка вела в ту же модель
    ok, msg = cbct.start_segment(case_id, model)
    if not ok:
        raise RuntimeError(msg)

    # Сегментация идёт в своём потоке внутри cbct_backend. Ждём её здесь, а не
    # отпускаем задачу: иначе очередь возьмёт следующую и на видеокарте
    # окажутся две сети сразу — ровно то, ради чего очередь и заводилась.
    stages = {"predicting": "сегментация нейросетью",
              "meshing": "построение 3D-модели"}
    while True:
        cur = cbct.get_case(case_id) or {}
        stage = cur.get("stage", "")
        if stage == "done":
            break
        if stage == "error":
            raise RuntimeError(cur.get("error") or "сегментация не удалась")
        progress(stages.get(stage, "обработка"))
        time.sleep(2.0)


# ------------------------------------------- панорама и прицельный снимок

def handle_2d(sid: str, progress) -> None:
    """2D-снимок: ансамбль YOLO, результат рядом с исследованием.

    Раньше 2D вообще не сохранялся — снимок жил во временном файле и
    удалялся сразу после предсказания. Здесь он остаётся у пациента вместе с
    находками, иначе к нему нельзя вернуться и не с чем сравнивать динамику.
    """
    import json
    files = [p for p in source_files(sid)
             if p.suffix.lower() in IMAGE_SUFFIXES or p.suffix.lower() == ".dcm"]
    if not files:
        raise ValueError("нет файла снимка")
    progress("анализ снимка ансамблем")
    import demo_server as srv            # веса уже загружены в процессе демо
    img = files[0]
    res = srv.run_inference(str(img), conf=0.25)
    # Размеченную картинку (`image_b64`) НЕ храним: это 1.4 МБ base64 на
    # каждый снимок, а перерисовать её можно в любой момент — исходник лежит
    # рядом в source/. Храним то, что нельзя восстановить: сами находки.
    res.pop("image_b64", None)
    out = pat.study_dir(sid) / "derived"
    out.mkdir(parents=True, exist_ok=True)
    (out / "findings2d.json").write_text(
        json.dumps({"file": img.name, **res}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    # Зубы с номерами и находки на зуб. Отдельным шагом и без права уронить
    # исследование: основной анализ уже сохранён, а зубы досчитаются при
    # первом открытии снимка, если здесь что-то пойдёт не так.
    progress("нумерация зубов и находки на зуб")
    try:
        srv.teeth2d_for(sid)
    except Exception as exc:  # noqa: BLE001
        import eventlog
        eventlog.add("study", f"зубы 2D не посчитаны: {exc}", study=sid)
    # Снимок — в набор для разметки врачами (см. annotation_feed). Тоже без
    # права уронить исследование: разметка — побочный продукт приёма.
    progress("в набор для разметки врачами")
    try:
        import annotation_feed
        annotation_feed.add_study(sid, img, loader=srv.load_image)
    except Exception as exc:  # noqa: BLE001
        import eventlog
        eventlog.add("study", f"снимок не передан в разметку: {exc}", study=sid)


def _run(cmd: list[str]) -> None:
    """Запуск шага конвейера отдельным процессом.

    Отдельным, а не импортом, по двум причинам: видеопамять освобождается
    гарантированно при выходе процесса (в этом проекте её нехватка проявляется
    не ошибкой, а замедлением в десятки раз), и падение шага не уносит с собой
    веб-сервер.
    """
    env_cmd = cmd
    r = subprocess.run(env_cmd, cwd=str(REPO), capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env={**_env()})
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError("шаг не выполнен: " + " / ".join(tail))


def _env() -> dict:
    import os
    e = dict(os.environ)
    # Консоль этой машины живёт в cp1252, а проект — в UTF-8. Без этого
    # русские строки в выводе шага роняют его на UnicodeEncodeError.
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def register_all() -> None:
    import jobs
    jobs.register("ios", handle_ios)
    jobs.register("cbct", handle_cbct)
    jobs.register("opg", handle_2d)
    jobs.register("periapical", handle_2d)
