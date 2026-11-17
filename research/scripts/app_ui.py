"""Stomadent — тестовый UI для детекции патологий на панорамных снимках.

Запуск:
    py research/scripts/app_ui.py [--port 7860]

Открыть: http://localhost:7860
Функции: загрузка PNG/JPG/DICOM, детекция YOLO12x-champion, слайдер порога
уверенности, таблица находок с локализацией, скачивание результата.
"""

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "output/runs/yolo12x_imgsz960/weights/best.pt"
EXAMPLES_DIR = ROOT / "datasets/dentex_yolo/images/val"

CLASS_RU = {
    "Impacted": "Ретенированный зуб",
    "Caries": "Кариес",
    "Periapical Lesion": "Периапикальное поражение",
    "Deep Caries": "Глубокий кариес",
}

model = YOLO(str(WEIGHTS))


def load_image(path: str) -> Image.Image:
    """Читает PNG/JPG или DICOM -> 8-bit RGB."""
    if path.lower().endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        arr = arr * slope + intercept
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            lo, hi = float(wc) - float(ww) / 2, float(wc) + float(ww) / 2
            arr = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
        else:
            arr = (arr - arr.min()) / max(arr.ptp(), 1)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 1.0 - arr
        img = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        img = Image.open(path)
    return img.convert("RGB")


def localize(cx: float, cy: float, w: int, h: int) -> str:
    """Приблизительная локализация: на ОПТГ право пациента — слева на снимке."""
    side = "правая" if cx < w / 2 else "левая"
    jaw = "верхняя челюсть" if cy < h / 2 else "нижняя челюсть"
    return f"{jaw}, {side} сторона"


def predict(file, conf: float):
    if file is None:
        return None, [], "Загрузите снимок (PNG/JPG/DICOM)."
    img = load_image(file)
    w, h = img.size
    res = model.predict(img, imgsz=960, conf=conf, max_det=30, verbose=False)[0]

    annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
    tmp = Path(tempfile.mkdtemp()) / "result.png"
    Image.fromarray(annotated).save(tmp)

    rows, counts = [], {}
    for box, cls, cf in zip(res.boxes.xyxy.cpu().numpy(),
                            res.boxes.cls.cpu().numpy().astype(int),
                            res.boxes.conf.cpu().numpy()):
        name = res.names[cls]
        ru = CLASS_RU.get(name, name)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        rows.append([ru, f"{cf:.0%}", localize(cx, cy, w, h),
                     f"{int(box[2]-box[0])}×{int(box[3]-box[1])}"])
        counts[ru] = counts.get(ru, 0) + 1

    rows.sort(key=lambda r: -float(r[1].rstrip("%")))
    summary = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    text = f"Найдено находок: {len(rows)}. {summary}" if rows else \
        f"Находок с уверенностью ≥ {conf:.0%} не найдено. Попробуйте снизить порог."
    return str(tmp), rows, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    examples = [str(p) for p in sorted(EXAMPLES_DIR.glob("*.png"))[:6]]

    with gr.Blocks(title="Stomadent — детекция патологий") as app:
        gr.Markdown("# Stomadent — AI-анализ панорамного снимка\n"
                    "Модель: YOLO12x (DENTEX, mAP50 0.647). "
                    "Классы: ретенированный зуб, кариес, глубокий кариес, периапикальное поражение.")
        with gr.Row():
            with gr.Column():
                inp = gr.File(label="Снимок (PNG / JPG / DICOM)", file_types=[".png", ".jpg", ".jpeg", ".dcm"])
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Порог уверенности")
                btn = gr.Button("Анализировать", variant="primary")
                gr.Examples(examples=examples, inputs=inp, label="Примеры из валидации DENTEX")
            with gr.Column():
                out_img = gr.Image(label="Результат")
                out_text = gr.Textbox(label="Сводка", lines=2)
                out_tbl = gr.Dataframe(headers=["Находка", "Уверенность", "Расположение (прибл.)", "Размер, px"],
                                       label="Таблица находок")
        btn.click(predict, inputs=[inp, conf], outputs=[out_img, out_tbl, out_text])
        gr.Markdown("_Локализация приблизительная (по центру бокса). "
                    "Результат не является медицинским диагнозом._")

    app.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
