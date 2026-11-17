"""PDF-отчёт по кейсу КЛКТ — то, что клиника отдаёт пациенту на руки.

Зачем. Вся работа до сих пор жила на экране: врач смотрит находки во вьюере и
там же их подтверждает. Но пациент уходит домой с бумагой, и направление к
смежному специалисту тоже идёт бумагой. Без отчёта результат нельзя ни
показать, ни передать, ни подшить в карту.

Что внутри и почему именно это:

  * ЗУБНАЯ ФОРМУЛА первой страницей. Врач читает карту зубов, а не список
    кодов: сначала «где», потом «что». Отсутствующие зубы показаны пустыми
    клетками, зубы с находками — цветом по важности.
  * НАХОДКИ ПО СПЕЦИАЛЬНОСТЯМ, а не сплошной лентой. Разделы берутся из
    `clinic_routes` — того же справочника, что и в интерфейсе, поэтому
    хирург видит своё, а ортопед своё. Одна лента на 40 находок не читается,
    это была прямая просьба с приёма.
  * ОСНОВАНИЕ у каждой находки: что измерено, сколько и при каком пороге.
    Процента уверенности здесь нет и не будет — вероятности у нас нет,
    находки считаются геометрией по маске. Врач должен видеть число, которое
    может проверить на срезе, а не выдуманные «87%».
  * РУЧНАЯ ПРОВЕРКА отдельным разделом: зубы, по которым измерение не
    получилось. Это честный отказ, и прятать его нельзя — иначе врач решит,
    что там всё в порядке.

СТАТУС ОТЧЁТА НА ПЕРВОЙ СТРАНИЦЕ, крупно. Пока набор находок не подписан
врачом, это черновик машины, и документ обязан говорить это сам, без
мелкого шрифта внизу.

Запуск:
    .venv/Scripts/python.exe research/scripts/cbct_report.py client5
    .venv/Scripts/python.exe research/scripts/cbct_report.py client5 --model sp --out отчёт.pdf
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbct_backend as cb  # noqa: E402
import cbct_plan  # noqa: E402
import clinic_routes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

PAGE_W, PAGE_H = 595.3, 841.9      # A4 в пунктах
MARGIN = 42

# Цвета важности. Держим их согласованными с интерфейсом: врач не должен
# заново учить, что значит красное, переходя с экрана на бумагу.
SEV_COLOR = {
    "alert": (0.72, 0.20, 0.16),
    "warning": (0.75, 0.52, 0.12),
    "info": (0.30, 0.45, 0.62),
}
SEV_LABEL = {"alert": "тревога", "warning": "внимание", "info": "к сведению"}

FDI_ROWS = [
    [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
]


def _fonts():
    """Кириллица в PDF: берём DejaVu из matplotlib, а не системный Arial.

    Шрифт лежит внутри окружения, поэтому отчёт соберётся одинаково на любой
    машине, где стоит наш `.venv` — включая сервер клиники без набора
    системных шрифтов.
    """
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    d = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Body", str(d / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(d / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(d / "DejaVuSansMono.ttf")))


class Report:
    """Рисуем страницу вручную: платипус тут мешает — много таблиц с
    переменной высотой и врезок, которые проще разместить по координатам."""

    def __init__(self, path: Path, case_name: str):
        from reportlab.pdfgen import canvas
        self.c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
        self.case_name = case_name
        self.y = PAGE_H - MARGIN
        self.page = 1

    # --- низкоуровневое ---
    def _space(self, h: float) -> None:
        if self.y - h < MARGIN + 26:
            self.new_page()

    def new_page(self) -> None:
        self._footer()
        self.c.showPage()
        self.page += 1
        self.y = PAGE_H - MARGIN

    def _footer(self) -> None:
        self.c.setFont("Body", 7.5)
        self.c.setFillColorRGB(0.55, 0.55, 0.55)
        self.c.drawString(MARGIN, MARGIN - 14,
                          f"{self.case_name} · страница {self.page}")
        self.c.drawRightString(PAGE_W - MARGIN, MARGIN - 14,
                               "Stomadent · черновик, требует проверки врачом")

    def text(self, s: str, size=9.5, bold=False, color=(0.1, 0.1, 0.1),
             indent=0, gap=3.2) -> None:
        self._space(size + gap)
        self.c.setFont("Body-Bold" if bold else "Body", size)
        self.c.setFillColorRGB(*color)
        self.c.drawString(MARGIN + indent, self.y - size, s)
        self.y -= size + gap

    def wrapped(self, s: str, size=9.5, indent=0, width=None,
                color=(0.25, 0.25, 0.25)) -> None:
        """Перенос по словам: длина строки считается по метрикам шрифта."""
        from reportlab.pdfbase.pdfmetrics import stringWidth
        width = width or (PAGE_W - 2 * MARGIN - indent)
        words, line = s.split(), ""
        for w in words:
            probe = (line + " " + w).strip()
            if stringWidth(probe, "Body", size) > width and line:
                self.text(line, size, indent=indent, color=color, gap=1.6)
                line = w
            else:
                line = probe
        if line:
            self.text(line, size, indent=indent, color=color, gap=1.6)

    def rule(self, gap=8) -> None:
        self._space(gap + 2)
        self.c.setStrokeColorRGB(0.85, 0.85, 0.85)
        self.c.setLineWidth(0.6)
        self.c.line(MARGIN, self.y, PAGE_W - MARGIN, self.y)
        self.y -= gap

    def image(self, png: bytes, max_w: float, caption: str = "") -> None:
        """Картинка по ширине колонки, пропорции сохраняем сами.

        Растянутый срез — это враньё о геометрии: врач меряет по нему глазом,
        и искажённый масштаб хуже отсутствующей картинки.
        """
        import io
        from reportlab.lib.utils import ImageReader
        img = ImageReader(io.BytesIO(png))
        iw, ih = img.getSize()
        w = min(max_w, float(iw))
        h = w * ih / iw
        # Картинка выше страницы не поместится ни при каком переносе.
        room = PAGE_H - 2 * MARGIN - 40
        if h > room:
            h, w = room, room * iw / ih
        self._space(h + (14 if caption else 6))
        self.c.drawImage(img, MARGIN, self.y - h, width=w, height=h,
                         preserveAspectRatio=True, mask="auto")
        self.y -= h + 3
        if caption:
            self.text(caption, 8, color=(0.45, 0.45, 0.45), gap=6)

    def heading(self, s: str, size=13, keep: float = 0) -> None:
        """`keep` — сколько места нужно ПОСЛЕ заголовка, чтобы он не оторвался.

        Без этого заголовок «Зуб 46» уезжал в самый низ страницы, а его срезы
        начинались на следующей — и над ними стояли картинки предыдущего зуба.
        Врач читает такую страницу как срезы не того зуба.
        """
        self._space(size + 16 + keep)
        self.y -= 8
        self.text(s, size, bold=True, color=(0.08, 0.08, 0.08), gap=6)


def _tooth_chart(rep: Report, present, findings) -> None:
    """Зубная формула: две дуги по 16 позиций, цвет — по худшей находке."""
    worst = {}
    order = {"info": 1, "warning": 2, "alert": 3}
    for f in findings:
        for t in (f.get("teeth") or ([f["tooth"]] if f.get("tooth") else [])):
            s = f.get("severity", "info")
            if order.get(s, 0) > order.get(worst.get(t, ""), 0):
                worst[t] = s
    cell, gap = 30.0, 2.4
    row_w = 16 * cell + 15 * gap
    x0 = (PAGE_W - row_w) / 2
    rep._space(2 * (cell + 14) + 16)
    for row in FDI_ROWS:
        x = x0
        for t in row:
            here = t in present
            # У отсутствующего зуба цвет находки НЕ показываем: находка
            # «зуб 18 отсутствует» красила клетку, и та читалась как
            # присутствующий зуб с проблемой — ровно наоборот смыслу.
            sev = worst.get(t) if here else None
            if sev:
                r, g, b = SEV_COLOR[sev]
                rep.c.setFillColorRGB(r, g, b)
            elif here:
                rep.c.setFillColorRGB(0.93, 0.94, 0.95)
            else:
                rep.c.setFillColorRGB(1, 1, 1)
            rep.c.setStrokeColorRGB(0.72, 0.72, 0.72)
            rep.c.setLineWidth(0.7)
            rep.c.roundRect(x, rep.y - cell, cell, cell, 3.5,
                            stroke=1, fill=1)
            if not here:
                # Отсутствующий зуб — перечёркнутая клетка. Пустая читалась бы
                # как «не смотрели», а это разные вещи.
                rep.c.setStrokeColorRGB(0.72, 0.72, 0.72)
                rep.c.line(x + 6, rep.y - cell + 6, x + cell - 6, rep.y - 6)
            rep.c.setFont("Body", 8.4)
            rep.c.setFillColorRGB(1, 1, 1) if sev else \
                rep.c.setFillColorRGB(0.25, 0.25, 0.25)
            rep.c.drawCentredString(x + cell / 2, rep.y - cell / 2 - 3, str(t))
            x += cell + gap
        rep.y -= cell + 8
    rep.y -= 4
    # легенда
    rep.c.setFont("Body", 8)
    lx = x0
    for sev, label in (("alert", "тревога"), ("warning", "внимание"),
                       ("info", "к сведению")):
        r, g, b = SEV_COLOR[sev]
        rep.c.setFillColorRGB(r, g, b)
        rep.c.roundRect(lx, rep.y - 8, 9, 9, 2, stroke=0, fill=1)
        rep.c.setFillColorRGB(0.35, 0.35, 0.35)
        rep.c.drawString(lx + 13, rep.y - 6.5, label)
        lx += 88
    rep.c.setFillColorRGB(0.35, 0.35, 0.35)
    rep.c.drawString(lx, rep.y - 6.5, "перечёркнут — отсутствует")
    rep.y -= 20


def _basis_line(f: dict) -> str:
    parts = []
    for b in (f.get("basis") or []):
        v, u = b.get("value"), b.get("unit") or ""
        if v is None:
            continue
        v = f"{v:g}" if isinstance(v, (int, float)) else str(v)
        parts.append(f"{b.get('label','')} {v} {u}".strip())
    return " · ".join(parts)


def build_report(case_id: str, model: str | None = None,
                 out: Path | None = None) -> Path:
    _fonts()
    c = cb.get_case(case_id)
    if c is None:
        raise SystemExit(f"нет кейса {case_id}")
    model = model or c.get("model") or cb.mask_model_for(c)
    if model is None:
        raise SystemExit(f"у кейса {case_id} нет готовой маски")

    data = cb.compute_findings(c, model) or {}
    findings = data.get("findings") or []
    quality = data.get("quality") or []
    present = set(data.get("teeth_present") or [])
    missing = data.get("teeth_missing") or []
    plan = cbct_plan.build_plan(data)
    # Подпись лежит в кейсе, а не в плане: у плана поле `review` — это список
    # позиций для ручной проверки, совсем про другое.
    review = c.get("review") or {}
    signed = review.get("signed") if isinstance(review, dict) else None

    out = Path(out) if out else (cb.CASES_DIR / case_id / f"отчёт_{model}.pdf")
    rep = Report(out, c.get("name") or case_id)

    # ---------- шапка ----------
    rep.text("Отчёт по данным конусно-лучевой томографии", 15.5, bold=True)
    rep.text(f"{c.get('name') or case_id} · модель «{data.get('model_label', model)}»"
             f" · {time.strftime('%d.%m.%Y')}", 9, color=(0.42, 0.42, 0.42))
    rep.y -= 4

    # Статус — крупно и первым, до всякого содержания.
    box_h = 30
    rep._space(box_h + 10)
    if signed:
        rep.c.setFillColorRGB(0.90, 0.95, 0.91)
        rep.c.setStrokeColorRGB(0.55, 0.72, 0.58)
    else:
        rep.c.setFillColorRGB(0.99, 0.95, 0.88)
        rep.c.setStrokeColorRGB(0.85, 0.68, 0.35)
    rep.c.setLineWidth(0.8)
    rep.c.roundRect(MARGIN, rep.y - box_h, PAGE_W - 2 * MARGIN, box_h, 4,
                    stroke=1, fill=1)
    rep.c.setFont("Body-Bold", 10)
    rep.c.setFillColorRGB(0.2, 0.2, 0.2)
    if signed:
        who = signed.get("by") or "врач"
        rep.c.drawString(MARGIN + 12, rep.y - 19,
                         f"Проверено и подписано: {who}")
    else:
        rep.c.drawString(MARGIN + 12, rep.y - 12,
                         "ЧЕРНОВИК. Находки посчитаны машиной и не подписаны врачом.")
        rep.c.setFont("Body", 8.4)
        rep.c.setFillColorRGB(0.4, 0.4, 0.4)
        rep.c.drawString(MARGIN + 12, rep.y - 23,
                         "Документ не является диагнозом и не может быть основанием "
                         "для лечения без осмотра.")
    rep.y -= box_h + 12

    # ---------- сводка ----------
    n_alert = sum(1 for f in findings if f.get("severity") == "alert")
    rep.text(f"Зубов в снимке: {len(present)}    "
             f"Отсутствует: {len(missing) if missing else 0}    "
             f"Находок: {len(findings)}    Из них тревог: {n_alert}",
             9.5, bold=True)
    rep.y -= 2

    # ---------- формула ----------
    rep.heading("Зубная формула")
    _tooth_chart(rep, present, findings)

    # ---------- находки по специальностям ----------
    # Панорама: врач читает ряд целиком и сверяет номера с тем, что видит на
    # приёме. Не строится (мало зубов для дуги) — раздел просто пропускаем,
    # пустая рамка хуже отсутствия.
    try:
        import cbct_panorama
        pano, _pm = cbct_panorama.build(c, model)
    except Exception:                                  # noqa: BLE001
        pano = None
    if pano:
        rep.heading("Панорама (развёртка по зубной дуге)")
        rep.image(pano, PAGE_W - 2 * MARGIN,
                  "Реконструкция из объёма, а не отдельный снимок ОПТГ.")

    rep.heading("Находки")
    if not findings:
        rep.wrapped("Находок нет. Это не то же самое, что «здоров»: "
                    "проверяется только то, что измеримо по снимку.")
    routes = {r["key"]: r for r in (data.get("routes") or [])}
    by_route = {}
    for f in findings:
        by_route.setdefault(f.get("route") or "other", []).append(f)
    for key in [r["key"] for r in (data.get("routes") or [])] + ["other"]:
        items = by_route.get(key)
        if not items:
            continue
        label = routes.get(key, {}).get("label", "Прочее")
        rep.text(label, 10.5, bold=True, color=(0.15, 0.15, 0.15), gap=5)
        for f in items:
            sev = f.get("severity", "info")
            teeth = f.get("teeth") or ([f["tooth"]] if f.get("tooth") else [])
            head = f.get("text") or f.get("code", "")
            rep._space(13)
            r, g, b = SEV_COLOR.get(sev, SEV_COLOR["info"])
            rep.c.setFillColorRGB(r, g, b)
            rep.c.circle(MARGIN + 5, rep.y - 5.5, 2.6, stroke=0, fill=1)
            rep.wrapped(head, 9.5, indent=14, color=(0.15, 0.15, 0.15))
            if teeth:
                rep.text("зубы: " + ", ".join(str(t) for t in teeth), 8.4,
                         indent=14, color=(0.45, 0.45, 0.45), gap=1.4)
            basis = _basis_line(f)
            if basis:
                rep.text("основание: " + basis, 8.2, indent=14,
                         color=(0.45, 0.45, 0.45), gap=1.4)
            if f.get("action"):
                rep.text("→ " + f["action"], 8.6, indent=14,
                         color=(0.30, 0.38, 0.48), gap=1.4)
            rep.y -= 3
        rep.y -= 3

    # ---------- план ----------
    rep.heading("План лечения — черновик")
    any_proc = False
    for st in (plan.get("stages") or []):
        procs = st.get("procedures") or []
        if not procs:
            continue
        any_proc = True
        rep.text(f"Этап {st['num']}. {st['label']}", 10.5, bold=True, gap=3)
        if st.get("note"):
            rep.text(st["note"], 8.4, indent=2, color=(0.5, 0.5, 0.5), gap=4)
        for p in procs:
            line = p["name"]
            if p.get("count", 1) > 1:
                line += f" ×{p['count']}"
            if p.get("teeth"):
                line += " · зубы " + ", ".join(str(t) for t in p["teeth"])
            rep.wrapped("• " + line, 9.5, indent=10, color=(0.15, 0.15, 0.15))
            if p.get("why"):
                rep.text(p["why"], 8.2, indent=20, color=(0.5, 0.5, 0.5), gap=1.4)
        rep.y -= 5
    if not any_proc:
        rep.wrapped("Позиций нет: находок, требующих вмешательства, не найдено.")

    # ---------- срезы зубов с находками ----------
    # Только зубы с тревогой или предупреждением: печатать срезы всех
    # тридцати двух — это десятки страниц, которые никто не смотрит.
    focus = []
    for f in findings:
        if f.get("severity") not in ("alert", "warning"):
            continue
        for t in (f.get("teeth") or ([f["tooth"]] if f.get("tooth") else [])):
            if t not in focus:
                focus.append(t)
    vol = mask = spacing = None
    if focus:
        try:
            import cbct_tooth
            vol, mask = cb.arrays_for(c, model)
            spacing = np.array(c["spacing"], float)
        except Exception:                              # noqa: BLE001
            focus = []
    for t in focus[:6]:          # больше шести — отчёт перестают читать
        shots = []
        for plane in ("md", "bl"):
            try:
                png, pmeta = cbct_tooth.render(vol, mask, spacing, t,
                                               plane, model)
            except Exception:                          # noqa: BLE001
                png = None
            if png:
                shots.append((png, pmeta))
        if not shots:
            continue
        # Держим заголовок вместе с первым срезом. Высоту считаем по реальным
        # пропорциям картинки, а не на глаз: первая прикидка в 250 пунктов
        # оказалась мала (срез 18x26 мм при ширине 235 пунктов занимает ~330),
        # и заголовок «Зуб 37» снова остался на странице один.
        first_png = shots[0][0]
        try:
            import io
            from reportlab.lib.utils import ImageReader
            iw, ih = ImageReader(io.BytesIO(first_png)).getSize()
            w0 = min((PAGE_W - 2 * MARGIN) * 0.46, float(iw))
            keep = w0 * ih / iw + 14
        except Exception:                              # noqa: BLE001
            keep = 340
        rep.heading(f"Зуб {t} — срезы вдоль собственной оси", size=11.5,
                    keep=keep)
        for png, pmeta in shots:
            rep.image(png, (PAGE_W - 2 * MARGIN) * 0.46,
                      f"{pmeta.get('title','')} · "
                      f"{pmeta.get('w_mm')}×{pmeta.get('h_mm')} мм")

    # ---------- ручная проверка ----------
    if quality:
        rep.heading("Требует ручной проверки")
        rep.wrapped("Здесь то, что измерить не удалось. Это отказ измерения, "
                    "а не признак нормы — по этим зубам снимок надо смотреть "
                    "глазами.", 8.8)
        rep.y -= 3
        for q in quality:
            rep.wrapped("• " + (q.get("text") or q.get("code", "")), 9,
                        indent=10)

    # ---------- оговорка ----------
    rep.heading("О чём этот отчёт не говорит")
    for line in (
        "Кариес и периапикальные очаги по КЛКТ не определяются — для них "
        "нужны прицельные снимки.",
        "Пороги измерений подобраны на ограниченной выборке и врачом не "
        "утверждены.",
        "Отсутствие находки означает, что измерение не сработало или "
        "признак не проверяется, а не что патологии нет.",
    ):
        rep.wrapped("• " + line, 8.8, indent=8)

    rep._footer()
    rep.c.save()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case_id")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    p = build_report(a.case_id, a.model, a.out)
    print(f"готово: {p}  ({p.stat().st_size/1024:.0f} КБ)")


if __name__ == "__main__":
    main()
