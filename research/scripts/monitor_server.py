"""Веб-мониторинг обучения: логи, метрики из results.csv, состояние GPU.

Запуск:
    py research/scripts/monitor_server.py [--port 8080]

Открыть: http://localhost:8080
Зависимости: только stdlib (nvidia-smi для вкладки GPU).
"""

import argparse
import csv
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
RUNS = OUTPUT / "runs"

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Stomadent — мониторинг обучения</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #1e1e1e; color: #ddd; }
  header { padding: 10px 20px; background: #252526; font-size: 18px; }
  nav { display: flex; gap: 4px; background: #2d2d30; padding: 0 20px; }
  nav button { background: none; border: none; color: #aaa; padding: 12px 18px; cursor: pointer; font-size: 15px; }
  nav button.active { color: #fff; border-bottom: 2px solid #4da3ff; }
  main { padding: 16px 20px; }
  pre { background: #111; padding: 12px; overflow: auto; max-height: 75vh; font-size: 12px; white-space: pre-wrap; }
  select { background: #333; color: #ddd; border: 1px solid #555; padding: 4px 8px; margin-bottom: 10px; }
  table { border-collapse: collapse; margin-top: 14px; font-size: 13px; }
  th, td { border: 1px solid #444; padding: 4px 10px; text-align: right; }
  th { background: #2d2d30; }
  .chart-box { background: #252526; padding: 10px; margin-bottom: 14px; max-width: 1100px; }
  .hint { color: #888; font-size: 12px; }
</style>
</head>
<body>
<header>Stomadent — мониторинг обучения</header>
<nav>
  <button data-tab="metrics" class="active">Метрики</button>
  <button data-tab="logs">Логи</button>
  <button data-tab="gpu">GPU</button>
</nav>
<main>
  <section id="tab-metrics">
    <select id="run-select"></select>
    <span class="hint" id="metrics-updated"></span>
    <div class="chart-box"><canvas id="loss-chart"></canvas></div>
    <div class="chart-box"><canvas id="map-chart"></canvas></div>
    <div id="metrics-table"></div>
  </section>
  <section id="tab-logs" style="display:none">
    <select id="log-select"></select>
    <span class="hint"> обновление каждые 3 с</span>
    <pre id="log-view"></pre>
  </section>
  <section id="tab-gpu" style="display:none">
    <span class="hint">обновление каждые 5 с</span>
    <pre id="gpu-view"></pre>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  ['metrics','logs','gpu'].forEach(t => $('tab-'+t).style.display = t === b.dataset.tab ? '' : 'none');
});

let lossChart, mapChart;
function mkChart(ctx, label, datasets) {
  return new Chart(ctx, { type: 'line',
    data: { labels: [], datasets },
    options: { animation: false, responsive: true,
      scales: { x: { ticks: { color: '#aaa' } }, y: { ticks: { color: '#aaa' } } },
      plugins: { legend: { labels: { color: '#ddd' } }, title: { display: true, text: label, color: '#fff' } } } });
}
function ds(label, color) { return { label, data: [], borderColor: color, pointRadius: 0, borderWidth: 1.5, tension: 0.2 }; }

lossChart = mkChart($('loss-chart'), 'Loss', [ds('train/box','#4da3ff'), ds('train/cls','#ff9f40'), ds('val/box','#4dff88'), ds('val/cls','#ff4d6d')]);
mapChart = mkChart($('map-chart'), 'mAP', [ds('mAP50','#c580ff'), ds('mAP50-95','#ffd24d')]);

async function loadRuns() {
  const r = await fetch('/api/runs'); const data = await r.json();
  const sel = $('run-select'); const cur = sel.value;
  sel.innerHTML = Object.keys(data).map(k => `<option>${k}</option>`).join('');
  if (cur && data[cur]) sel.value = cur;
  renderRun(data[sel.value]);
  sel.onchange = () => renderRun(data[sel.value]);
}
function col(rows, name) { return rows.map(r => parseFloat(r[name])); }
function renderRun(rows) {
  if (!rows || !rows.length) return;
  const epochs = rows.map(r => r.epoch);
  const set = (chart, cols) => { chart.data.labels = epochs;
    chart.data.datasets.forEach(d => { const key = Object.keys(rows[0]).find(k => k.includes(cols[d.label]));
      d.data = key ? col(rows, key) : []; }); chart.update(); };
  set(lossChart, {'train/box':'train/box_loss','train/cls':'train/cls_loss','val/box':'val/box_loss','val/cls':'val/cls_loss'});
  set(mapChart, {'mAP50':'metrics/mAP50(B)','mAP50-95':'metrics/mAP50-95(B)'});
  const last = rows[rows.length-1], keys = Object.keys(last);
  $('metrics-table').innerHTML = `<table><tr>${keys.map(k=>`<th>${k}</th>`).join('')}</tr>
    <tr>${keys.map(k=>`<td>${last[k]}</td>`).join('')}</tr></table>
    <div class="hint">последняя эпоха: ${last.epoch} / всего строк: ${rows.length}</div>`;
  $('metrics-updated').textContent = ' обновлено: ' + new Date().toLocaleTimeString();
}

async function loadLogs() {
  const r = await fetch('/api/logs'); const data = await r.json();
  const sel = $('log-select'); const cur = sel.value;
  sel.innerHTML = Object.keys(data).map(k => `<option>${k}</option>`).join('');
  if (cur && data[cur]) sel.value = cur;
  const txt = (data[sel.value] || '').replace(/\\r/g, '\\n');
  const view = $('log-view');
  const stick = view.scrollTop + view.clientHeight >= view.scrollHeight - 40;
  view.textContent = txt.split('\\n').slice(-400).join('\\n');
  if (stick) view.scrollTop = view.scrollHeight;
  sel.onchange = loadLogs;
}

async function loadGpu() { const r = await fetch('/api/gpu'); $('gpu-view').textContent = await r.text(); }

loadRuns(); loadLogs(); loadGpu();
setInterval(loadRuns, 5000); setInterval(loadLogs, 3000); setInterval(loadGpu, 5000);
</script>
</body>
</html>"""


def tail_text(path: Path, nbytes: int = 120_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError as e:
        return f"<ошибка чтения: {e}>"


def get_logs() -> dict:
    logs = {}
    for p in sorted(OUTPUT.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True):
        logs[p.name] = tail_text(p)
    return logs or {"<нет .log файлов в research/output>": ""}


def get_runs() -> dict:
    runs = {}
    for csv_path in sorted(RUNS.glob("*/results.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
            if rows:
                runs[csv_path.parent.name] = rows
        except OSError:
            continue
    # nnU-Net: парсим training_log_*.txt из research_3d/nnUNet_results
    nnunet_results = ROOT.parent / "research_3d" / "nnUNet_results"
    for log in sorted(nnunet_results.glob("**/fold_*/training_log_*.txt"),
                      key=lambda x: x.stat().st_mtime, reverse=True):
        rows = []
        cur: dict | None = None
        try:
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if ": Epoch " in line:
                    if cur:
                        rows.append(cur)
                    cur = {"epoch": line.rsplit("Epoch", 1)[1].strip()}
                elif cur is None:
                    continue
                elif ": train_loss " in line:
                    cur["train/box_loss"] = line.rsplit(None, 1)[1]
                elif ": val_loss " in line:
                    cur["val/box_loss"] = line.rsplit(None, 1)[1]
                elif ": Pseudo dice " in line:
                    import ast
                    try:
                        dice = ast.literal_eval(line.split("Pseudo dice", 1)[1].strip())
                        cur["metrics/mAP50(B)"] = f"{sum(dice) / max(len(dice), 1):.4f}"
                    except (ValueError, SyntaxError):
                        pass
            if cur:
                rows.append(cur)
        except OSError:
            continue
        if rows:
            # имя рана: Dataset112_ToothFairy2 / fold_0
            ds = next((p for p in log.parts if p.startswith("Dataset")), log.parent.parent.name)
            runs[f"nnUNet {ds} {log.parent.name}"] = rows
    return runs or {"<пока нет запусков>": []}


def get_gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout or out.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"nvidia-smi недоступен: {e}"


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, ctype: str = "text/html; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/":
            self._send(PAGE)
        elif route == "/api/logs":
            self._send(json.dumps(get_logs(), ensure_ascii=False), "application/json; charset=utf-8")
        elif route == "/api/runs":
            self._send(json.dumps(get_runs(), ensure_ascii=False), "application/json; charset=utf-8")
        elif route == "/api/gpu":
            self._send(get_gpu(), "text/plain; charset=utf-8")
        else:
            self.send_error(404)

    def log_message(self, *args) -> None:  # тишина в консоли
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Мониторинг: http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
