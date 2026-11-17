"""Заглушка на порт демо: «идут работы», пока GPU занят обучением.

Зачем. Демо-сервер держит в видеопамяти три модели nnU-Net, и вместе с
обучением они не помещаются: на 27.08 замерено 30.9 ГБ занято из 32.6, у
обучения оставалось 1.2 ГБ, и эпоха шла 35 минут вместо 17 — драйвер Windows
в такой ситуации не падает, а тихо сбрасывает память в оперативную. Поэтому на
время обучения демо выключается.

Без заглушки посетитель домена получает ошибку Cloudflare (502, «Bad
gateway») — это выглядит как поломка. Здесь отдаётся понятная страница с
кодом 503 и заголовком `Retry-After`: и человеку ясно, и поисковик не сочтёт
страницу удалённой.

Только stdlib: никакого torch, GPU не трогаем вовсе (как в monitor_server.py).

Запуск:
    .venv/Scripts/python.exe research/scripts/maintenance_server.py --port 7861
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stomadent AI — идут технические работы</title>
<style>
  body { margin:0; min-height:100vh; display:flex; align-items:center;
         justify-content:center; background:#070d15; color:#e6ecf5;
         font-family:"Segoe UI",system-ui,sans-serif; }
  .box { max-width:520px; padding:40px 32px; text-align:center; }
  .logo { width:56px; height:56px; border-radius:16px; margin:0 auto 22px;
          background:linear-gradient(135deg,#22d3ee,#3b82f6); display:flex;
          align-items:center; justify-content:center; font-size:26px; }
  h1 { font-size:22px; margin:0 0 12px; }
  p { color:#94a3b8; font-size:14.5px; line-height:1.6; margin:0 0 10px; }
  .accent { color:#22d3ee; }
</style></head>
<body><div class="box">
  <div class="logo">🦷</div>
  <h1>Идут технические работы</h1>
  <p>Обучаем новую модель сегментации — ту, что будет находить
     <span class="accent">и все зубы, и каналы одновременно</span>.</p>
  <p>Сервис вернётся, как только обучение закончится.
     Загруженные ранее исследования не потеряны.</p>
</div></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes) -> None:
        # 503, а не 200: страница временная, и поисковик не должен считать,
        # что сайт стал заглушкой навсегда.
        self.send_response(503)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", "86400")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(PAGE.encode("utf-8"))

    def do_POST(self) -> None:
        self._send(b'{"error": "service temporarily unavailable"}')

    def log_message(self, fmt, *args):
        pass          # в консоль не шумим: сервер живёт сутками


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"заглушка на http://127.0.0.1:{a.port} (Ctrl+C — остановить)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
