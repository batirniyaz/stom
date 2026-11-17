@echo off
cd /d D:\code\dbc\stomadent
start "Stomadent Monitor :8080" /min .venv\Scripts\python.exe research\scripts\monitor_server.py --port 8080
start "Stomadent Demo :7861" /min .venv\Scripts\python.exe research\scripts\demo_server.py --port 7861
start "Stomadent UI :7860" /min .venv\Scripts\python.exe research\scripts\app_ui.py --port 7860
start "Stomadent Annotate :7862" /min .venv\Scripts\python.exe research\scripts\annotate_server.py --port 7862
timeout /t 15 /nobreak > nul
start "" http://localhost:7861
