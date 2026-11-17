@echo off
cd /d D:\code\dbc\stomadent
start "" http://localhost:7860
.venv\Scripts\python.exe research\scripts\app_ui.py --port 7860
