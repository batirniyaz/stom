@echo off
cd /d "%~dp0.."
start "" http://localhost:8080
.venv\Scripts\python.exe research\scripts\monitor_server.py --port 8080
