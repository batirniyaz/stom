@echo off
cd /d D:\code\dbc\stomadent
start "" http://localhost:8080
.venv\Scripts\python.exe research\scripts\monitor_server.py --port 8080
