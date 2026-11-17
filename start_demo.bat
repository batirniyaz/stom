@echo off
rem Stomadent AI — запуск демо-интерфейса (порт 7861)
cd /d "%~dp0"
start "" /min cmd /c "timeout /t 3 >nul & start http://localhost:7861"
.venv\Scripts\python.exe research\scripts\demo_server.py --port 7861
