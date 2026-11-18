@echo off
rem Stomadent: rebuild the cloudflared Windows service.
rem
rem RUN AS ADMINISTRATOR. Everything it does is written to
rem data\logs\fix_tunnel_service.log so the result can be read afterwards -
rem an elevated console window is invisible to tooling, and that is exactly
rem why the previous attempt looked like "nothing happens".
rem
rem WHY A SEPARATE SCRIPT: `cloudflared service install` registered the
rem service with no arguments (just the exe, no --config, no "tunnel run").
rem Such a service reports Running while the tunnel is down - outside you get
rem Cloudflare 1033/530. It then wedged in StopPending, and `sc stop` blocks
rem forever on it, which hung the installer. So: kill the service process by
rem PID first, then delete and recreate.
rem
rem ASCII ONLY - cmd.exe reads .bat in the OEM codepage; Russian text here
rem breaks command parsing itself (documented in CLAUDE.md).
rem
rem No quotes are needed inside binPath/TR: the short path C:\PROGRA~2 and the
rem project path D:\code\dbc\stomadent contain no spaces. Nested quotes are
rem parsed differently by cmd, sc and PowerShell - another silent-failure
rem source we already hit.

setlocal
set "LOG=%~dp0..\data\logs\fix_tunnel_service.log"
set "CFD=C:\PROGRA~2\cloudflared\cloudflared.exe"
set "CFG=%USERPROFILE%\.cloudflared\config.yml"
set "BAT=%~dp0start_stomadent.bat"

echo ==== %DATE% %TIME% > "%LOG%"
net session >nul 2>&1
if errorlevel 1 (
    echo [!!] NOT ELEVATED - right-click this file, "Run as administrator" >> "%LOG%"
    echo [!!] Not elevated. Right-click this file -^> Run as administrator.
    pause
    exit /b 1
)

echo [1] current service state >> "%LOG%"
sc queryex cloudflared >> "%LOG%" 2>&1

echo [2] killing the service process by PID >> "%LOG%"
for /f "tokens=3" %%p in ('sc queryex cloudflared ^| findstr /i "  PID"') do (
    echo     PID %%p >> "%LOG%"
    taskkill /F /PID %%p >> "%LOG%" 2>&1
)

echo [3] delete service >> "%LOG%"
sc delete cloudflared >> "%LOG%" 2>&1
timeout /t 4 /nobreak >nul

echo [4] create service with explicit arguments >> "%LOG%"
sc create cloudflared binPath= "%CFD% --config %CFG% tunnel run" start= auto DisplayName= "Cloudflare Tunnel Stomadent" >> "%LOG%" 2>&1
sc failure cloudflared reset= 86400 actions= restart/5000/restart/15000/restart/60000 >> "%LOG%" 2>&1

echo [5] start service >> "%LOG%"
sc start cloudflared >> "%LOG%" 2>&1
timeout /t 12 /nobreak >nul
sc qc cloudflared >> "%LOG%" 2>&1
sc query cloudflared >> "%LOG%" 2>&1

echo [6] boot task for the app servers >> "%LOG%"
schtasks /Create /TN Stomadent /TR "%BAT% /auto" /SC ONSTART /RU SYSTEM /RL HIGHEST /DELAY 0000:30 /F >> "%LOG%" 2>&1
schtasks /Query /TN Stomadent /FO LIST >> "%LOG%" 2>&1

echo [7] outside check >> "%LOG%"
curl -s -o nul -m 20 -w "demo.aicenterkk.tech -> %%{http_code}" https://demo.aicenterkk.tech/ >> "%LOG%" 2>&1
echo. >> "%LOG%"
echo ==== done %TIME% >> "%LOG%"

type "%LOG%"
echo.
echo Log: %LOG%
pause
