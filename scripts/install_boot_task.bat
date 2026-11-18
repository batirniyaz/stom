@echo off
rem Stomadent: create the boot task that starts the app servers before login.
rem
rem RUN AS ADMINISTRATOR. Output goes to data\logs\boot_task.log as well as to
rem the window: an elevated console is invisible to tooling, and that is
rem exactly why an earlier attempt looked like "nothing happens".
rem
rem The tunnel is already a Windows service; this is the other half - without
rem it the tunnel comes up into nothing and the clinic sees 502 until someone
rem logs in.
rem
rem SYSTEM, not the user account: a task bound to a user only runs after that
rem user logs on, which is the very problem being solved. 30 s delay because
rem the network is not up the instant Windows boots.
rem
rem ASCII ONLY - cmd.exe reads .bat in the OEM codepage (see CLAUDE.md).
rem No nested quotes needed: D:\code\dbc\stomadent has no spaces.

setlocal
set "LOG=%~dp0..\data\logs\boot_task.log"
set "BAT=%~dp0start_stomadent.bat"

echo ==== %DATE% %TIME% > "%LOG%"
net session >nul 2>&1
if errorlevel 1 (
    echo [!!] NOT ELEVATED >> "%LOG%"
    echo [!!] Not elevated. Right-click this file -^> Run as administrator.
    pause
    exit /b 1
)

schtasks /Create /TN Stomadent /TR "%BAT% /auto" /SC ONSTART /RU SYSTEM /RL HIGHEST /DELAY 0000:30 /F >> "%LOG%" 2>&1
schtasks /Query /TN Stomadent /FO LIST >> "%LOG%" 2>&1

type "%LOG%"
echo.
echo Log: %LOG%
timeout /t 20
