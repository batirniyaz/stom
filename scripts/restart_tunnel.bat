@echo off
rem Actina - restart the cloudflared service so a new hostname in config.yml
rem starts working.
rem
rem RUN AS ADMINISTRATOR: the service cannot be stopped without elevation,
rem and a non-elevated attempt fails with "Cannot open cloudflared service".
rem
rem ASCII ONLY - cmd.exe reads .bat in the OEM codepage; Russian text here
rem breaks command parsing itself (documented in CLAUDE.md).
rem
rem WHAT IT AFFECTS: one tunnel serves every hostname on this machine
rem (demo, annotate, studio, rail, sofa, actina), so the restart drops all of
rem them for a few seconds. Nothing else changes - the config file is only
rem read, never written.
rem
rem Everything is written to data\logs\restart_tunnel.log: an elevated console
rem window is invisible to tooling, which is why an earlier attempt looked
rem like "nothing happens".

setlocal
set "LOG=%~dp0..\data\logs\restart_tunnel.log"
set "CFD=C:\PROGRA~2\cloudflared\cloudflared.exe"
set "CFG=%USERPROFILE%\.cloudflared\config.yml"

echo ==== %DATE% %TIME% > "%LOG%"
net session >nul 2>&1
if errorlevel 1 (
    echo [!!] NOT ELEVATED - right-click this file, "Run as administrator" >> "%LOG%"
    echo [!!] Not elevated. Right-click this file -^> Run as administrator.
    pause
    exit /b 1
)

echo [1] validating ingress rules >> "%LOG%"
"%CFD%" --config "%CFG%" tunnel ingress validate >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [!!] config.yml is INVALID - service NOT restarted >> "%LOG%"
    echo [!!] config.yml is invalid. Nothing was restarted. See %LOG%
    pause
    exit /b 1
)

echo [2] restarting service >> "%LOG%"
sc stop cloudflared >> "%LOG%" 2>&1
timeout /t 4 >nul
sc start cloudflared >> "%LOG%" 2>&1
timeout /t 6 >nul

echo [3] service state >> "%LOG%"
sc query cloudflared >> "%LOG%" 2>&1
sc query cloudflared | find "RUNNING" >nul
if errorlevel 1 (
    echo [!!] service is NOT running >> "%LOG%"
    echo [!!] Service is not running. See %LOG%
    pause
    exit /b 1
)

echo [ok] cloudflared restarted >> "%LOG%"
echo [ok] Tunnel restarted. Check https://actina.aicenterkk.tech/cbct
pause
