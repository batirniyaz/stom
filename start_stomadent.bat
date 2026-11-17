@echo off
rem Stomadent launcher. Starts demo server and Cloudflare Tunnel.
rem
rem WHY THIS FILE IS ASCII-ONLY: cmd.exe reads .bat in the system OEM codepage,
rem not UTF-8. Russian comments here broke command parsing itself - the file
rem silently produced "'HONIOENCODING' is not recognized" and started nothing.
rem Same class of bug as the nnU-Net dataset.json issue documented in CLAUDE.md.
rem All logic lives in start_stomadent.ps1, which handles UTF-8 correctly.
rem
rem Usage:  start_stomadent.bat          - run and keep window open
rem         start_stomadent.bat /auto    - run silently (used by autostart task)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_stomadent.ps1" %*
if "%~1"=="/auto" goto :eof
pause
