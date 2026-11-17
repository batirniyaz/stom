@echo off
rem Actina - start the new interface service on port 7871.
rem
rem ASCII ONLY: cmd.exe reads .bat in the OEM codepage, and Russian text here
rem breaks command parsing itself (documented in CLAUDE.md). The Russian
rem explanation lives in start_actina.ps1 next to the logic.
rem
rem This is the EXPERIMENT copy of the project. The clinic demo runs from
rem D:\code\dbc\stomadent on port 7861 and is not touched by this script.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_actina.ps1" %*
