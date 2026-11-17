@echo off
rem Stomadent: install Windows services (tunnel + app servers at boot).
rem
rem RUN AS ADMINISTRATOR: right-click this file -> "Run as administrator".
rem Without elevation neither the service nor the boot task can be created.
rem
rem ASCII ONLY - cmd.exe reads .bat in the OEM codepage, and Russian comments
rem here break command parsing itself. All logic is in install_services.ps1,
rem which handles UTF-8 correctly. Same gotcha as start_stomadent.bat.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_services.ps1"
