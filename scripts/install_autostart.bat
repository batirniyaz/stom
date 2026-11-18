@echo off
rem Registers Stomadent autostart via the user Startup folder.
rem
rem NO ADMIN RIGHTS NEEDED. Task Scheduler (schtasks /sc onlogon) would be nicer
rem - it has a built-in delayed start - but on this machine it returns
rem "Access is denied" without elevation. The Startup folder needs no rights;
rem the network delay is handled inside start_stomadent.ps1 instead.
rem
rem ASCII ONLY - see the comment in start_stomadent.bat.

set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "LINK=%STARTUP%\Stomadent.cmd"

echo Installing autostart...
echo   from: %~dp0start_stomadent.bat
echo   to:   %LINK%
echo.

rem A tiny .cmd wrapper instead of a shortcut: no COM, no PowerShell needed,
rem and it is plain text you can read and delete by hand.
> "%LINK%" echo @echo off
>> "%LINK%" echo rem Autostart for Stomadent. Delete this file to disable.
>> "%LINK%" echo call "%~dp0start_stomadent.bat" /auto

if exist "%LINK%" (
    echo [OK] Done. Demo server and tunnel will start automatically
    echo      about a minute after you log in.
    echo.
    echo   disable: delete "%LINK%"
) else (
    echo [!!] Failed to write "%LINK%"
)

echo.
pause
