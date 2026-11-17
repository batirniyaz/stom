@echo off
:: Установка Cloudflare Tunnel как службы Windows.
:: ЗАПУСК: правый клик по файлу -> "Запуск от имени администратора"
"C:\Program Files (x86)\cloudflared\cloudflared.exe" service install
echo.
echo Если выше "success" — туннель будет стартовать сам при включении ПК.
pause
