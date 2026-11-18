# Поднимает Stomadent целиком: демо-сервер, сервер разметки и Cloudflare Tunnel.
#
# ЗАЧЕМ. После перезагрузки машины ничего не поднимается само: служба туннеля
# не установлена, а демо стартует только вручную. Врач в этот момент видит
# недоступный сайт и сделать ничего не может — поэтому запуск должен быть
# одним действием и вешаться в автозапуск (install_autostart.bat).
#
# ПОЧЕМУ POWERSHELL, А НЕ .BAT. Первая версия была батником с русскими
# комментариями и не работала вовсе: cmd.exe читает .bat в системной кодовой
# странице, кириллица ломает разбор САМИХ КОМАНД — файл выдавал
# «'HONIOENCODING' is not recognized» и не запускал ничего, молча. Тот же класс
# ошибки, что с dataset.json у nnU-Net. PowerShell читает UTF-8 нормально.
#
# Запускать МОЖНО ПОВТОРНО: работающее не дублируется. Второй демо занял бы
# порт и оставил мёртвый процесс, второй cloudflared поднял бы конкурирующие
# соединения к тому же туннелю.

param([switch]$auto)

# В автозапуске ждём, пока поднимется сеть. Сразу после входа в систему её ещё
# нет, и туннель тратит первую минуту на переподключения. Планировщик задач
# умеет отложенный старт сам, но его регистрация требует прав администратора,
# поэтому автозапуск сделан через папку «Автозагрузка», а задержка — здесь.
if ($auto) { Start-Sleep -Seconds 45 }

$ErrorActionPreference = "Stop"
# Скрипты лежат в scripts\, а работают с корнем проекта: .venv,
# data\logs и research\scripts\ — на уровень выше.
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py   = Join-Path $root ".venv\Scripts\python.exe"
$cfd  = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$logs = Join-Path $root "data\logs"
$port = 7861
$annotatePort = 7862
$canalPort = 7863

if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Path $logs -Force | Out-Null }

function Test-Port([int]$p = $port) {
    $null -ne (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
}

Write-Host "[Stomadent] $root"
Write-Host ""

# ---------- демо-сервер ----------
# Проверяем ПОРТ, а не имя процесса: python.exe на машине крутится и по другим
# поводам (обучение, скрипты), а занят ли порт — вопрос однозначный.
if (Test-Port) {
    Write-Host "[ =] Демо уже слушает :$port"
} elseif (-not (Test-Path $py)) {
    Write-Host "[!!] Нет интерпретатора: $py"
} else {
    Write-Host "[ +] Запускаю демо-сервер"
    # PYTHONIOENCODING обязателен: консоль здесь не в UTF-8, а весь вывод
    # проекта русский — без него сервер падает на первой строке лога.
    $env:PYTHONIOENCODING = "utf-8"
    $args = @("-u", (Join-Path $root "research\scripts\demo_server.py"), "--port", "$port")
    Start-Process -FilePath $py -ArgumentList $args `
        -RedirectStandardOutput (Join-Path $logs "demo.log") `
        -RedirectStandardError  (Join-Path $logs "demo.err.log") `
        -WindowStyle Hidden
}

# ---------- сервер разметки ----------
# Туннель отдаёт annotate.aicenterkk.tech на :7862, а сам сервер раньше
# поднимался только руками: после перезагрузки врачи-разметчики видели
# ошибку туннеля. При старте он же преразмечает снимки, пришедшие из карточек
# пациентов, пока он лежал (annotation_feed).
if (Test-Port $annotatePort) {
    Write-Host "[ =] Разметка уже слушает :$annotatePort"
} elseif (Test-Path $py) {
    Write-Host "[ +] Запускаю сервер разметки"
    $env:PYTHONIOENCODING = "utf-8"
    $aargs = @("-u", (Join-Path $root "research\scripts\annotate_server.py"), "--port", "$annotatePort")
    Start-Process -FilePath $py -ArgumentList $aargs `
        -RedirectStandardOutput (Join-Path $logs "annotate.log") `
        -RedirectStandardError  (Join-Path $logs "annotate.err.log") `
        -WindowStyle Hidden
}

# ---------- разметка каналов на КЛКТ ----------
# Отдельный сервис, а не часть демо: он не держит ни одной модели в памяти
# (вырезы и черновики посчитаны заранее), поэтому поднимается мгновенно и
# перезапускается, не трогая демо с его минутой прогрева.
if (Test-Port $canalPort) {
    Write-Host "[ =] Разметка каналов уже слушает :$canalPort"
} elseif (Test-Path $py) {
    Write-Host "[ +] Запускаю разметку каналов"
    $env:PYTHONIOENCODING = "utf-8"
    $cargs = @("-u", (Join-Path $root "research\scripts\canal_annotate_server.py"), "--port", "$canalPort")
    Start-Process -FilePath $py -ArgumentList $cargs `
        -RedirectStandardOutput (Join-Path $logs "canal_annotate.log") `
        -RedirectStandardError  (Join-Path $logs "canal_annotate.err.log") `
        -WindowStyle Hidden
}

# ---------- Cloudflare Tunnel ----------
# Проверяем НАШ туннель по файлу конфигурации, а не наличие процесса
# cloudflared. На этой машине живёт ещё один туннель (gamerly.yml, другой
# проект), и проверка по имени процесса видела ЕГО: скрипт рапортовал
# «туннель уже запущен», наш не стартовал вовсе, а снаружи оба адреса
# отдавали 530. Диагноз занял время именно потому, что сбой был молчаливым.
# Та же ошибка, что уже исправлена выше у демо: там проверяется ПОРТ, а не
# имя процесса.
$cfdConfig = Join-Path $env:USERPROFILE ".cloudflared\config.yml"
$mine = Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$cfdConfig*" -or $_.CommandLine -notlike "*--config*" }
if (Get-Service -Name cloudflared -ErrorAction SilentlyContinue) {
    # Служба стартует до входа в систему и сама себя перезапускает — если она
    # есть, своим процессом туннель больше не поднимаем.
    Write-Host "[ =] Туннель поднимает служба Windows"
} elseif ($mine) {
    Write-Host "[ =] Туннель Stomadent уже запущен"
} elseif (-not (Test-Path $cfd)) {
    Write-Host "[!!] Нет cloudflared: $cfd"
} else {
    Write-Host "[ +] Запускаю туннель"
    Start-Process -FilePath $cfd -ArgumentList @("--config", $cfdConfig, "tunnel", "run") `
        -RedirectStandardOutput (Join-Path $logs "tunnel.log") `
        -RedirectStandardError  (Join-Path $logs "tunnel.err.log") `
        -WindowStyle Hidden
}

# ---------- ожидание ----------
# Демо поднимает веса моделей около минуты — проверять раньше бессмысленно,
# получим ложное «не отвечает».
Write-Host ""
Write-Host "[..] Жду готовности демо (до 3 минут)"
$ok = $false
foreach ($i in 1..36) {
    Start-Sleep -Seconds 5
    if (Test-Port) { $ok = $true; break }
}

Write-Host ""
if ($ok) {
    Write-Host "[OK] Локально: http://127.0.0.1:$port"
    Write-Host "[OK] Снаружи:  https://demo.aicenterkk.tech"
    Write-Host "[OK] Разметка: https://annotate.aicenterkk.tech"
} else {
    Write-Host "[!!] Демо не поднялось. Смотри $($logs)\demo.log и demo.err.log"
}
Write-Host ""
Write-Host "Логи: $logs"
