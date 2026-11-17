# Actina — запуск сервиса нового интерфейса на порту 7871.
#
# Зачем отдельный сервис. Боевое демо клиники живёт в D:\code\dbc\stomadent
# на порту 7861, его ссылку уже дали стоматологу, и менять интерфейс под
# врачом на ходу нельзя. Новый вид показывается с этой рабочей копии по
# отдельному адресу actina.aicenterkk.tech, а откат сводится к «не открывать
# новый адрес».
#
# Данные у копии СВОИ: research_3d\web_cases пуст, в базе только
# демонстрационный пациент и фикстура. Настоящие снимки клиники на публичный
# адрес не уезжают. Веса моделей общие — research_3d\nnUNet_results это
# junction на боевой проект, поэтому сегментация работает та же.
#
# Ключ ИИ-чата не копируется в эту копию: он читается из боевого
# data\secrets.env и передаётся процессу только переменной окружения.
# В гит рабочей копии он таким образом попасть не может.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$live = 'D:\code\dbc\stomadent'
$port = 7871
$logs = Join-Path $root 'data\logs'
if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Force $logs | Out-Null }

# Проверяем ПОРТ, а не имя процесса: python.exe на этой машине крутится и по
# другим поводам (боевое демо, разметка, мониторинг). Проверка по имени уже
# однажды «нашла» чужой процесс и решила, что сервис запущен.
$busy = $null
try { $busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop } catch {}
if ($busy) {
  Write-Host "Actina уже слушает порт $port — ничего не делаю."
  Write-Host "Адрес: http://127.0.0.1:$port/cbct"
  exit 0
}

# Ключи из боевого secrets.env, если их нет в окружении.
$sec = Join-Path $live 'data\secrets.env'
if (Test-Path $sec) {
  foreach ($line in Get-Content $sec -Encoding UTF8) {
    if ($line -match '^\s*([A-Z0-9_]+)\s*=\s*(.+?)\s*$') {
      $name = $matches[1]; $val = $matches[2].Trim('"').Trim("'")
      if (-not [Environment]::GetEnvironmentVariable($name)) {
        Set-Item -Path ("env:" + $name) -Value $val
      }
    }
  }
}

# Консоль и служебные файлы на этой машине живут в cp1252, а проект в UTF-8:
# без этого русские строки в логе превращаются в мусор, а иногда роняют вывод.
$env:PYTHONIOENCODING = 'utf-8'

$py = Join-Path $root '.venv\Scripts\python.exe'
$srv = Join-Path $root 'research\scripts\demo_server.py'
$out = Join-Path $logs 'actina.log'
$err = Join-Path $logs 'actina.err.log'

Write-Host "Запускаю Actina на порту $port…"
Start-Process -FilePath $py -ArgumentList @($srv, '--port', "$port") `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput $out -RedirectStandardError $err

# Веса грузятся при старте — первый ответ приходит примерно через минуту.
for ($i = 0; $i -lt 90; $i++) {
  Start-Sleep -Seconds 2
  try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/cbct" -UseBasicParsing -TimeoutSec 4
    if ($r.StatusCode -eq 200) {
      Write-Host "Готово: http://127.0.0.1:$port/cbct"
      Write-Host "Снаружи: https://actina.aicenterkk.tech/cbct"
      exit 0
    }
  } catch {}
}
Write-Host "Не ответил за три минуты. Смотрите $err"
exit 1
