# Stomadent: службы Windows — туннель и сервисы поднимаются ДО входа в систему.
#
# Запуск: правый клик по install_services.bat -> «Запуск от имени
# администратора». Без прав администратора ни служба, ни задача при загрузке
# не создаются — это ограничение Windows, а не наше.
#
# ЗАЧЕМ. Автозапуск из папки «Автозагрузка» срабатывает только ПОСЛЕ входа в
# учётную запись: пока никто не залогинился, клиника видит 530.
#
# СТАВИТСЯ ДВЕ ВЕЩИ:
#   1. cloudflared как служба Windows — туннель при включении ПК.
#   2. Задача планировщика «Stomadent» от имени SYSTEM при загрузке — демо,
#      разметка панорам и разметка каналов. Без неё туннель поднимется в
#      пустоту: снаружи вместо 530 станет 502.
#
# ТРИ ГРАБЛИ, НА КОТОРЫХ ЭТОТ СКРИПТ УЖЕ СПОТКНУЛСЯ:
#
# 1. СЛУЖБА БЕЗ АРГУМЕНТОВ. `cloudflared service install` зарегистрировал
#    службу командой «cloudflared.exe» — без `--config` и без `tunnel run`.
#    Служба числится Running, туннель не поднят, снаружи 1033/530, лог пуст.
#    Поэтому команда службы задаётся ЯВНО и сверяется при каждом запуске:
#    «служба есть» — ещё не значит «служба работает».
# 2. ЗАВИСАНИЕ НА `sc stop`. Та самая служба без аргументов не отвечает на
#    команду остановки и залипает в StopPending, а скрипт вместе с ней.
#    Теперь остановка с таймаутом, после него процесс снимается принудительно.
# 3. `$ErrorActionPreference = "Stop"` + вывод sc.exe в stderr роняет скрипт
#    посреди работы. Проверяем КОД ВОЗВРАТА, а не полагаемся на исключения.
#
# Путь к cloudflared берётся коротким (PROGRA~2): иначе внутри строки
# binPath= нужны вложенные кавычки, которые sc.exe и PowerShell понимают
# по-разному — ещё один источник молчаливых сбоев.
#
# ЧУЖОЙ ТУННЕЛЬ (`gamerly.yml`, другой проект) не трогаем: он запускается сам.

$ErrorActionPreference = "Continue"
$here  = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfdS  = "C:\PROGRA~2\cloudflared\cloudflared.exe"
$cfd   = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$cfg   = Join-Path $env:USERPROFILE ".cloudflared\config.yml"
$task  = "Stomadent"
$site  = "https://demo.aicenterkk.tech"

function Is-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Say($t) { Write-Host $t }

Say "[Stomadent] настройка служб"
Say ""
if (-not (Is-Admin)) {
    Say "[!!] Нужны права администратора: правый клик по install_services.bat"
    Say "     -> «Запуск от имени администратора»."
    Read-Host "Enter — закрыть"; exit 1
}
if (-not (Test-Path $cfd)) { Say "[!!] Нет cloudflared: $cfd"; Read-Host; exit 1 }
if (-not (Test-Path $cfg)) { Say "[!!] Нет конфигурации туннеля: $cfg"; Read-Host; exit 1 }

# ---------- 1. Служба туннеля ----------
$binPath = "$cfdS --config $cfg tunnel run"
$svc = Get-Service -Name cloudflared -ErrorAction SilentlyContinue

if ($svc) {
    Say "[ .] Останавливаю службу (до 20 секунд)"
    & sc.exe stop cloudflared | Out-Null
    $t0 = Get-Date
    while (((Get-Service cloudflared).Status -ne "Stopped") -and
           ((Get-Date) - $t0).TotalSeconds -lt 20) { Start-Sleep -Seconds 2 }
    if ((Get-Service cloudflared).Status -ne "Stopped") {
        Say "[ ~] Служба залипла в $((Get-Service cloudflared).Status) — снимаю процесс"
        Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" | ForEach-Object {
            $o = Invoke-CimMethod -InputObject $_ -MethodName GetOwner
            # чужой туннель запущен от пользователя — его не трогаем
            if ($o.User -ne $env:USERNAME) {
                Say "     снимаю PID $($_.ProcessId) (владелец $($o.User))"
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Seconds 3
    }
    if ((Get-Service cloudflared -ErrorAction SilentlyContinue).Status -ne "Stopped") {
        # Залипшую службу правкой команды не вылечить: sc config на
        # StopPending либо отказывает, либо применяется к мёртвой записи.
        # Удаляем и создаём заново — запись исчезает, как только отпустит
        # последний дескриптор.
        Say "[ ~] Служба всё ещё не остановлена — удаляю и создам заново"
        & sc.exe delete cloudflared | Out-Null
        $t0 = Get-Date
        while ((Get-Service cloudflared -ErrorAction SilentlyContinue) -and
               ((Get-Date) - $t0).TotalSeconds -lt 20) { Start-Sleep -Seconds 2 }
    }
    if (Get-Service cloudflared -ErrorAction SilentlyContinue) {
        Say "[ +] Задаю команду службы"
        & sc.exe config cloudflared binPath= $binPath start= auto | Out-Null
        if ($LASTEXITCODE -ne 0) { Say "[!!] sc config вернул $LASTEXITCODE" }
    } else {
        Say "[ +] Создаю службу заново"
        & sc.exe create cloudflared binPath= $binPath start= auto DisplayName= "Cloudflare Tunnel (Stomadent)" | Out-Null
        if ($LASTEXITCODE -ne 0) { Say "[!!] sc create вернул $LASTEXITCODE" }
    }
} else {
    Say "[ +] Создаю службу cloudflared"
    & sc.exe create cloudflared binPath= $binPath start= auto DisplayName= "Cloudflare Tunnel (Stomadent)" | Out-Null
    if ($LASTEXITCODE -ne 0) { Say "[!!] sc create вернул $LASTEXITCODE" }
}

# Туннель рвётся вместе с интернетом; без автоперезапуска он не вернётся до
# следующей загрузки.
& sc.exe failure cloudflared reset= 86400 actions= restart/5000/restart/15000/restart/60000 | Out-Null

Say "[ .] Запускаю службу"
& sc.exe start cloudflared | Out-Null
Start-Sleep -Seconds 8
$svc = Get-Service cloudflared
Say ("[ i] Служба: {0}" -f $svc.Status)
Say ("[ i] Команда: {0}" -f (Get-CimInstance Win32_Service -Filter "Name='cloudflared'").PathName)

# ---------- 2. Сервисы при загрузке ----------
Say ""
$bat = Join-Path $here "start_stomadent.bat"
if (-not (Test-Path $bat)) {
    Say "[!!] Нет $bat"
} else {
    $out = & schtasks /Create /TN $task /TR "`"$bat`" /auto" /SC ONSTART /RU SYSTEM /RL HIGHEST /DELAY 0000:30 /F 2>&1
    if ($LASTEXITCODE -eq 0) {
        Say "[OK] Задача «$task»: старт при загрузке, от SYSTEM, задержка 30 секунд"
    } else {
        Say "[!!] Задача не создалась (код $LASTEXITCODE): $out"
    }
}

# ---------- 3. Проверка СНАРУЖИ ----------
# «Running» ничего не доказывает — ровно на этом мы и обожглись дважды.
Say ""
Say "[..] Проверяю $site (до минуты)"
$code = 0
foreach ($i in 1..12) {
    Start-Sleep -Seconds 5
    try {
        $code = [int](Invoke-WebRequest -Uri $site -TimeoutSec 10 -UseBasicParsing).StatusCode
    } catch {
        $code = 0
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
    }
    if ($code -eq 200) { break }
}
Say ""
if ($code -eq 200) {
    Say "[OK] Снаружи 200 — туннель поднят службой."
} else {
    Say "[!!] Снаружи код $code — служба запущена, но туннель не подключился."
    Say "     Журнал: Просмотр событий -> Журналы Windows -> Приложение, источник cloudflared."
}
Say ""
Say "Настоящая проверка: перезагрузить компьютер, НЕ входить в учётную запись"
Say "и открыть $site с телефона. Если войти — сработает автозапуск из"
Say "«Автозагрузки» и замаскирует отказ службы."
Say ""
Read-Host "Enter — закрыть"
