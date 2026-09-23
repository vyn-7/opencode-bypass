# Manual run/start/stop/restart/status/logs for the opencode free-tier proxy (Windows).
# Usage:
#   .\run.ps1                     foreground (Ctrl+C to stop)
#   .\run.ps1 start [-Port 18788] [extra opencode_proxy.py args...]
#   .\run.ps1 stop|status|restart|logs
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("run", "start", "stop", "restart", "status", "logs")]
    [string]$Command = "run",
    [int]$Port = 18788,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Pythonw = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
$ProxyScript = Join-Path $PSScriptRoot "opencode_proxy.py"
$PidFile = Join-Path $env:LOCALAPPDATA "opencode-proxy.pid"
$LogPath = Join-Path $PSScriptRoot "opencode-proxy.log"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Creating venv (.venv)..."
    $boot = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
    & $boot -3 -m venv .venv
    & $Python -m pip install -q -r requirements.txt
}

function Get-ProxyPid {
    if (Test-Path -LiteralPath $PidFile) {
        $raw = (Get-Content -LiteralPath $PidFile -Raw).Trim()
        if ($raw -match '^\d+$') {
            $procId = [int]$raw
            if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
                return $procId
            }
        }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }
    return $null
}

function Stop-Proxy {
    $procId = Get-ProxyPid
    if ($null -eq $procId) {
        Write-Host "not running"
        return
    }
    # Kill the whole tree so the managed serve child dies too.
    & taskkill /PID $procId /T /F 2>$null | Out-Null
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "stopped (pid $procId)"
}

switch ($Command) {
    "run" {
        $args = @($ProxyScript, "--port", $Port) + $Rest
        & $Python @args
    }
    "start" {
        $procId = Get-ProxyPid
        if ($procId) {
            Write-Host "Already running (pid $procId)"
            exit 0
        }
        $argList = @($ProxyScript, "--port", $Port) + $Rest
        $p = Start-Process -FilePath $Pythonw `
            -ArgumentList $argList `
            -WorkingDirectory $PSScriptRoot `
            -RedirectStandardOutput "$LogPath.out" `
            -RedirectStandardError "$LogPath.err" `
            -WindowStyle Hidden `
            -PassThru
        Set-Content -LiteralPath $PidFile -Value $p.Id -Encoding ASCII
        Write-Host "Started (pid $($p.Id), log $LogPath.out / $LogPath.err)"
    }
    "stop") { Stop-Proxy }
    "restart") {
        Stop-Proxy
        Start-Sleep -Seconds 1
        & $PSCommandPath start -Port $Port @Rest
    }
    "status") {
        $procId = Get-ProxyPid
        if ($procId) { Write-Host "running (pid $procId)" }
        else { Write-Host "not running" }
    }
    "logs") {
        if (Test-Path -LiteralPath "$LogPath.out") {
            Get-Content -LiteralPath "$LogPath.out" -Tail 200 -Wait
        } elseif (Test-Path -LiteralPath $LogPath) {
            Get-Content -LiteralPath $LogPath -Tail 200 -Wait
        } else {
            Write-Host "no log yet"
        }
    }
}
