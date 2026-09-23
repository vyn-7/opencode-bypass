# One-command install + autostart for the opencode free-tier proxy (Windows).
#
#   .\install.ps1                 install, enable autostart, start now
#   .\install.ps1 -Port 18789     same on a custom port
#   .\install.ps1 -NoAutostart    install only, do not enable autostart
#   .\install.ps1 -Uninstall      remove autostart (code + venv are kept)
#
# Autostart: drops a launcher .cmd into the user Startup folder
# (no admin required; runs at interactive logon).
[CmdletBinding()]
param(
    [int]$Port = 18788,
    [switch]$NoAutostart,
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$LauncherName = "opencode-bypass-autostart.cmd"
$StartupDir = [Environment]::GetFolderPath("Startup")
$LauncherPath = Join-Path $StartupDir $LauncherName
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Pythonw = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
$ProxyScript = Join-Path $PSScriptRoot "opencode_proxy.py"
$LogPath = Join-Path $PSScriptRoot "opencode-proxy.log"

function Find-Python {
    foreach ($cmd in @("python", "py")) {
        $c = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($c) { return $c.Source }
    }
    throw "Python 3.10+ not found. Install from https://www.python.org/downloads/ (check 'Add python.exe to PATH')."
}

function Remove-Autostart {
    if (Test-Path -LiteralPath $LauncherPath) {
        Remove-Item -LiteralPath $LauncherPath -Force
        Write-Host "Autostart launcher removed: $LauncherPath"
    } else {
        Write-Host "No autostart launcher found."
    }
}

if ($Uninstall) {
    Write-Host "Removing autostart..."
    Remove-Autostart
    & (Join-Path $PSScriptRoot "run.ps1") stop 2>$null
    Write-Host "Autostart removed. Code and .venv kept (delete the directory to remove fully)."
    exit 0
}

Write-Host "== 1/4 prerequisites =="
$py = Find-Python
Write-Host "python: $py"
$oc = Get-Command opencode -ErrorAction SilentlyContinue
if ($oc) {
    Write-Host "opencode CLI: $($oc.Source)"
} else {
    Write-Host "WARNING: 'opencode' not on PATH — install it from https://opencode.ai"
    Write-Host "and make sure it is authenticated, then re-run .\install.ps1."
}

Write-Host "== 2/4 python environment =="
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".venv"))) {
    & $py -m venv .venv
}
& $Python -m pip install -q -r requirements.txt
& $Python -m py_compile opencode_proxy.py
Write-Host "venv ready."

Write-Host "== 3/4 autostart =="
if ($NoAutostart) {
    Write-Host "Skipped (-NoAutostart). Start manually with: .\run.ps1 start -Port $Port"
} else {
    # Launcher survives terminal close; pythonw avoids a permanent console window.
    $proj = $PSScriptRoot -replace '"', '""'
    $lines = @(
        "@echo off",
        "cd /d `"$proj`"",
        "if not exist `"$LogPath`" (echo. > `"$LogPath`")",
        "start `"`" /b `"$Pythonw`" `"$ProxyScript`" --port $Port >> `"$LogPath`" 2>&1"
    )
    Set-Content -LiteralPath $LauncherPath -Value $lines -Encoding ASCII
    Write-Host "Autostart installed: $LauncherPath (port $Port)"
    Write-Host "Logs: $LogPath  |  Remove with: .\install.ps1 -Uninstall"
}

Write-Host "== 4/4 verify =="
& (Join-Path $PSScriptRoot "run.ps1") status 2>$null | Out-Null
$running = $false
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 -ErrorAction Stop
    $running = $true
} catch { }

if (-not $running) {
    & (Join-Path $PSScriptRoot "run.ps1") start -Port $Port
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -ErrorAction Stop | Out-Null
            $running = $true
            break
        } catch { }
    }
}

if (-not $running) {
    Write-Error "Proxy is not responding on :$Port. Check $LogPath"
    exit 1
}

$health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
$health | ConvertTo-Json -Compress
Write-Host ""
Write-Host "Done. Proxy is live at http://127.0.0.1:$Port"
Write-Host "Smoke test: .\test_proxy.ps1 http://127.0.0.1:$Port"
