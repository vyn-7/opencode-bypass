# Smoke-test the proxy. Expects it running (.\install.ps1 or .\run.ps1 start).
# Fails loudly (nonzero exit) on the first broken endpoint.
[CmdletBinding()]
param(
    [string]$Base = "http://127.0.0.1:18788"
)
$ErrorActionPreference = "Stop"

function Get-Json([string]$Uri, [hashtable]$Body) {
    if ($Body) {
        $json = $Body | ConvertTo-Json -Depth 10 -Compress
        return Invoke-RestMethod -Uri $Uri -Method Post `
            -ContentType "application/json" -Body $json -TimeoutSec 180
    }
    return Invoke-RestMethod -Uri $Uri -TimeoutSec 30
}

Write-Host "== /health =="
$health = Get-Json "$Base/health"
$health | ConvertTo-Json -Compress

Write-Host "== /v1/models =="
$models = Get-Json "$Base/v1/models"
$ids = @($models.data | ForEach-Object { $_.id })
Write-Host ("{0} models: {1}" -f $ids.Count, ($ids -join ", "))
if ($ids.Count -lt 1) { throw "no models advertised" }

Write-Host "== non-streaming chat =="
$blocking = Get-Json "$Base/v1/chat/completions" @{
    model    = "big-pickle"
    messages = @(@{ role = "user"; content = "say hi in 3 words" })
}
Write-Host ($blocking.choices[0].message.content)

Write-Host "== streaming chat (SSE parse) =="
$sseBody = @{
    model    = "big-pickle"
    stream   = $true
    messages = @(@{ role = "user"; content = "say hi in 3 words" })
} | ConvertTo-Json -Depth 10 -Compress
$resp = Invoke-WebRequest -Uri "$Base/v1/chat/completions" -Method Post `
    -ContentType "application/json" -Body $sseBody -TimeoutSec 180
$streamText = $resp.Content
if ($streamText -notmatch 'data:') { throw "stream has no data: lines" }
if ($streamText -notmatch '\[DONE\]') { throw "stream missing [DONE]" }
Write-Host "stream ok ($([math]::Min($streamText.Length, 120)) chars preview)"

Write-Host "== memory recall (turn 2, delta session) =="
$t1 = Get-Json "$Base/v1/chat/completions" @{
    model    = "big-pickle"
    messages = @(@{ role = "user"; content = "My favorite color is teal. Reply with just OK." })
}
$t1Content = $t1.choices[0].message.content
Write-Host $t1Content
$t2 = Get-Json "$Base/v1/chat/completions" @{
    model    = "big-pickle"
    messages = @(
        @{ role = "user"; content = "My favorite color is teal. Reply with just OK." },
        @{ role = "assistant"; content = $t1Content },
        @{ role = "user"; content = "What is my favorite color? One word only." }
    )
}
$t2Content = $t2.choices[0].message.content
Write-Host "turn2 said: $t2Content"
if ($t2Content -notmatch "teal") { throw "turn 2 does not recall 'teal'" }

Write-Host "== tools bridge (blocking) =="
$toolResp = Get-Json "$Base/v1/chat/completions" @{
    model       = "big-pickle"
    messages    = @(@{ role = "user"; content = "List files with the list_files tool. One word only after." })
    tools       = @(
        @{
            type     = "function"
            function = @{
                name        = "list_files"
                description = "List files in the current directory"
                parameters  = @{
                    type       = "object"
                    properties = @{ path = @{ type = "string" } }
                    required   = @("path")
                }
            }
        }
    )
    tool_choice = "auto"
}
$choice = $toolResp.choices[0]
Write-Host ("finish=" + $choice.finish_reason)
if ($choice.finish_reason -eq "tool_calls") {
    $tc = $choice.message.tool_calls[0]
    Write-Host ("tool_call: " + $tc.function.name + " " + $tc.function.arguments)
    if (-not $tc.function.name) { throw "tool_call missing function.name" }
} else {
    Write-Host ("content: " + $choice.message.content)
    Write-Host "model answered in text (acceptable; retry if flaky)"
}

Write-Host "== legacy path /chat/completions =="
$legacy = Get-Json "$Base/chat/completions" @{
    model    = "big-pickle"
    messages = @(@{ role = "user"; content = "say hi in 3 words" })
}
Write-Host ($legacy.choices[0].message.content)

Write-Host "ALL SMOKE TESTS DONE"
