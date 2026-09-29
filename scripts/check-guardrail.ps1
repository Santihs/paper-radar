<#
.SYNOPSIS
    Checks that the OpenRouter workspace Guardrail is enforced on the paper-radar key.

.DESCRIPTION
    Calls OpenRouter directly (bypassing the paper-radar governance gate) to prove the
    platform layer works on its own:
      - models outside the allowlist must be rejected by the Guardrail (no cost), and
      - an approved model must still answer (costs a fraction of a cent).
    The API key is read from .env and never printed.

.EXAMPLE
    pwsh scripts/check-guardrail.ps1
    pwsh scripts/check-guardrail.ps1 -AllApproved   # also call all 4 approved models
#>
param(
    [switch]$AllApproved,
    [string]$EnvFile = (Join-Path $PSScriptRoot '..' '.env')
)

$ErrorActionPreference = 'Stop'

# Real model IDs (a made-up ID fails as "invalid model" and proves nothing).
$Blocked = @('deepseek/deepseek-v4.1-flash', 'qwen/qwen3.8-flash')
$Approved = @('google/gemini-3.8-flash')  # cheapest approved model
if ($AllApproved) {
    $Approved += @('anthropic/claude-sonnet-5.5', 'openai/gpt-6-sol', 'x-ai/grok-4.7')
}

if (-not (Test-Path $EnvFile)) { throw ".env not found at $EnvFile" }
$match = Select-String -Path $EnvFile -Pattern '^\s*OPENROUTER_API_KEY\s*=\s*"?([^"\s]+)"?' |
    Select-Object -First 1
if (-not $match) { throw 'OPENROUTER_API_KEY missing in .env' }
$key = $match.Matches[0].Groups[1].Value

function Invoke-Model([string]$Model) {
    # No temperature on purpose: some approved models (GPT-6 Sol) reject it.
    $body = @{
        model      = $Model
        max_tokens = 16
        messages   = @(@{ role = 'user'; content = 'Reply with OK' })
    } | ConvertTo-Json -Depth 5
    try {
        Invoke-RestMethod 'https://openrouter.ai/api/v1/chat/completions' -Method Post `
            -Headers @{ Authorization = "Bearer $key"; 'X-Title' = 'paper-radar' } `
            -ContentType 'application/json' -Body $body | Out-Null
        return @{ Ok = $true; Status = 200; Detail = 'answered' }
    } catch {
        $status = [int]$_.Exception.Response.StatusCode
        $detail = $_.ErrorDetails.Message
        try {
            $err = ($detail | ConvertFrom-Json).error
            $step = $err.metadata.failed_routing_step
            $reasons = ($err.metadata.ineligibility_reasons | ForEach-Object { $_.reason }) -join ', '
            $detail = if ($step) { "$step ($reasons)" } else { $err.message }
        } catch { }
        return @{ Ok = $false; Status = $status; Detail = $detail }
    }
}

$failures = 0
Write-Host "`nModels outside the allowlist (expected: rejected by Guardrail, no cost)"
foreach ($model in $Blocked) {
    $r = Invoke-Model $model
    $byGuardrail = -not $r.Ok -and $r.Detail -match 'guardrail'
    if ($byGuardrail) { Write-Host "  PASS  $model -> $($r.Status) $($r.Detail)" -ForegroundColor Green }
    else { Write-Host "  FAIL  $model -> $($r.Status) $($r.Detail)" -ForegroundColor Red; $failures++ }
}

Write-Host "`nApproved models (expected: answer)"
foreach ($model in $Approved) {
    $r = Invoke-Model $model
    if ($r.Ok) { Write-Host "  PASS  $model -> $($r.Status) $($r.Detail)" -ForegroundColor Green }
    else { Write-Host "  FAIL  $model -> $($r.Status) $($r.Detail)" -ForegroundColor Red; $failures++ }
}

if ($failures) {
    Write-Host "`n$failures check(s) failed. Hints: key created outside the workspace? budget reached? model missing from the allowlist?" -ForegroundColor Red
    exit 1
}
Write-Host "`nGuardrail enforced: blocked models rejected, approved models answer." -ForegroundColor Green
