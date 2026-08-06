<#
  confirm_800.ps1 - full-pool confirmation of the search-150 finalist.

  g_card_twostage was the only config to beat b_persona with a bootstrap CI
  excluding zero on BOTH seeds at 150 raters (dMAE -0.014 / -0.011). The full
  eligible pool (~700+ raters) has a ~0.003 noise floor - the eval that can
  actually resolve a ~0.012 effect.

  Runs (tags fp_*):
    b_persona s0, b_persona s0 REPEAT (empirical floor check),
    b_persona s1, g_card_twostage s0, g_card_twostage s1.

  Resilient: EAP Continue, 2 attempts/run, resume-by-tag, --max-retries 6.
  ASCII only (PS 5.1 reads BOM-less .ps1 as ANSI).
#>

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

$Data      = "../car_ratings_long_July8_July22_merged_with_color_and_Q23.csv"
$ImageRoot = "../selected_2000_isometric_upload_chunks_renamed"
$Raters    = 800
$TestSize  = 20
$Conc      = 8
$Runs      = @(
    @{C = "b_persona";       S = 0; T = "fp_b_persona_s0"},
    @{C = "g_card_twostage"; S = 0; T = "fp_g_card_twostage_s0"},
    @{C = "b_persona";       S = 1; T = "fp_b_persona_s1"},
    @{C = "g_card_twostage"; S = 1; T = "fp_g_card_twostage_s1"},
    @{C = "b_persona";       S = 0; T = "fp_b_persona_s0_repeat"},
    # No-context anchor (N=0, default prompt): separates generic car-reading
    # skill from what context/personalization adds, on the same paired cells.
    @{C = "b_nocontext";     S = 0; T = "fp_b_nocontext_s0"},
    @{C = "b_nocontext";     S = 1; T = "fp_b_nocontext_s1"}
)

$Py = Join-Path $PSScriptRoot "car_ai_judges\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "venv interpreter not found at $Py" }

$logDir = Join-Path $PSScriptRoot "sweep_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log   = Join-Path $logDir ("confirm800_{0}.log" -f $stamp)

$done = @()
if (Test-Path "results.tsv") {
    $done = Select-String -Path "results.tsv" -Pattern "confirm800 tag=(\S+)" -AllMatches |
        ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value }
}

$failed = 0
foreach ($r in $Runs) {
    $tag = $r.T
    if ($done -contains $tag) {
        Write-Host ("skip {0} (already in results.tsv)" -f $tag) -ForegroundColor DarkGray
        continue
    }
    $ok = $false
    foreach ($try in 1, 2) {
        Write-Host ("`n--- {0} attempt {1} ({2}) ---" -f $tag, $try, (Get-Date -Format o)) -ForegroundColor Yellow
        $a = @(
            "run_autojudge.py",
            "--data", $Data, "--image-root", $ImageRoot,
            "--config", ("configs/{0}.json" -f $r.C),
            "--raters", $Raters, "--test-size", $TestSize,
            "--split-seed", $r.S, "--concurrency", $Conc,
            "--max-retries", 6,
            "--run-tag", $tag,
            "--note", ("confirm800 tag={0} config={1} seed={2}" -f $tag, $r.C, $r.S)
        )
        & $Py @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        Write-Host ("{0} attempt {1} FAILED (exit {2})" -f $tag, $try, $LASTEXITCODE) -ForegroundColor Red
        Start-Sleep -Seconds 60
    }
    if (-not $ok) { $failed++ }
}

if ($failed -gt 0) { Write-Host ("{0} run(s) failed - see {1}" -f $failed, $log) -ForegroundColor Red; exit 1 }
Write-Host "DONE - full-pool confirmation complete" -ForegroundColor Green
