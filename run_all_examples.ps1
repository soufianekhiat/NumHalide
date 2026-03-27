<#
.SYNOPSIS
    Run all NumHalide example executables and report pass/fail.

.DESCRIPTION
    Discovers every example_*.exe in the RelWithDebInfo output directory,
    runs each one with the repo root as the working directory (so output PNGs
    land in out\), and prints a summary of successes and failures.

.PARAMETER Filter
    Optional glob to restrict which examples run (e.g. "example_0[1-3]*").

.EXAMPLE
    .\run_all_examples.ps1
    .\run_all_examples.ps1 -Filter "example_04*"
#>

#Requires -Version 5.1
param(
    [string]$Filter = "example_*.exe"
)

$ErrorActionPreference = "Stop"

$root   = Split-Path $MyInvocation.MyCommand.Path -Parent
$binDir = Join-Path $root "build\RelWithDebInfo\RelWithDebInfo"
$outDir = Join-Path $root "out"

Set-Location $root
Write-Host "=== NumHalide - Run All Examples ===" -ForegroundColor Cyan
Write-Host "Bin : $binDir"
Write-Host "Out : $outDir"
Write-Host ""

# Ensure output directory exists
if (!(Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$exes = Get-ChildItem (Join-Path $binDir $Filter) -ErrorAction SilentlyContinue |
        Sort-Object Name

if ($exes.Count -eq 0) {
    Write-Host "No example executables found matching '$Filter' in $binDir" -ForegroundColor Yellow
    Write-Host "Run .\build_all_examples.ps1 first."
    exit 1
}

Write-Host "Found $($exes.Count) example(s) to run." -ForegroundColor DarkGray
Write-Host ""

$passed  = [System.Collections.Generic.List[string]]::new()
$failed  = [System.Collections.Generic.List[string]]::new()
$totalSw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($exe in $exes) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "--- $($exe.Name) ---" -ForegroundColor Yellow

    Push-Location $root
    try {
        & $exe.FullName
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }

    $sw.Stop()
    if ($exitCode -eq 0) {
        Write-Host "  OK  ($($sw.Elapsed.ToString('ss\.fff'))s)" -ForegroundColor Green
        $passed.Add($exe.Name)
    } else {
        Write-Host "  FAIL  exit=$exitCode  ($($sw.Elapsed.ToString('ss\.fff'))s)" -ForegroundColor Red
        $failed.Add($exe.Name)
    }
    Write-Host ""
}

$totalSw.Stop()

Write-Host "==========================================" -ForegroundColor Cyan
if ($failed.Count -eq 0) {
    Write-Host "Results: $($passed.Count) passed, 0 failed  (total $($totalSw.Elapsed.ToString('mm\:ss')))" -ForegroundColor Green
} else {
    Write-Host "Results: $($passed.Count) passed, $($failed.Count) failed  (total $($totalSw.Elapsed.ToString('mm\:ss')))" -ForegroundColor Red
    Write-Host ""
    Write-Host "Failed examples:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
