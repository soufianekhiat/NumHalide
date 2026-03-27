<#
.SYNOPSIS
    Run the NumHalide GoogleTest suite and report results.

.DESCRIPTION
    Finds numhalide_tests.exe in the RelWithDebInfo output directory and runs it.
    Optional -Filter passes a --gtest_filter pattern, and -Xml writes a JUnit
    XML report to the specified path.

.PARAMETER Filter
    GoogleTest filter string (e.g. "La6*" or "Shape.*").

.PARAMETER Xml
    Path for the JUnit XML output file (passed as --gtest_output=xml:<path>).

.EXAMPLE
    .\run_all_tests.ps1
    .\run_all_tests.ps1 -Filter "La6*"
    .\run_all_tests.ps1 -Xml test_results.xml
#>

#Requires -Version 5.1
param(
    [string]$Filter = "",
    [string]$Xml    = ""
)

$ErrorActionPreference = "Stop"

$root    = Split-Path $MyInvocation.MyCommand.Path -Parent
$binDir  = Join-Path $root "build\RelWithDebInfo\RelWithDebInfo"
$testExe = Join-Path $binDir "numhalide_tests.exe"

Set-Location $root
Write-Host "=== NumHalide - Run Tests ===" -ForegroundColor Cyan
Write-Host "Exe : $testExe"
Write-Host ""

if (!(Test-Path $testExe)) {
    Write-Host "Test executable not found: $testExe" -ForegroundColor Red
    Write-Host "Run .\build_tests.ps1 first."
    exit 1
}

# Build argument list
$gtestArgs = @()
if ($Filter) { $gtestArgs += "--gtest_filter=$Filter" }
if ($Xml)    { $gtestArgs += "--gtest_output=xml:$Xml" }

$sw = [System.Diagnostics.Stopwatch]::StartNew()

& $testExe @gtestArgs
$exitCode = $LASTEXITCODE

$sw.Stop()
Write-Host ""

if ($exitCode -eq 0) {
    Write-Host "=== All tests passed  ($($sw.Elapsed.ToString('mm\:ss'))) ===" -ForegroundColor Green
} else {
    Write-Host "=== Tests FAILED  exit=$exitCode  ($($sw.Elapsed.ToString('mm\:ss'))) ===" -ForegroundColor Red
    exit $exitCode
}
