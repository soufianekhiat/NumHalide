<#
.SYNOPSIS
    Configure (if needed) and build the NumHalide test executable via CMake.

.DESCRIPTION
    Configures the CMake RelWithDebInfo build if CMakeCache.txt does not exist,
    then builds the numhalide_tests target.

.EXAMPLE
    .\build_tests.ps1
#>

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$root  = Split-Path $MyInvocation.MyCommand.Path -Parent
$build = Join-Path $root "build\RelWithDebInfo"
$vcpkg = Join-Path $root "build\vcpkg_installed\x64-windows"

Set-Location $root
Write-Host "=== NumHalide - Build Tests ===" -ForegroundColor Cyan
Write-Host "Root  : $root"
Write-Host "Build : $build"
Write-Host ""

# ---- Configure (skip if cache already present) ----
$cacheFile = Join-Path $build "CMakeCache.txt"
if (!(Test-Path $cacheFile)) {
    Write-Host "--- CMake configure ---" -ForegroundColor Yellow
    cmake -S . -B $build `
        -DCMAKE_BUILD_TYPE=RelWithDebInfo `
        "-DCMAKE_PREFIX_PATH=$vcpkg"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }
    Write-Host ""
} else {
    Write-Host "CMake cache found - skipping configure (delete $cacheFile to force re-configure)" -ForegroundColor DarkGray
    Write-Host ""
}

# ---- Build test target ----
Write-Host "--- cmake --build (numhalide_tests) ---" -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()

cmake --build $build --config RelWithDebInfo --target numhalide_tests --parallel

$sw.Stop()
if ($LASTEXITCODE -ne 0) { throw "CMake build failed (exit $LASTEXITCODE)" }

$testExe = Join-Path $build "RelWithDebInfo\numhalide_tests.exe"
Write-Host ""
if (Test-Path $testExe) {
    Write-Host "=== Done: test executable built in $($sw.Elapsed.ToString('mm\:ss')) ===" -ForegroundColor Green
    Write-Host "  $testExe"
} else {
    Write-Host "=== Done in $($sw.Elapsed.ToString('mm\:ss')) (executable not found at expected path) ===" -ForegroundColor Yellow
}
