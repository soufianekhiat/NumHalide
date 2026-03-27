<#
.SYNOPSIS
    Configure (if needed) and build all NumHalide examples via CMake.

.DESCRIPTION
    Configures the CMake RelWithDebInfo build if CMakeCache.txt does not exist,
    then builds every example target.  Both examples and tests are enabled in
    the configuration so the project is always in a consistent state.

.EXAMPLE
    .\build_all_examples.ps1
#>

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$root  = Split-Path $MyInvocation.MyCommand.Path -Parent
$build = Join-Path $root "build\RelWithDebInfo"
$vcpkg = Join-Path $root "build\vcpkg_installed\x64-windows"

Set-Location $root
Write-Host "=== NumHalide - Build All Examples ===" -ForegroundColor Cyan
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

# ---- Build all examples ----
Write-Host "--- cmake --build (all examples) ---" -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()

cmake --build $build --config RelWithDebInfo --parallel

$sw.Stop()
if ($LASTEXITCODE -ne 0) { throw "CMake build failed (exit $LASTEXITCODE)" }

# ---- Report ----
$exes = Get-ChildItem (Join-Path $build "RelWithDebInfo\example_*.exe") -ErrorAction SilentlyContinue |
        Sort-Object Name
Write-Host ""
Write-Host "=== Done: $($exes.Count) example executable(s) built in $($sw.Elapsed.ToString('mm\:ss')) ===" -ForegroundColor Green
$exes | ForEach-Object { Write-Host "  $($_.Name)" }
