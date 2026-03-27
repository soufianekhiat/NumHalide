@echo off
cd /d C:\git\NumHalide

echo === Running Sharpmake ===
buildsystem\tools\sharpmake\Sharpmake.Application.exe "/sources(@'buildsystem\sharpmake\main.cs')"
if %ERRORLEVEL% NEQ 0 (
    echo SHARPMAKE FAILED with error %ERRORLEVEL%
    exit /b 1
)
echo === Sharpmake Done ===

call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64

echo === Building Examples 21-40 ===

set EXAMPLES=example_21_trigonometry example_22_math example_23_cumulative example_24_splitting example_25_closeness example_26_statistics_ext example_27_random_ext example_28_array_compare example_29_bitwise example_30_windows example_31_rfft example_32_gradient example_33_morphology example_34_color example_35_polynomial example_36_distance example_37_stencil example_38_histogram example_39_spectral example_40_threshold

for %%E in (%EXAMPLES%) do (
    echo --- Building %%E ---
    for /f "delims=" %%F in ('dir /b /s "projects\win64\%%E\*.vcxproj" 2^>nul') do (
        MSBuild "%%F" /p:Configuration=Release /p:Platform=x64 /m /v:minimal
        if %ERRORLEVEL% NEQ 0 (
            echo BUILD FAILED: %%E
        )
    )
)

echo === All Builds Done ===
