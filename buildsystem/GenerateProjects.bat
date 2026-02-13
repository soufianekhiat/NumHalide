@echo off

echo Generating Visual Studio projects with Sharpmake...

if not exist "buildsystem\tools\sharpmake\Sharpmake.Application.exe" (
    echo ERROR: Sharpmake not built yet!
    echo Please run buildsystem\Startup.bat first to build Sharpmake.
    pause
    exit /b 1
)

set SHARPMAKE_FILE=buildsystem/sharpmake/main.cs
buildsystem\tools\sharpmake\Sharpmake.Application.exe /sources('%SHARPMAKE_FILE%') /verbose

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! Projects generated!
    echo.
    echo You can now open: numhalide_win64.sln
    echo.
) else (
    echo.
    echo ERROR: Project generation failed!
    echo.
)
