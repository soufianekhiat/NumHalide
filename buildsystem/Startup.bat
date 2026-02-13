@echo off

echo ========================================
echo Building Sharpmake
echo ========================================
echo.

REM Create output directory
mkdir buildsystem\tools\sharpmake 2>nul

REM Build Sharpmake using dotnet with correct platform
echo Building Sharpmake with dotnet...
cd extern\Sharpmake

dotnet build Sharpmake.sln -c Release /p:Platform="Any CPU"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to build Sharpmake!
    echo Make sure you have .NET 6.0 SDK installed.
    echo Download from: https://dotnet.microsoft.com/download/dotnet/6.0
    cd ..\..
    pause
    exit /b 1
)

cd ..\..

echo.
echo ========================================
echo Copying Sharpmake binaries
echo ========================================
echo.

REM Copy Sharpmake.Application (main executable)
xcopy /Y /Q extern\Sharpmake\Sharpmake.Application\bin\Release\net6.0\*.dll buildsystem\tools\sharpmake\ 2>nul
xcopy /Y /Q extern\Sharpmake\Sharpmake.Application\bin\Release\net6.0\*.exe buildsystem\tools\sharpmake\ 2>nul
xcopy /Y /Q extern\Sharpmake\Sharpmake.Application\bin\Release\net6.0\*.json buildsystem\tools\sharpmake\ 2>nul
xcopy /Y /Q extern\Sharpmake\Sharpmake.Application\bin\Release\net6.0\*.runtimeconfig.json buildsystem\tools\sharpmake\ 2>nul

REM Copy Sharpmake.Generators (required for project generation)
xcopy /Y /Q extern\Sharpmake\Sharpmake.Generators\bin\Release\net6.0\*.dll buildsystem\tools\sharpmake\ 2>nul

REM Copy Sharpmake core library
xcopy /Y /Q extern\Sharpmake\Sharpmake\bin\Release\net6.0\Sharpmake.dll buildsystem\tools\sharpmake\ 2>nul

echo.
if exist buildsystem\tools\sharpmake\Sharpmake.Application.exe (
    echo ========================================
    echo Sharpmake Setup Complete!
    echo ========================================
    echo.
    echo Next steps:
    echo   1. Run: buildsystem\GenerateProjects.bat
    echo   2. Open: numhalide_win64.sln
) else (
    echo ========================================
    echo WARNING: Sharpmake.Application.exe not found!
    echo ========================================
    echo Build may have failed. Check output above.
)
echo.

pause
