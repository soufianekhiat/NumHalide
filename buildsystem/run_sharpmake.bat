cd /d C:\git\NumHalide
echo Running Sharpmake... > sharpmake_output.txt
buildsystem\tools\sharpmake\Sharpmake.Application.exe "/sources(@'buildsystem\sharpmake\main.cs')" /verbose >> sharpmake_output.txt 2>&1
echo Done with exit code %ERRORLEVEL% >> sharpmake_output.txt
