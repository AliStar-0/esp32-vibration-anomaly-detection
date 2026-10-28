@echo off
rem Records spectra into data\<name>.csv while showing the live plot.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto nosetup

set "NAME="
set /p "NAME=Recording name, for example healthy: "
if "%NAME%"=="" exit /b 1
set "SECS="
set /p "SECS=How many seconds? [120]: "
if "%SECS%"=="" set "SECS=120"

".venv\Scripts\python.exe" tools\live_view.py --record "%NAME%" --seconds %SECS%
echo.
pause
exit /b 0

:nosetup
echo Run setup.bat first.
pause
exit /b 1
