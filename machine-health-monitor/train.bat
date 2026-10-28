@echo off
rem Trains the anomaly detector on data\healthy*.csv and writes include\model.h.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto nosetup
".venv\Scripts\python.exe" -W ignore tools\train.py
echo.
pause
exit /b 0

:nosetup
echo Run setup.bat first.
pause
exit /b 1
