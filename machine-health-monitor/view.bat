@echo off
rem Opens the live vibration plot. Close the PlatformIO serial monitor first.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto nosetup
".venv\Scripts\python.exe" tools\live_view.py
echo.
pause
exit /b 0

:nosetup
echo Run setup.bat first.
pause
exit /b 1
