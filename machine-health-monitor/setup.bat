@echo off
rem One-time setup: creates a private Python environment in .venv and installs
rem the packages the tools need. Uses the Python that ships with PlatformIO.
cd /d "%~dp0"

set "PIOPY=%USERPROFILE%\.platformio\python3\python.exe"
if not exist "%PIOPY%" set "PIOPY=%USERPROFILE%\.platformio\penv\Scripts\python.exe"
if not exist "%PIOPY%" goto nopython

if exist ".venv\Scripts\python.exe" goto install
echo Creating a private Python environment in .venv ...
"%PIOPY%" -m venv .venv
if errorlevel 1 goto fail

:install
echo Installing packages. This can take a few minutes ...
".venv\Scripts\python.exe" -m pip install -r tools\requirements.txt
if errorlevel 1 goto fail

echo.
echo Setup finished. You can close this window.
pause
exit /b 0

:nopython
echo Could not find the Python that ships with PlatformIO.
echo Install Python from python.org, then run:  pip install -r tools\requirements.txt
pause
exit /b 1

:fail
echo.
echo Setup failed. Copy the text above and send it to Claude.
pause
exit /b 1
