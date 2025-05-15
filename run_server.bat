@echo off
echo Eco Waste Classification System
echo ==============================
echo.

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running with administrator privileges...
) else (
    echo Please run this script as Administrator to modify hosts file
    echo Right-click on this batch file and select "Run as administrator"
    pause
    exit /b
)

echo Adding eco-waste.local to hosts file...
echo 127.0.0.1 eco-waste.local >> %WINDIR%\System32\drivers\etc\hosts

echo.
echo Starting waste classification server...
echo Access the application at: http://eco-waste.local:8080/classifier
echo.

python app.py

echo.
echo Server stopped.
pause 