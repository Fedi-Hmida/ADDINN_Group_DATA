@echo off
echo ========================================
echo YOLOv8 API - Windows Firewall Setup
echo ========================================
echo.
echo This script will add a firewall rule to allow
echo incoming connections on port 8000 for your API.
echo.
echo NOTE: This script must be run as Administrator!
echo.
pause

echo Adding firewall rule for YOLOv8 API...
netsh advfirewall firewall add rule name="YOLOv8 API Inbound" dir=in action=allow protocol=TCP localport=8000

if %errorlevel% equ 0 (
    echo ✅ SUCCESS: Firewall rule added successfully!
    echo.
    echo Your API should now be accessible from other computers
    echo on the same network at: http://192.168.100.14:8000/docs
) else (
    echo ❌ ERROR: Failed to add firewall rule
    echo Make sure you're running this script as Administrator
)

echo.
echo ========================================
echo Press any key to exit...
pause > nul
