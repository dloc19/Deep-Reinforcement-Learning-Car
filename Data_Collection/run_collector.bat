@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo CARLA semantic data collector
echo Config: collector_config.json
echo ==============================================

python collect_data.py --config collector_config.json

echo.
if errorlevel 1 (
    echo Collector ket thuc do co loi. Kiem tra thong bao phia tren.
) else (
    echo Da thu xong session hien tai. Ban co the chuyen sang map tiep theo.
)
pause
endlocal
