@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt pyinstaller
python -c "from alignscan.asset_generator import ensure_assets; ensure_assets('assets')"
pyinstaller --noconfirm --clean --windowed --onefile --name AlignScan --icon assets\app_icon.ico --add-data "assets;assets" --collect-all PySide6 --collect-all cv2 --collect-all onnxruntime app.py
if errorlevel 1 exit /b 1
echo.
echo Build completed: dist\AlignScan.exe
