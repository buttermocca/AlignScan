@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt pyinstaller
python -c "from alignscan.asset_generator import ensure_assets; ensure_assets('assets')"
pyinstaller --noconfirm --clean --windowed --onedir --name AlignScan --icon assets\app_icon.ico --add-data "assets;assets" --collect-all PySide6 --collect-all cv2 --collect-all onnxruntime --exclude-module matplotlib --exclude-module scipy --exclude-module torch --exclude-module torchvision --exclude-module ultralytics app.py
if errorlevel 1 exit /b 1
copy /Y README.md dist\AlignScan\README.txt >nul
echo.
echo Build completed: dist\AlignScan\AlignScan.exe
echo Keep the complete AlignScan folder together when moving the program.
