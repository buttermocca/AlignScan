@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "from alignscan.asset_generator import ensure_assets; ensure_assets('assets')"
python app.py
