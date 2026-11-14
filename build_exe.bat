@echo off
cd /d "%~dp0"
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --windowed --name "ML IDE" ^
  --exclude-module torch --exclude-module ultralytics --exclude-module tkinter ^
  --icon assets\icon.ico --add-data "assets;assets" ^
  --collect-submodules mlide mlide\__main__.py
echo.
echo Done: dist\ML IDE\ML IDE.exe
