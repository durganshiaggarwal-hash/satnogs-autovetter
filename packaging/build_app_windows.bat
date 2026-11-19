@echo off
REM Builds "Satellite Signal Analyser.exe".  Run from the project folder, with (.venv) switched on:
REM     packaging\build_app_windows.bat
REM The result is the folder  dist\Satellite Signal Analyser  (copy the whole folder).
pip install pyinstaller
pyinstaller --noconfirm --distpath dist --workpath build packaging\satellite_signal_analyser.spec
echo.
echo Done. Copy the trained AI into:  dist\Satellite Signal Analyser\satrecovery_data\models
