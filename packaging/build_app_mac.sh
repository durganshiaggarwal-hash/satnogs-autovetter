#!/bin/sh
# Builds "Satellite Signal Analyser.app".  Run from the project folder, with (.venv) switched on:
#     sh packaging/build_app_mac.sh
# The result is  dist/Satellite Signal Analyser.app
set -e
pip install pyinstaller
pyinstaller --noconfirm --distpath dist --workpath build packaging/satellite_signal_analyser.spec
echo
echo "Done. Put the trained AI in:  dist/satrecovery_data/models  (next to the .app)"
