# PyInstaller recipe for the double-click Satellite Signal Analyser.
# Build it with build_app_windows.bat (Windows) or build_app_mac.sh (Mac);
# each must be run on that kind of computer.
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent
NAME = "Satellite Signal Analyser"

datas = [
    (str(ROOT / "satrecovery" / "gui" / "assets"), "satrecovery/gui/assets"),   # fonts and icons
    (str(ROOT / "examples"), "examples"),                                       # files for "Your own file"
]
hidden = collect_submodules("satrecovery") + collect_submodules("uvicorn") + ["h5py", "pyqtgraph"]

a = Analysis(
    [str(ROOT / "packaging" / "app_launcher.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "pytest", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=False)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME)
app = BUNDLE(coll, name=f"{NAME}.app", bundle_identifier="io.github.durganshiaggarwal-hash.satellite-signal-analyser")
