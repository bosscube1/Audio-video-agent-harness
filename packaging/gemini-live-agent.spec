# PyInstaller spec for the Gemini Live Agent (onedir layout).
#
# Build with:
#   .venv\Scripts\pyinstaller packaging\gemini-live-agent.spec --clean --noconfirm
#
# Onedir (not --onefile) is deliberate: --onefile re-extracts the whole bundle
# to temp on every launch, which costs 3-8 s of cold start with PySide6 +
# Pillow. Wrap dist\gemini-live-agent\ in an Inno Setup installer instead.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Paths in a .spec are resolved relative to the spec file's directory, so
# anchor everything to SPECPATH (the packaging/ dir) explicitly.
_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# sounddevice ships its PortAudio DLLs in the separate _sounddevice_data
# package; mss is imported lazily by media/screen.py.
binaries = collect_dynamic_libs("_sounddevice_data")
datas = collect_data_files("_sounddevice_data")

# Heavy PySide6 modules this app never uses. Excluding them keeps the bundle
# well under 300 MB.
excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DAnimation",
    "PySide6.QtCharts",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQml",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtWebSockets",
    "PySide6.QtNetworkAuth",
    "PySide6.QtDataVisualization",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtTextToSpeech",
    "PySide6.QtVirtualKeyboard",
    # Dev-only tools that must never ship.
    "pytest",
    "pytestqt",
    "mypy",
    "ruff",
    "tkinter",
]

a = Analysis(
    [os.path.join(_ROOT, "main.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "mss",
        "mss.windows",
        "keyring.backends.Windows",
        "_sounddevice_data",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="gemini-live-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Keep a console so --headless mode, --help, and tracebacks stay usable.
    console=True,
    icon=os.path.join(SPECPATH, "icon.ico"),
    version=os.path.join(SPECPATH, "version.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="gemini-live-agent",
)
