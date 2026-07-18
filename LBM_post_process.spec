# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-folder Win64 build for LBM_post_process."""

from pathlib import Path


ROOT = Path(SPECPATH).resolve()
APP_NAME = "LBM_post_process"
ICON = ROOT / "assets" / "lbm_post_process.ico"
VERSION_INFO = ROOT / "windows_version_info.txt"

hidden_imports = [
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "vtkmodules.vtkInteractionStyle",
    "vtkmodules.vtkRenderingFreeType",
    "vtkmodules.vtkRenderingOpenGL2",
]

# These large optional ecosystems are not used by this application. Excluding
# them keeps the portable folder and installer focused on NumPy, TIFF, Qt/VTK.
excluded_modules = [
    "IPython",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "jupyter",
    "matplotlib",
    "pandas",
    "pytest",
    "scipy",
    "tkinter",
]

a = Analysis(
    [str(ROOT / "phase_viewer.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ICON), "assets"),
        (str(ROOT / "README.md"), "."),
        (str(ROOT / "使用说明.txt"), "."),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
    version=str(VERSION_INFO),
    uac_admin=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
