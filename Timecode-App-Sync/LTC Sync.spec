# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ltc_sync_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['tkinter', 'tkinter.ttk', 'tkinter.messagebox', 'tkinter.filedialog', 'tkinter.colorchooser', 'tkinterdnd2', 'PIL', 'PIL.Image', 'PIL.ImageTk', 'PIL.ImageDraw', 'PIL.ImageFont', 'numpy', 'scipy', 'scipy.signal', 'scipy.io', 'scipy.io.wavfile', 'sounddevice', 'soundfile', 'cv2', 'vlc', 'json', 'threading', 'queue', 'tempfile', 'shutil', 'subprocess', 'xml.etree.ElementTree', 'urllib.parse', 'fractions', 'struct', 'wave', 'typing', 'dataclasses'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest', 'test', 'tests', 'unittest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LTC Sync',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LTC Sync',
)
