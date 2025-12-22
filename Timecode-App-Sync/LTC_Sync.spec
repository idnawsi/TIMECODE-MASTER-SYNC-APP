# -*- mode: python ; coding: utf-8 -*-
"""
LTC Sync App - PyInstaller Spec File

Cross-platform spec file for building Windows, Linux, and macOS executables.

Build with:
    pyinstaller LTC_Sync.spec

Or use the build script:
    python build.py
"""

import os
import sys
import platform
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Detect platform
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')

# App info
APP_NAME = 'LTC Sync'
APP_VERSION = '1.0.0'

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # Tkinter and GUI
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.colorchooser',
    'tkinter.font',

    # TkinterDnD (drag and drop)
    'tkinterdnd2',

    # Image processing
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'PIL.ImageOps',

    # Numerical/Scientific
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    'scipy',
    'scipy.signal',
    'scipy.io',
    'scipy.io.wavfile',
    'scipy.fft',

    # Audio
    'sounddevice',
    'soundfile',
    'soundfile._soundfile_data',

    # Video
    'cv2',

    # VLC (optional - wrapper only, needs VLC installed)
    'vlc',

    # Standard library that might be missed
    'json',
    'threading',
    'queue',
    'tempfile',
    'shutil',
    'subprocess',
    'xml.etree.ElementTree',
    'urllib.parse',
    'fractions',
    'struct',
    'wave',
    'typing',
    'dataclasses',
    'collections',
    'functools',
    'itertools',
    'pathlib',
    'datetime',
    'time',
    're',
    'os',
    'sys',
]

# Collect tkinterdnd2 data files
try:
    tkdnd_datas = collect_data_files('tkinterdnd2')
except Exception:
    tkdnd_datas = []

# Collect soundfile data
try:
    sf_datas = collect_data_files('soundfile')
except Exception:
    sf_datas = []

# Platform-specific settings
if IS_WINDOWS:
    icon_file = 'app_icon.ico' if os.path.exists('app_icon.ico') else None
    console = False
elif IS_MACOS:
    icon_file = 'app_icon.icns' if os.path.exists('app_icon.icns') else None
    console = False
else:  # Linux
    icon_file = 'app_icon.png' if os.path.exists('app_icon.png') else None
    console = False

a = Analysis(
    ['ltc_sync_app.py'],
    pathex=[],
    binaries=[],
    datas=tkdnd_datas + sf_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'matplotlib',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
        'test',
        'tests',
        'unittest',
        'doctest',
        'pdb',
        'profile',
        'cProfile',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if IS_MACOS:
    # macOS: Create .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,  # UPX can cause issues on macOS
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=True,  # For file dropping on macOS
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_file,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=APP_NAME,
    )

    app = BUNDLE(
        coll,
        name=f'{APP_NAME}.app',
        icon=icon_file,
        bundle_identifier='com.ltcsync.app',
        info_plist={
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleVersion': APP_VERSION,
            'CFBundleShortVersionString': APP_VERSION,
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,  # Support dark mode
        },
    )

else:
    # Windows and Linux
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME if IS_WINDOWS else APP_NAME.replace(' ', '_'),
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_file,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME if IS_WINDOWS else APP_NAME.replace(' ', '_'),
    )
