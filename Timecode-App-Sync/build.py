#!/usr/bin/env python3
"""
LTC Sync App - Cross-Platform Build Script

Builds native executables for Windows, Linux, and macOS using PyInstaller.

Usage:
    python build.py              # Build for current platform
    python build.py --onefile    # Build single executable (larger but simpler)
    python build.py --clean      # Clean build artifacts before building

Requirements:
    - Python 3.8+
    - PyInstaller: pip install pyinstaller
    - All app dependencies: pip install -r requirements.txt

External Dependencies (must be installed separately):
    - FFmpeg: https://ffmpeg.org/download.html
    - VLC: https://www.videolan.org/vlc/

Platform Notes:
    - Windows: Creates .exe in dist/LTC Sync/
    - Linux: Creates binary in dist/LTC Sync/
    - macOS: Creates .app bundle in dist/
"""

import os
import sys
import shutil
import subprocess
import platform

APP_NAME = "LTC Sync"
MAIN_SCRIPT = "ltc_sync_app.py"
VERSION = "1.0.0"

# Hidden imports that PyInstaller might miss
HIDDEN_IMPORTS = [
    # Tkinter and GUI
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.colorchooser',
    'tkinterdnd2',

    # Image processing
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL.ImageDraw',
    'PIL.ImageFont',

    # Numerical/Scientific
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.io',
    'scipy.io.wavfile',

    # Audio
    'sounddevice',
    'soundfile',

    # Video
    'cv2',
    'vlc',

    # Standard library
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
]

# Modules to exclude (reduce size)
EXCLUDES = [
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
]


def get_platform():
    """Get current platform name."""
    system = platform.system().lower()
    if system == 'darwin':
        return 'macos'
    return system


def clean_build():
    """Remove build artifacts."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['*.pyc', '*.pyo']

    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"Removing {d}/")
            shutil.rmtree(d)

    for pattern in files_to_clean:
        import glob
        for f in glob.glob(pattern):
            print(f"Removing {f}")
            os.remove(f)


def build_windows(onefile=False):
    """Build for Windows."""
    print("Building for Windows...")

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', APP_NAME,
        '--windowed',  # No console window
        '--noconfirm',  # Overwrite without asking
    ]

    if onefile:
        cmd.append('--onefile')

    # Add hidden imports
    for imp in HIDDEN_IMPORTS:
        cmd.extend(['--hidden-import', imp])

    # Add excludes
    for exc in EXCLUDES:
        cmd.extend(['--exclude-module', exc])

    # Add icon if exists
    if os.path.exists('app_icon.ico'):
        cmd.extend(['--icon', 'app_icon.ico'])

    cmd.append(MAIN_SCRIPT)

    subprocess.run(cmd, check=True)
    print(f"\nBuild complete! Executable at: dist/{APP_NAME}/")


def build_linux(onefile=False):
    """Build for Linux."""
    print("Building for Linux...")

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', APP_NAME.replace(' ', '_'),  # No spaces in Linux
        '--windowed',
        '--noconfirm',
    ]

    if onefile:
        cmd.append('--onefile')

    # Add hidden imports
    for imp in HIDDEN_IMPORTS:
        cmd.extend(['--hidden-import', imp])

    # Add excludes
    for exc in EXCLUDES:
        cmd.extend(['--exclude-module', exc])

    # Add icon if exists
    if os.path.exists('app_icon.png'):
        cmd.extend(['--icon', 'app_icon.png'])

    cmd.append(MAIN_SCRIPT)

    subprocess.run(cmd, check=True)
    print(f"\nBuild complete! Executable at: dist/{APP_NAME.replace(' ', '_')}/")


def build_macos(onefile=False):
    """Build for macOS."""
    print("Building for macOS...")

    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', APP_NAME,
        '--windowed',
        '--noconfirm',
        '--osx-bundle-identifier', 'com.ltcsync.app',
    ]

    if onefile:
        cmd.append('--onefile')

    # Add hidden imports
    for imp in HIDDEN_IMPORTS:
        cmd.extend(['--hidden-import', imp])

    # Add excludes
    for exc in EXCLUDES:
        cmd.extend(['--exclude-module', exc])

    # Add icon if exists
    if os.path.exists('app_icon.icns'):
        cmd.extend(['--icon', 'app_icon.icns'])

    cmd.append(MAIN_SCRIPT)

    subprocess.run(cmd, check=True)
    print(f"\nBuild complete! App bundle at: dist/{APP_NAME}.app/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build LTC Sync App')
    parser.add_argument('--onefile', action='store_true',
                       help='Create a single executable file')
    parser.add_argument('--clean', action='store_true',
                       help='Clean build artifacts before building')
    parser.add_argument('--platform', choices=['windows', 'linux', 'macos'],
                       help='Target platform (default: current)')

    args = parser.parse_args()

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.clean:
        clean_build()

    target = args.platform or get_platform()

    print(f"=" * 50)
    print(f"LTC Sync App Builder v{VERSION}")
    print(f"Target Platform: {target}")
    print(f"Build Mode: {'Single File' if args.onefile else 'Directory'}")
    print(f"=" * 50)
    print()

    # Check for main script
    if not os.path.exists(MAIN_SCRIPT):
        print(f"Error: {MAIN_SCRIPT} not found!")
        sys.exit(1)

    # Check for PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Error: PyInstaller not installed!")
        print("Install with: pip install pyinstaller")
        sys.exit(1)

    print()

    if target == 'windows':
        build_windows(args.onefile)
    elif target == 'linux':
        build_linux(args.onefile)
    elif target == 'macos':
        build_macos(args.onefile)
    else:
        print(f"Error: Unknown platform '{target}'")
        sys.exit(1)

    print()
    print("=" * 50)
    print("BUILD SUCCESSFUL!")
    print("=" * 50)
    print()
    print("IMPORTANT: Users must install these separately:")
    print("  - FFmpeg: https://ffmpeg.org/download.html")
    print("  - VLC: https://www.videolan.org/vlc/")
    print()


if __name__ == '__main__':
    main()
