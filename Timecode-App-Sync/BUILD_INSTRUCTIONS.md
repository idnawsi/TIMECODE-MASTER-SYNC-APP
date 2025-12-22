# LTC Sync App - Build Instructions

Build native executables for Windows, Linux, and macOS.

## Prerequisites

### All Platforms
1. Python 3.8 or higher
2. pip (Python package manager)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Building

### Option 1: Using Build Script (Recommended)
```bash
# Build for current platform
python build.py

# Build single executable (larger file but simpler distribution)
python build.py --onefile

# Clean and rebuild
python build.py --clean
```

### Option 2: Using PyInstaller Directly
```bash
pyinstaller LTC_Sync.spec
```

## Platform-Specific Notes

### Windows

**Build:**
```cmd
python build.py
```

**Output:** `dist\LTC Sync\LTC Sync.exe`

**Requirements for end users:**
- Windows 10/11
- VLC Media Player (64-bit): https://www.videolan.org/vlc/
- FFmpeg (add to PATH): https://ffmpeg.org/download.html

**Optional:** Add `app_icon.ico` before building for custom icon.

---

### Linux (Ubuntu/Debian)

**Install system dependencies:**
```bash
sudo apt update
sudo apt install python3-tk python3-pip ffmpeg vlc
```

**Build:**
```bash
python3 build.py
```

**Output:** `dist/LTC_Sync/LTC_Sync`

**Requirements for end users:**
- VLC: `sudo apt install vlc`
- FFmpeg: `sudo apt install ffmpeg`

**Optional:** Add `app_icon.png` before building for custom icon.

**Creating AppImage (optional):**
```bash
# Install appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# Create AppDir structure and package
# (Additional setup required)
```

---

### macOS

**Install dependencies:**
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python ffmpeg
brew install --cask vlc
```

**Build:**
```bash
python3 build.py
```

**Output:** `dist/LTC Sync.app`

**Requirements for end users:**
- macOS 10.14 (Mojave) or later
- VLC Media Player: https://www.videolan.org/vlc/
- FFmpeg: `brew install ffmpeg` or download from https://ffmpeg.org/

**Optional:** Add `app_icon.icns` before building for custom icon.

**Code signing (for distribution):**
```bash
codesign --force --deep --sign "Developer ID Application: Your Name" "dist/LTC Sync.app"
```

---

## Output Structure

### Windows/Linux (Directory Mode)
```
dist/
└── LTC Sync/
    ├── LTC Sync.exe (or LTC_Sync on Linux)
    ├── python3x.dll
    ├── _internal/
    │   ├── numpy/
    │   ├── PIL/
    │   └── ...
    └── ...
```

### Windows/Linux (Single File Mode)
```
dist/
└── LTC Sync.exe (or LTC_Sync on Linux)
```

### macOS
```
dist/
└── LTC Sync.app/
    └── Contents/
        ├── Info.plist
        ├── MacOS/
        │   └── LTC Sync
        └── Resources/
```

---

## Troubleshooting

### "Module not found" errors during build
Add the missing module to `hiddenimports` in `LTC_Sync.spec` or `build.py`.

### VLC not working in built app
VLC must be installed separately by the user. The app uses the python-vlc wrapper which requires VLC to be installed on the system.

### FFmpeg not found
Users must install FFmpeg and ensure it's in their system PATH.

### tkinterdnd2 (drag-drop) not working
Ensure tkinterdnd2 data files are included. The spec file should handle this automatically.

### Large executable size
Use UPX compression (enabled by default on Windows/Linux):
```bash
# Install UPX
# Windows: Download from https://github.com/upx/upx/releases
# Linux: sudo apt install upx
# macOS: brew install upx
```

### macOS: "App is damaged" error
The app needs to be code-signed or users need to allow it:
```bash
# Allow from System Preferences > Security & Privacy
# Or run:
xattr -cr "LTC Sync.app"
```

---

## Distribution Checklist

- [ ] Build on target platform (can't cross-compile)
- [ ] Test the built executable
- [ ] Include README with FFmpeg/VLC installation instructions
- [ ] (Windows) Consider creating an installer with NSIS or Inno Setup
- [ ] (macOS) Code sign and notarize for distribution
- [ ] (Linux) Consider creating .deb or AppImage

---

## External Dependencies

Users must install these separately:

| Dependency | Windows | macOS | Linux |
|------------|---------|-------|-------|
| VLC | [Download](https://www.videolan.org/vlc/) | `brew install --cask vlc` | `sudo apt install vlc` |
| FFmpeg | [Download](https://ffmpeg.org/download.html) | `brew install ffmpeg` | `sudo apt install ffmpeg` |
