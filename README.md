# TIMECODE-MASTER-SYNC-APP
a time code sync app to make your workflow much faster
# LTC Sync App

A desktop application for syncing video and audio files using LTC (Linear Timecode). Automatically aligns multi-camera footage and audio recordings based on embedded timecode.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

## Features

### Timecode
- **LTC Detection** - Automatically reads Linear Timecode from audio tracks
- **BWF Support** - Reads embedded timecode from Broadcast Wave Format files
- **Apply LTC** - Embed timecode into audio files (non-destructive, fast overwrite)
- **FPS Detection** - Auto-detect 23.976, 24, 25, 29.97, 30 fps
- **Manual FPS Override** - Override detected FPS per clip

### Sync
- **Multi-camera Sync** - Align footage from unlimited cameras
- **Audio Sync** - Sync audio recorders (ZOOM, Sound Devices, etc.)
- **Split File Handling** - Seamless support for recorder file splits (-0001, -0002)
- **24h+ Recording** - Handles timecode wrap for recordings over 24 hours
- **Linked Audio** - Auto-link multi-track audio to LTC reference track

### Sync Modes
- **TC Only** - Simple timecode-based sync
- **TC + Date** - Multi-day shoots with overlapping timecodes
- **TC + Filename** - Separate sources that share same timecode

### Timeline
- **Visual Timeline** - Drag, trim, and arrange clips
- **Video Preview** - Real-time video playback with VLC
- **Audio Waveforms** - Visual audio waveform display
- **Multi-track** - Separate video and audio lanes
- **Mute/Solo** - Per-track audio control

### Export
- **DaVinci Resolve** - FCPXML 1.8 with full metadata
- **Final Cut Pro** - Native FCPXML support
- **Premiere Pro** - Import via FCPXML
- **Preserves Metadata** - Camera names, reel numbers, timecode

## Requirements

- **FFmpeg** - [Download](https://ffmpeg.org/download.html)
- **VLC** - [Download](https://www.videolan.org/vlc/)

## Installation

### Option 1: Run from Source
```bash
# Clone repository
git clone https://github.com/yourusername/ltc-sync-app.git
cd ltc-sync-app

# Install dependencies
pip install -r requirements.txt

# Run
python ltc_sync_app.py
```

### Option 2: Download Release
Download the pre-built executable from [Releases](https://github.com/idnawsi/TIMECODE-MASTER-SYNC-APP/releases).

## Quick Start

1. **Add Files** - Drag & drop video/audio files into the app
2. **Analyze** - App automatically detects LTC timecode
3. **Sync** - Click "Sync by TC" to align all clips
4. **Export** - Export FCPXML for your NLE

## Sync Modes

| Mode | Use Case |
|------|----------|
| **TC Only** | Single day shoot, all devices synced to same TC |
| **TC + Date** | Multi-day shoots with overlapping timecodes |
| **TC + Filename** | Multiple cameras with same TC, separate by source |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `J` / `L` | Reverse / Forward playback |
| `K` | Stop |
| `←` / `→` | Frame step |
| `I` / `O` | Set In/Out points |
| `M` | Mute/Unmute track |

## Export Formats

- **DaVinci Resolve** - FCPXML 1.8
- **Final Cut Pro** - FCPXML 1.8
- **Premiere Pro** - FCPXML (import via Media Browser)

## Building from Source

```bash
# Install build tools
pip install pyinstaller

# Build
python build.py

# Output: dist/LTC Sync/
```


## License

No License


