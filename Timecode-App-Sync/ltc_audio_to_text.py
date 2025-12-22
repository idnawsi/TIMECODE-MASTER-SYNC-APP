"""
LTC Audio to Text Converter
Reads LTC timecode from audio and outputs detailed timing information to text file.
"""

import os
import sys
import wave
import subprocess
import tempfile
import json
from collections import deque
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)


class LTCAnalyzer:
    """Detailed LTC analyzer with comprehensive output."""

    SYNC_WORD = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1]  # 0xBFFC
    SYNC_WORD_REV = [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0]

    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.reset()

    def reset(self):
        self.last_sample = 0
        self.samples_since_edge = 0
        self.edge_intervals = []
        self.all_edges = []  # Store all edge positions for analysis
        self.bit_buffer = deque(maxlen=400)
        self.frames = []
        self.raw_bits = []
        self.samples_per_bit = 0
        self.half_bit_samples = 0

    def analyze_audio(self, samples):
        """Analyze audio and extract all information."""
        samples = np.array(samples, dtype=np.float32)

        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val

        # Find all zero crossings
        signs = np.sign(samples)
        signs[signs == 0] = 1
        crossings = np.where(np.diff(signs) != 0)[0]

        print(f"  Found {len(crossings)} zero crossings in {len(samples)} samples")

        if len(crossings) < 100:
            print("  ERROR: Not enough zero crossings - may not be LTC audio")
            return

        # Calculate intervals between crossings
        intervals = np.diff(crossings)
        self.edge_intervals = intervals.tolist()
        self.all_edges = crossings.tolist()

        # Analyze timing
        self._analyze_timing(intervals)

        # Decode frames
        self._decode_all_frames(samples, crossings)

    def _analyze_timing(self, intervals):
        """Analyze timing to detect bit rate."""
        # Remove outliers
        q1, q3 = np.percentile(intervals, [25, 75])
        iqr = q3 - q1
        mask = (intervals >= q1 - 2*iqr) & (intervals <= q3 + 2*iqr)
        filtered = intervals[mask]

        print(f"\n  Timing Analysis:")
        print(f"    Min interval: {np.min(intervals):.1f} samples")
        print(f"    Max interval: {np.max(intervals):.1f} samples")
        print(f"    Mean interval: {np.mean(intervals):.1f} samples")
        print(f"    Median interval: {np.median(intervals):.1f} samples")
        print(f"    Std deviation: {np.std(intervals):.1f} samples")

        # Find the two main clusters (half-bit and full-bit periods)
        hist, bin_edges = np.histogram(filtered, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist) * 0.5:
                peaks.append((hist[i], bin_centers[i]))

        peaks.sort(reverse=True)

        if len(peaks) >= 2:
            short_val = min(peaks[0][1], peaks[1][1])
            long_val = max(peaks[0][1], peaks[1][1])
            ratio = long_val / short_val if short_val > 0 else 0

            print(f"\n    Detected two timing peaks:")
            print(f"      Short (half-bit): {short_val:.1f} samples")
            print(f"      Long (full-bit): {long_val:.1f} samples")
            print(f"      Ratio: {ratio:.2f} (should be ~2.0)")

            self.half_bit_samples = short_val
            self.samples_per_bit = short_val * 2

            # Calculate frame rate
            # LTC has 80 bits per frame
            # samples_per_frame = samples_per_bit * 80
            samples_per_frame = self.samples_per_bit * 80
            fps = self.sample_rate / samples_per_frame

            print(f"\n    Calculated parameters:")
            print(f"      Samples per bit: {self.samples_per_bit:.1f}")
            print(f"      Samples per frame: {samples_per_frame:.1f}")
            print(f"      Estimated FPS: {fps:.3f}")

            # Expected values for common frame rates
            print(f"\n    Expected values at {self.sample_rate}Hz sample rate:")
            for expected_fps in [23.976, 24, 25, 29.97, 30]:
                expected_spf = self.sample_rate / expected_fps
                expected_spb = expected_spf / 80
                print(f"      {expected_fps:6.3f} fps: {expected_spb:.1f} samples/bit, {expected_spf:.1f} samples/frame")
        else:
            print("  WARNING: Could not detect two clear timing peaks")
            # Fallback
            self.half_bit_samples = np.median(filtered)
            self.samples_per_bit = self.half_bit_samples * 2

    def _decode_all_frames(self, samples, crossings):
        """Decode LTC frames from the audio."""
        if self.samples_per_bit == 0:
            print("  ERROR: Cannot decode - timing not detected")
            return

        bit_buffer = deque(maxlen=200)
        last_edge_was_half = False

        for i in range(1, len(crossings)):
            interval = crossings[i] - crossings[i-1]

            # Classify interval
            half_threshold_low = self.half_bit_samples * 0.5
            half_threshold_high = self.half_bit_samples * 1.5
            full_threshold_high = self.samples_per_bit * 1.5

            if half_threshold_low <= interval <= half_threshold_high:
                # Half-bit interval
                if last_edge_was_half:
                    bit_buffer.append(1)
                    last_edge_was_half = False
                else:
                    last_edge_was_half = True
            elif half_threshold_high < interval <= full_threshold_high:
                # Full-bit interval
                bit_buffer.append(0)
                last_edge_was_half = False
            else:
                # Out of range - reset
                last_edge_was_half = False
                continue

            # Try to find sync word and decode frame
            if len(bit_buffer) >= 80:
                frame = self._try_decode_frame(list(bit_buffer), crossings[i])
                if frame:
                    self.frames.append(frame)

    def _try_decode_frame(self, bits, sample_pos):
        """Try to find sync word and decode a frame."""
        # Look for sync word in the bit buffer
        for start in range(len(bits) - 79):
            # Check forward sync
            if bits[start:start+16] == self.SYNC_WORD:
                if start >= 64:
                    frame_bits = bits[start-64:start+16]
                    return self._decode_frame_bits(frame_bits, sample_pos, reversed_=False)
            # Check reverse sync
            if bits[start:start+16] == self.SYNC_WORD_REV:
                if start >= 64:
                    frame_bits = bits[start-64:start+16]
                    return self._decode_frame_bits(frame_bits, sample_pos, reversed_=True)
        return None

    def _decode_frame_bits(self, bits, sample_pos, reversed_=False):
        """Decode 80 bits into timecode values."""
        if len(bits) != 80:
            return None

        if reversed_:
            bits = bits[::-1]

        def get_bcd(bit_indices):
            val = 0
            for i, idx in enumerate(bit_indices):
                if bits[idx]:
                    val += (1 << i)
            return val

        try:
            frames_units = get_bcd([0,1,2,3])
            frames_tens = get_bcd([8,9])
            secs_units = get_bcd([16,17,18,19])
            secs_tens = get_bcd([24,25,26])
            mins_units = get_bcd([32,33,34,35])
            mins_tens = get_bcd([40,41,42])
            hours_units = get_bcd([48,49,50,51])
            hours_tens = get_bcd([56,57])

            drop_frame = bits[10]

            frames = frames_units + frames_tens * 10
            secs = secs_units + secs_tens * 10
            mins = mins_units + mins_tens * 10
            hours = hours_units + hours_tens * 10

            # Validate
            if frames > 30 or secs > 59 or mins > 59 or hours > 23:
                return None

            time_seconds = sample_pos / self.sample_rate

            return {
                'hours': hours,
                'mins': mins,
                'secs': secs,
                'frames': frames,
                'drop_frame': drop_frame,
                'sample_pos': sample_pos,
                'time_seconds': time_seconds,
                'tc_string': f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"
            }
        except:
            return None

    def write_report(self, output_path, input_file):
        """Write detailed report to text file."""
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("LTC AUDIO ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Input File: {input_file}\n")
            f.write(f"Sample Rate: {self.sample_rate} Hz\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-" * 70 + "\n")
            f.write("TIMING ANALYSIS\n")
            f.write("-" * 70 + "\n\n")

            if self.edge_intervals:
                intervals = np.array(self.edge_intervals)
                f.write(f"Total zero crossings: {len(self.all_edges)}\n")
                f.write(f"Min interval: {np.min(intervals):.1f} samples\n")
                f.write(f"Max interval: {np.max(intervals):.1f} samples\n")
                f.write(f"Mean interval: {np.mean(intervals):.1f} samples\n")
                f.write(f"Median interval: {np.median(intervals):.1f} samples\n")
                f.write(f"Std deviation: {np.std(intervals):.1f} samples\n\n")

                f.write(f"Detected half-bit period: {self.half_bit_samples:.1f} samples\n")
                f.write(f"Detected full-bit period: {self.samples_per_bit:.1f} samples\n\n")

                if self.samples_per_bit > 0:
                    samples_per_frame = self.samples_per_bit * 80
                    fps = self.sample_rate / samples_per_frame
                    f.write(f"Calculated samples per frame: {samples_per_frame:.1f}\n")
                    f.write(f"Calculated FPS: {fps:.3f}\n\n")

            f.write("-" * 70 + "\n")
            f.write("DECODED TIMECODE FRAMES\n")
            f.write("-" * 70 + "\n\n")

            if not self.frames:
                f.write("No valid LTC frames decoded.\n\n")
            else:
                f.write(f"Total frames decoded: {len(self.frames)}\n\n")
                f.write(f"{'#':>5}  {'Timecode':<12}  {'DF':>3}  {'Time (sec)':>12}  {'Sample Pos':>12}\n")
                f.write("-" * 55 + "\n")

                for i, frame in enumerate(self.frames):
                    df_str = "DF" if frame['drop_frame'] else "NDF"
                    f.write(f"{i+1:5d}  {frame['tc_string']:<12}  {df_str:>3}  "
                           f"{frame['time_seconds']:12.3f}  {frame['sample_pos']:12d}\n")

                # Summary
                f.write("\n" + "-" * 70 + "\n")
                f.write("SUMMARY\n")
                f.write("-" * 70 + "\n\n")

                if len(self.frames) >= 2:
                    first_tc = self.frames[0]['tc_string']
                    last_tc = self.frames[-1]['tc_string']
                    duration = self.frames[-1]['time_seconds'] - self.frames[0]['time_seconds']

                    f.write(f"First Timecode: {first_tc}\n")
                    f.write(f"Last Timecode: {last_tc}\n")
                    f.write(f"Duration analyzed: {duration:.2f} seconds\n")
                    f.write(f"Frames decoded: {len(self.frames)}\n")

                    # Calculate actual FPS from decoded frames
                    if duration > 0:
                        actual_fps = len(self.frames) / duration
                        f.write(f"Actual decode rate: {actual_fps:.2f} frames/sec\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        print(f"\nReport written to: {output_path}")


def extract_audio_channel(input_path, channel, output_path, sample_rate=48000):
    """Extract a single audio channel using FFmpeg."""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vn', '-af', f'pan=mono|c0=c{channel}',
        '-acodec', 'pcm_s16le', '-ar', str(sample_rate),
        output_path
    ]

    kwargs = {'capture_output': True, 'text': True}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

    result = subprocess.run(cmd, **kwargs)
    return result.returncode == 0


def read_wav(path):
    """Read WAV file and return samples and sample rate."""
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels())[:, 0]
        return data.astype(np.float32) / 32768.0, sr


def get_audio_info(path):
    """Get audio stream info using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-select_streams', 'a', path
    ]

    kwargs = {'capture_output': True, 'text': True}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

    result = subprocess.run(cmd, **kwargs)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        if streams:
            return {
                'channels': streams[0].get('channels', 2),
                'sample_rate': int(streams[0].get('sample_rate', 48000))
            }
    return {'channels': 2, 'sample_rate': 48000}


def main():
    print("\n" + "=" * 60)
    print("LTC Audio to Text Converter")
    print("=" * 60 + "\n")

    # Get input file
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = input("Enter audio/video file path: ").strip().strip('"')

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        return

    print(f"Input: {input_path}")

    # Get audio info
    info = get_audio_info(input_path)
    num_channels = info['channels']
    sample_rate = info['sample_rate']

    print(f"Audio channels: {num_channels}")
    print(f"Sample rate: {sample_rate} Hz")

    # Get channel to analyze
    if len(sys.argv) > 2:
        channel = int(sys.argv[2])
    else:
        if num_channels > 1:
            channel = input(f"Which channel to analyze? (0-{num_channels-1}, default=last): ").strip()
            channel = int(channel) if channel else num_channels - 1
        else:
            channel = 0

    print(f"Analyzing channel: {channel}")

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='ltc_analyze_')
    wav_path = os.path.join(temp_dir, 'channel.wav')

    try:
        # Extract channel
        print(f"\nExtracting channel {channel}...")
        if not extract_audio_channel(input_path, channel, wav_path, sample_rate):
            print("ERROR: Failed to extract audio channel")
            return

        # Read audio
        print("Reading audio...")
        samples, sr = read_wav(wav_path)
        print(f"  Loaded {len(samples)} samples ({len(samples)/sr:.2f} seconds)")

        # Analyze
        print("\nAnalyzing LTC signal...")
        analyzer = LTCAnalyzer(sr)
        analyzer.analyze_audio(samples)

        # Write report
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_ltc_report.txt")

        if len(sys.argv) > 3:
            output_path = sys.argv[3]

        analyzer.write_report(output_path, input_path)

        # Quick summary
        print(f"\n{'=' * 60}")
        print("QUICK SUMMARY")
        print(f"{'=' * 60}")

        if analyzer.frames:
            print(f"Frames decoded: {len(analyzer.frames)}")
            print(f"First TC: {analyzer.frames[0]['tc_string']}")
            print(f"Last TC: {analyzer.frames[-1]['tc_string']}")
            df = "Drop Frame" if analyzer.frames[0]['drop_frame'] else "Non-Drop Frame"
            print(f"Type: {df}")
        else:
            print("No valid timecode frames decoded!")
            print("Possible issues:")
            print("  - Wrong audio channel selected")
            print("  - LTC signal too weak or distorted")
            print("  - Not an LTC audio track")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nDone!")


if __name__ == '__main__':
    main()
