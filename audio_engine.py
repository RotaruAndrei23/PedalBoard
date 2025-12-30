"""
audio_engine.py
Backend logic for Real-Time Audio Application.
Handles Pedalboard effects, SoundDevice streams, and audio processing.
"""

import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Dict, List, Optional

# Audio processing imports
from pedalboard import Pedalboard
from pedalboard import (
    Chorus, Distortion, Phaser, Clipping,
    Compressor, Gain, Limiter,
    HighpassFilter, LowpassFilter,
    Delay, Reverb, PitchShift, Bitcrush
)

# Effect definitions - Adjusted for cleaner default sound
EFFECTS = {
    "Chorus": {
        "rate_hz": (0.1, 10.0, 1.0),
        "depth": (0.0, 1.0, 0.25),
        "centre_delay_ms": (1.0, 50.0, 7.0),
        "feedback": (0.0, 1.0, 0.0),
        "mix": (0.0, 1.0, 0.5)
    },
    "Distortion": {
        "drive_db": (0.0, 40.0, 10.0)
    },
    "Phaser": {
        "rate_hz": (0.1, 10.0, 1.0),
        "depth": (0.0, 1.0, 0.5),
        "centre_frequency_hz": (200.0, 12000.0, 1300.0),
        "feedback": (0.0, 1.0, 0.0),
        "mix": (0.0, 1.0, 0.5)
    },
    "Clipping": {
        "threshold_db": (-60.0, 0.0, -6.0)
    },
    "Compressor": {
        "threshold_db": (-60.0, 0.0, -20.0),
        "ratio": (1.0, 20.0, 4.0),
        "attack_ms": (0.1, 100.0, 5.0),
        "release_ms": (10.0, 1000.0, 100.0)
    },
    "Gain": {
        "gain_db": (-60.0, 60.0, 0.0)
    },
    "Limiter": {
        "threshold_db": (-60.0, 0.0, -3.0),
        "release_ms": (10.0, 1000.0, 100.0)
    },
    "HighpassFilter": {
        "cutoff_frequency_hz": (20.0, 20000.0, 440.0)
    },
    "LowpassFilter": {
        "cutoff_frequency_hz": (20.0, 20000.0, 5000.0)
    },
    "Delay": {
        "delay_seconds": (0.0, 2.0, 0.5),
        "feedback": (0.0, 1.0, 0.3),
        "mix": (0.0, 1.0, 0.5)
    },
    "Reverb": {
        "room_size": (0.0, 1.0, 0.5),
        "damping": (0.0, 1.0, 0.5),
        "wet_level": (0.0, 1.0, 0.33),
        "dry_level": (0.0, 1.0, 0.7),
        "width": (0.0, 1.0, 1.0)
    },
    "PitchShift": {
        "semitones": (-12.0, 12.0, 0.0)
    },
    "Bitcrush": {
        "bit_depth": (1.0, 32.0, 8.0)
    }
}


class RealTimeAudioProcessor:
    """Real-time audio processor with live effects"""

    EFFECT_MAP = {
        "Chorus": Chorus, "Distortion": Distortion, "Phaser": Phaser,
        "Clipping": Clipping, "Compressor": Compressor, "Gain": Gain,
        "Limiter": Limiter, "HighpassFilter": HighpassFilter,
        "LowpassFilter": LowpassFilter, "Delay": Delay, "Reverb": Reverb,
        "PitchShift": PitchShift, "Bitcrush": Bitcrush
    }

    def __init__(self, samplerate=44100):
        self.samplerate = samplerate
        self.board = Pedalboard([])
        # Safety limiter prevents clipping at the output stage
        self.safety_limiter = Limiter(threshold_db=-1.0, release_ms=100.0)
        self.lock = threading.Lock()
        self.is_active = False
        self.recorded_chunks = []

    def update_effects(self, effect_configs: List[Dict]):
        """Thread-safe effect chain update"""
        with self.lock:
            new_board = Pedalboard([])
            for effect_config in effect_configs:
                effect_class = self.EFFECT_MAP[effect_config["name"]]
                effect = effect_class(**effect_config["parameters"])
                new_board.append(effect)

            self.board = new_board

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process a chunk of audio through the effect chain"""
        with self.lock:
            # Ensure input is float32
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            if len(self.board) > 0:
                try:
                    # reset=False keeps delay/reverb trails between chunks
                    processed = self.board(audio_chunk.T, self.samplerate, reset=False)
                    processed = processed.T

                    # Apply safety limiter
                    processed = self.safety_limiter(processed, self.samplerate)

                    return processed
                except Exception as e:
                    print(f"Processing error: {e}")
                    return audio_chunk
            return audio_chunk

    def save_recording(self, filename: str) -> bool:
        """Save all recorded chunks to file"""
        if self.recorded_chunks:
            try:
                audio = np.concatenate(self.recorded_chunks, axis=0)
                sf.write(filename, audio, self.samplerate)
                return True
            except Exception as e:
                print(f"Save error: {e}")
                return False
        return False


class LiveRecordingSession:
    """Manages live recording with real-time effects"""

    def __init__(self, processor: RealTimeAudioProcessor):
        self.processor = processor
        self.stream = None
        self.is_running = False

    def start(self):
        """Start live recording with real-time effects"""
        self.is_running = True
        self.processor.is_active = True
        self.processor.recorded_chunks = []

        def audio_callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"Status: {status}")

            if self.is_running:
                processed = self.processor.process_chunk(indata.copy())
                self.processor.recorded_chunks.append(processed.copy())
                outdata[:] = processed
            else:
                outdata[:] = np.zeros_like(indata)

        self.stream = sd.Stream(
            samplerate=self.processor.samplerate,
            channels=2,
            callback=audio_callback,
            dtype='float32'
        )
        self.stream.start()

    def stop(self):
        """Stop live recording"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.processor.is_active = False


class FilePlaybackSession:
    """Manages file playback with real-time effects"""

    def __init__(self, processor: RealTimeAudioProcessor, audio_file: str):
        self.processor = processor
        self.audio_file = audio_file
        self.stream = None
        self.is_running = False
        self.audio_data = None
        self.current_frame = 0

    def load_file(self):
        """Load audio file with safety checks for format and volume"""
        try:
            # Read file always as float32
            data, samplerate = sf.read(self.audio_file, always_2d=True, dtype='float32')

            # --- FIX: NORMALIZE AUDIO IF IT CLIPS ---
            # Check maximum amplitude
            max_val = np.max(np.abs(data))

            # If the file is louder than 0dB (1.0), normalize it down
            # This leaves headroom for effects like distortion or reverb to add volume
            if max_val > 1.0:
                print(f"Warning: Audio file clipped (max: {max_val:.2f}). Normalizing...")
                data = data / max_val * 0.5
            elif max_val > 0.8:
                # Even if it's not technically clipping, if it's very loud,
                # reduce it slightly to -3dB to give headroom for effects.
                data = data / max_val * 0.5

            self.audio_data = data
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False

    def start(self):
        """Start playback with real-time effects"""
        if not self.load_file():
            return False

        self.is_running = True
        self.processor.is_active = True
        self.processor.recorded_chunks = []
        self.current_frame = 0

        def audio_callback(outdata, frames, time_info, status):
            if status:
                print(f"Status: {status}")

            if self.is_running and self.current_frame < len(self.audio_data):
                end_frame = min(self.current_frame + frames, len(self.audio_data))
                chunk = self.audio_data[self.current_frame:end_frame]

                if len(chunk) < frames:
                    chunk = np.pad(chunk, ((0, frames - len(chunk)), (0, 0)))

                processed = self.processor.process_chunk(chunk)
                self.processor.recorded_chunks.append(processed[:end_frame - self.current_frame].copy())

                outdata[:] = processed
                self.current_frame = end_frame

                if self.current_frame >= len(self.audio_data):
                    self.is_running = False
            else:
                outdata[:] = np.zeros((frames, 2))
                if self.current_frame >= len(self.audio_data):
                    self.is_running = False

        self.stream = sd.OutputStream(
            samplerate=self.processor.samplerate,
            channels=2,
            callback=audio_callback,
            dtype='float32'
        )
        self.stream.start()
        return True

    def stop(self):
        """Stop playback"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.processor.is_active = False