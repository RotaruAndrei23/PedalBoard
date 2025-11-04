"""
Real-Time Audio Effect Chain Application
Live recording and playback with real-time effects + download capability

Installation:
pip install customtkinter pedalboard sounddevice soundfile numpy
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import tempfile
import os
import queue
import time

# Audio processing imports
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from pedalboard import (
    Chorus, Distortion, Phaser, Clipping,
    Compressor, Gain, Limiter,
    HighpassFilter, LowpassFilter,
    Delay, Reverb, PitchShift, Bitcrush
)

# Real-time audio
import sounddevice as sd
import soundfile as sf
import numpy as np


# Effect definitions
EFFECTS = {
    "Chorus": {
        "rate_hz": (0.1, 10.0, 1.0),
        "depth": (0.0, 1.0, 0.25),
        "centre_delay_ms": (1.0, 50.0, 7.0),
        "feedback": (0.0, 1.0, 0.0),
        "mix": (0.0, 1.0, 0.5)
    },
    "Distortion": {
        "drive_db": (0.0, 40.0, 25.0)
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
        "threshold_db": (-60.0, 0.0, -10.0),
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
        "feedback": (0.0, 1.0, 0.0),
        "mix": (0.0, 1.0, 0.5)
    },
    "Reverb": {
        "room_size": (0.0, 1.0, 0.5),
        "damping": (0.0, 1.0, 0.5),
        "wet_level": (0.0, 1.0, 0.33),
        "dry_level": (0.0, 1.0, 0.4),
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
        self.lock = threading.Lock()
        self.is_active = False
        self.recorded_chunks = []

    def update_effects(self, effect_configs: List[Dict]):
        """Thread-safe effect chain update"""
        with self.lock:
            self.board = Pedalboard([])
            for effect_config in effect_configs:
                effect_class = self.EFFECT_MAP[effect_config["name"]]
                effect = effect_class(**effect_config["parameters"])
                self.board.append(effect)

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process a chunk of audio through the effect chain"""
        with self.lock:
            if len(self.board) > 0:
                try:
                    processed = self.board(audio_chunk.T, self.samplerate)
                    return processed.T
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
                # Process input through effects
                processed = self.processor.process_chunk(indata.copy())

                # Store for later download
                self.processor.recorded_chunks.append(processed.copy())

                # Output to speakers
                outdata[:] = processed
            else:
                outdata[:] = np.zeros_like(indata)

        self.stream = sd.Stream(
            samplerate=self.processor.samplerate,
            channels=2,
            callback=audio_callback
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
        """Load audio file"""
        try:
            self.audio_data, _ = sf.read(self.audio_file, always_2d=True)
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
                # Get next chunk from file
                end_frame = min(self.current_frame + frames, len(self.audio_data))
                chunk = self.audio_data[self.current_frame:end_frame]

                # Pad if necessary
                if len(chunk) < frames:
                    chunk = np.pad(chunk, ((0, frames - len(chunk)), (0, 0)))

                # Process through effects
                processed = self.processor.process_chunk(chunk)

                # Store for later download
                self.processor.recorded_chunks.append(processed[:end_frame - self.current_frame].copy())

                # Output
                outdata[:] = processed
                self.current_frame = end_frame

                # Stop when done
                if self.current_frame >= len(self.audio_data):
                    self.is_running = False
            else:
                outdata[:] = np.zeros((frames, 2))
                if self.current_frame >= len(self.audio_data):
                    self.is_running = False

        self.stream = sd.OutputStream(
            samplerate=self.processor.samplerate,
            channels=2,
            callback=audio_callback
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


class DownloadDialog(ctk.CTkToplevel):
    """Dialog to download processed audio"""

    def __init__(self, parent, processor: RealTimeAudioProcessor):
        super().__init__(parent)

        self.processor = processor

        self.title("Download Processed Audio")
        self.geometry("400x200")
        self.resizable(False, False)

        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 200) // 2
        self.geometry(f"+{x}+{y}")

        # Message
        ctk.CTkLabel(
            self,
            text="💾 Save Processed Audio",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#2ecc71"
        ).pack(pady=(20, 10))

        duration = len(np.concatenate(processor.recorded_chunks)) / processor.samplerate
        ctk.CTkLabel(
            self,
            text=f"Duration: {duration:.1f} seconds",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=5)

        # Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=20, fill="x", padx=20)

        ctk.CTkButton(
            button_frame,
            text="💾 Save As...",
            font=ctk.CTkFont(size=14),
            height=40,
            command=self.save_file,
            fg_color=("#2ecc71", "#27ae60")
        ).pack(side="left", expand=True, fill="x", padx=(0, 5))

        ctk.CTkButton(
            button_frame,
            text="Cancel",
            font=ctk.CTkFont(size=14),
            height=40,
            command=self.destroy,
            fg_color=("gray70", "gray30")
        ).pack(side="left", expand=True, fill="x", padx=(5, 0))

    def save_file(self):
        save_path = filedialog.asksaveasfilename(
            title="Save Processed Audio",
            defaultextension=".wav",
            initialfile="processed_audio.wav",
            filetypes=[
                ("WAV File", "*.wav"),
                ("FLAC File", "*.flac"),
                ("All Files", "*.*")
            ]
        )

        if save_path:
            if self.processor.save_recording(save_path):
                messagebox.showinfo("Success", f"Audio saved to:\n{save_path}")
                self.destroy()
            else:
                messagebox.showerror("Error", "Failed to save audio file")


class EffectCard(ctk.CTkFrame):
    """Visual card for an effect in the chain"""

    def __init__(self, parent, effect_name: str, on_remove, on_select, on_move_up, on_move_down):
        super().__init__(parent, fg_color=("gray85", "gray25"), corner_radius=8)

        self.effect_name = effect_name
        self.parameters = {}

        # Initialize default parameters
        for param, (min_val, max_val, default) in EFFECTS[effect_name].items():
            self.parameters[param] = default

        self.grid_columnconfigure(1, weight=1)

        # Effect name
        ctk.CTkLabel(
            self, text=effect_name,
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, columnspan=4, padx=10, pady=(10, 5), sticky="w")

        # Buttons
        ctk.CTkButton(
            self, text="↑", width=30, command=lambda: on_move_up(self),
            fg_color=("gray75", "gray30")
        ).grid(row=1, column=0, padx=(10, 2), pady=5)

        ctk.CTkButton(
            self, text="↓", width=30, command=lambda: on_move_down(self),
            fg_color=("gray75", "gray30")
        ).grid(row=1, column=1, padx=2, pady=5)

        ctk.CTkButton(
            self, text="Edit", width=60, command=lambda: on_select(self),
            fg_color=("#3b8ed0", "#1f6aa5")
        ).grid(row=1, column=2, padx=5, pady=5)

        ctk.CTkButton(
            self, text="Remove", width=70, command=lambda: on_remove(self),
            fg_color=("#d9534f", "#c9302c")
        ).grid(row=1, column=3, padx=(5, 10), pady=5)


class RealTimeAudioApp(ctk.CTk):
    """Main application window"""

    def __init__(self):
        super().__init__()

        self.title("Real-Time Audio Effect Processor")
        self.geometry("1200x750")

        self.effect_chain: List[EffectCard] = []
        self.selected_card: Optional[EffectCard] = None
        self.input_file = None

        # Audio processing
        self.processor = RealTimeAudioProcessor()
        self.live_session: Optional[LiveRecordingSession] = None
        self.playback_session: Optional[FilePlaybackSession] = None
        self.is_active = False

        self.setup_ui()
        self.update_effect_chain()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top bar: Input mode and controls
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        top_frame.grid_columnconfigure(2, weight=1)

        # Mode selector
        ctk.CTkLabel(top_frame, text="Mode:").grid(row=0, column=0, padx=5, sticky="w")
        self.mode_selector = ctk.CTkSegmentedButton(
            top_frame,
            values=["Record Live", "Play File"],
            command=self.on_mode_change
        )
        self.mode_selector.set("Record Live")
        self.mode_selector.grid(row=0, column=1, padx=5, sticky="w")

        # Status label
        self.status_label = ctk.CTkLabel(
            top_frame,
            text="🎤 Ready to record with live effects",
            anchor="w",
            fg_color=("gray85", "gray25"),
            corner_radius=6,
            font=ctk.CTkFont(size=13)
        )
        self.status_label.grid(row=0, column=2, padx=5, sticky="ew")

        # Control buttons frame
        control_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        control_frame.grid(row=0, column=3, padx=5)

        # Start/Stop button
        self.start_btn = ctk.CTkButton(
            control_frame,
            text="🎤 Start Recording",
            width=150,
            command=self.toggle_audio,
            fg_color=("#2ecc71", "#27ae60"),
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.start_btn.pack(side="left", padx=2)

        # Download button (hidden initially)
        self.download_btn = ctk.CTkButton(
            control_frame,
            text="💾 Download",
            width=120,
            command=self.download_audio,
            fg_color=("#3b8ed0", "#1f6aa5"),
            font=ctk.CTkFont(size=13, weight="bold")
        )

        # Left panel: Available effects
        left_panel = ctk.CTkFrame(self, width=200)
        left_panel.grid(row=1, column=0, padx=(10, 5), pady=(0, 10), sticky="nsew")
        left_panel.grid_propagate(False)

        ctk.CTkLabel(
            left_panel, text="Available Effects",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=10, pady=10)

        scroll_frame = ctk.CTkScrollableFrame(left_panel)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        categories = {
            "Guitar Effects": ["Chorus", "Distortion", "Phaser", "Clipping"],
            "Dynamics": ["Compressor", "Gain", "Limiter"],
            "Filters": ["HighpassFilter", "LowpassFilter"],
            "Spatial": ["Delay", "Reverb"],
            "Pitch": ["PitchShift"],
            "Quality": ["Bitcrush"]
        }

        for category, effects in categories.items():
            ctk.CTkLabel(
                scroll_frame, text=category,
                font=ctk.CTkFont(size=12, weight="bold"), anchor="w"
            ).pack(fill="x", padx=5, pady=(10, 5))

            for effect in effects:
                ctk.CTkButton(
                    scroll_frame, text=f"+ {effect}",
                    command=lambda e=effect: self.add_effect(e), anchor="w"
                ).pack(fill="x", padx=5, pady=2)

        # Middle panel: Effect chain
        middle_panel = ctk.CTkFrame(self)
        middle_panel.grid(row=1, column=1, padx=5, pady=(0, 10), sticky="nsew")

        header_frame = ctk.CTkFrame(middle_panel, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            header_frame, text="Effect Chain (Signal Flow →)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left")

        # Live indicator
        self.live_indicator = ctk.CTkLabel(
            header_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#e74c3c"
        )
        self.live_indicator.pack(side="right", padx=10)

        self.chain_frame = ctk.CTkScrollableFrame(middle_panel)
        self.chain_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.empty_label = ctk.CTkLabel(
            self.chain_frame,
            text="No effects added.\nAdd effects from the left - they'll apply in real-time!",
            font=ctk.CTkFont(size=14), text_color="gray"
        )
        self.empty_label.pack(expand=True, pady=50)

        # Right panel: Parameters
        right_panel = ctk.CTkFrame(self, width=300)
        right_panel.grid(row=1, column=2, padx=(5, 10), pady=(0, 10), sticky="nsew")
        right_panel.grid_propagate(False)

        ctk.CTkLabel(
            right_panel, text="Effect Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=10, pady=10)

        ctk.CTkLabel(
            right_panel, text="Changes apply instantly!",
            font=ctk.CTkFont(size=11), text_color="#2ecc71"
        ).pack(padx=10, pady=(0, 5))

        self.params_frame = ctk.CTkScrollableFrame(right_panel)
        self.params_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.param_widgets = {}
        self.params_empty_label = ctk.CTkLabel(
            self.params_frame,
            text="Select an effect to edit\nits parameters",
            font=ctk.CTkFont(size=14), text_color="gray"
        )
        self.params_empty_label.pack(expand=True, pady=50)

    def on_mode_change(self, value):
        """Handle mode change"""
        if self.is_active:
            self.stop_audio()

        if value == "Record Live":
            self.status_label.configure(text="🎤 Ready to record with live effects")
            self.start_btn.configure(text="🎤 Start Recording")
        else:  # Play File
            self.status_label.configure(text="📁 Select a file to play with live effects")
            self.start_btn.configure(text="📁 Select File")

    def toggle_audio(self):
        """Toggle audio processing"""
        if self.is_active:
            self.stop_audio()
        else:
            self.start_audio()

    def start_audio(self):
        """Start audio processing"""
        mode = self.mode_selector.get()

        if mode == "Record Live":
            # Start live recording
            self.live_session = LiveRecordingSession(self.processor)
            self.live_session.start()
            self.is_active = True

            self.status_label.configure(text="🔴 LIVE - Recording with effects...")
            self.start_btn.configure(
                text="⏹ Stop Recording",
                fg_color=("#e74c3c", "#c0392b")
            )
            self.live_indicator.configure(text="● LIVE")
            self.mode_selector.configure(state="disabled")

        else:  # Play File
            if not self.input_file:
                # Select file
                filename = filedialog.askopenfilename(
                    title="Select Audio File",
                    filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
                )
                if not filename:
                    return
                self.input_file = filename

            # Start playback
            self.playback_session = FilePlaybackSession(self.processor, self.input_file)
            if self.playback_session.start():
                self.is_active = True

                self.status_label.configure(text=f"▶ Playing: {Path(self.input_file).name}")
                self.start_btn.configure(
                    text="⏹ Stop Playback",
                    fg_color=("#e74c3c", "#c0392b")
                )
                self.live_indicator.configure(text="● LIVE")
                self.mode_selector.configure(state="disabled")

                # Monitor playback completion
                self.monitor_playback()
            else:
                messagebox.showerror("Error", "Failed to load audio file")

    def stop_audio(self):
        """Stop audio processing"""
        if self.live_session:
            self.live_session.stop()
            self.live_session = None

        if self.playback_session:
            self.playback_session.stop()
            self.playback_session = None

        self.is_active = False

        # Reset UI
        mode = self.mode_selector.get()
        if mode == "Record Live":
            self.status_label.configure(text="✅ Recording stopped - Ready to download or start new")
            self.start_btn.configure(
                text="🎤 Start Recording",
                fg_color=("#2ecc71", "#27ae60")
            )
        else:
            self.status_label.configure(text="✅ Playback stopped - Ready to download")
            self.start_btn.configure(
                text="▶ Play Again",
                fg_color=("#2ecc71", "#27ae60")
            )

        self.live_indicator.configure(text="")
        self.mode_selector.configure(state="normal")

        # Show download button if we have audio
        if self.processor.recorded_chunks:
            self.download_btn.pack(side="left", padx=2)

    def monitor_playback(self):
        """Monitor playback session for completion"""
        if self.playback_session and not self.playback_session.is_running:
            self.stop_audio()
        elif self.is_active:
            self.after(100, self.monitor_playback)

    def download_audio(self):
        """Download processed audio"""
        if self.processor.recorded_chunks:
            DownloadDialog(self, self.processor)

    def add_effect(self, effect_name: str):
        card = EffectCard(
            self.chain_frame, effect_name,
            on_remove=self.remove_effect, on_select=self.select_effect,
            on_move_up=self.move_effect_up, on_move_down=self.move_effect_down
        )
        card.pack(fill="x", padx=5, pady=5)
        self.effect_chain.append(card)
        if self.empty_label.winfo_exists():
            self.empty_label.pack_forget()

        self.update_effect_chain()

    def remove_effect(self, card: EffectCard):
        if card in self.effect_chain:
            self.effect_chain.remove(card)
            card.destroy()

            if self.selected_card == card:
                self.clear_parameters()

            if not self.effect_chain:
                self.empty_label.pack(expand=True, pady=50)

            self.update_effect_chain()

    def move_effect_up(self, card: EffectCard):
        idx = self.effect_chain.index(card)
        if idx > 0:
            self.effect_chain[idx], self.effect_chain[idx-1] = self.effect_chain[idx-1], self.effect_chain[idx]
            self.refresh_chain_display()
            self.update_effect_chain()

    def move_effect_down(self, card: EffectCard):
        idx = self.effect_chain.index(card)
        if idx < len(self.effect_chain) - 1:
            self.effect_chain[idx], self.effect_chain[idx+1] = self.effect_chain[idx+1], self.effect_chain[idx]
            self.refresh_chain_display()
            self.update_effect_chain()

    def refresh_chain_display(self):
        for card in self.effect_chain:
            card.pack_forget()
        for card in self.effect_chain:
            card.pack(fill="x", padx=5, pady=5)

    def update_effect_chain(self):
        """Update processor with current effect chain"""
        config = [
            {"name": card.effect_name, "parameters": card.parameters.copy()}
            for card in self.effect_chain
        ]
        self.processor.update_effects(config)

    def select_effect(self, card: EffectCard):
        self.selected_card = card
        self.display_parameters(card)

    def clear_parameters(self):
        for widget in list(self.params_frame.winfo_children()):
            widget.destroy()

        self.param_widgets.clear()
        self.selected_card = None

        self.params_empty_label = ctk.CTkLabel(
            self.params_frame,
            text="Select an effect to edit\nits parameters",
            font=ctk.CTkFont(size=14), text_color="gray"
        )
        self.params_empty_label.pack(expand=True, pady=50)

    def display_parameters(self, card: EffectCard):
        for widget in list(self.params_frame.winfo_children()):
            widget.destroy()

        self.param_widgets.clear()

        ctk.CTkLabel(
            self.params_frame,
            text=f"🎛️ {card.effect_name}",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 20))

        params = EFFECTS[card.effect_name]
        for param_name, (min_val, max_val, default) in params.items():
            frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
            frame.pack(fill="x", pady=10)

            ctk.CTkLabel(
                frame, text=param_name.replace("_", " ").title(),
                font=ctk.CTkFont(size=12)
            ).pack(anchor="w")

            value_label = ctk.CTkLabel(
                frame, text=f"{card.parameters[param_name]:.2f}",
                font=ctk.CTkFont(size=11), text_color="gray"
            )
            value_label.pack(anchor="w")

            slider = ctk.CTkSlider(
                frame, from_=min_val, to=max_val,
                command=lambda val, p=param_name, vl=value_label, c=card:
                    self.update_parameter(c, p, val, vl)
            )
            slider.set(card.parameters[param_name])
            slider.pack(fill="x", pady=(5, 0))

            self.param_widgets[param_name] = {
                "frame": frame, "slider": slider, "value_label": value_label
            }

    def update_parameter(self, card: EffectCard, param_name: str, value: float, value_label: ctk.CTkLabel):
        card.parameters[param_name] = value
        value_label.configure(text=f"{value:.2f}")

        # Update effect chain in real-time
        self.update_effect_chain()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = RealTimeAudioApp()
    app.mainloop()