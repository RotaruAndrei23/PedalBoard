"""
app.py
GUI Frontend for Real-Time Audio Effect Chain Application.
Imports logic from audio_engine.py.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List, Optional
import numpy as np

# Import backend logic
from audio_engine import (
    EFFECTS,
    RealTimeAudioProcessor,
    LiveRecordingSession,
    FilePlaybackSession
)

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

        if processor.recorded_chunks:
            duration = len(np.concatenate(processor.recorded_chunks)) / processor.samplerate
        else:
            duration = 0.0

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

        # Initialize default parameters from the backend dictionary
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
        self.control_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        self.control_frame.grid(row=0, column=3, padx=5)

        # Change File Button (Hidden initially)
        self.change_file_btn = ctk.CTkButton(
            self.control_frame,
            text="📂 Change File",
            width=100,
            command=self.browse_file,
            fg_color=("#3b8ed0", "#1f6aa5"),
            font=ctk.CTkFont(size=13)
        )
        # Not packed initially

        # Start/Stop button
        self.start_btn = ctk.CTkButton(
            self.control_frame,
            text="🎤 Start Recording",
            width=150,
            command=self.toggle_audio,
            fg_color=("#2ecc71", "#27ae60"),
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.start_btn.pack(side="left", padx=2)

        # Download button (hidden initially)
        self.download_btn = ctk.CTkButton(
            self.control_frame,
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
            self.change_file_btn.pack_forget()
            self.status_label.configure(text="🎤 Ready to record with live effects")
            self.start_btn.configure(text="🎤 Start Recording")
        else:  # Play File
            # Show change file button BEFORE start button
            self.change_file_btn.pack(side="left", padx=(0, 5), before=self.start_btn)

            if self.input_file:
                self.status_label.configure(text=f"📁 Ready to play: {Path(self.input_file).name}")
                self.start_btn.configure(text="▶ Start Playback")
            else:
                self.status_label.configure(text="📁 Select a file to play with live effects")
                self.start_btn.configure(text="📁 Select File")

    def browse_file(self):
        """Open file dialog to select input file"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
        )
        if filename:
            self.input_file = filename
            self.status_label.configure(text=f"📁 Ready to play: {Path(self.input_file).name}")
            self.start_btn.configure(text="▶ Start Playback")

            # If we were playing, stop
            if self.is_active:
                self.stop_audio()

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
            self.change_file_btn.configure(state="disabled")

        else:  # Play File
            if not self.input_file:
                self.browse_file()
                if not self.input_file:
                    return

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
                self.change_file_btn.configure(state="disabled")

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
            self.change_file_btn.configure(state="normal")
        else:
            self.status_label.configure(text=f"✅ Playback stopped - Ready: {Path(self.input_file).name}")
            self.start_btn.configure(
                text="▶ Play Again",
                fg_color=("#2ecc71", "#27ae60")
            )
            self.change_file_btn.configure(state="normal")

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