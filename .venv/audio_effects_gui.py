"""
Audio Effect Chain GUI (Frontend Only)
Modern interface for building audio effect chains

Installation:
pip install customtkinter
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
from typing import Dict, List, Any
from pathlib import Path

# Effect definitions with their parameters
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


class EffectCard(ctk.CTkFrame):
    """Visual card representing an effect in the chain"""

    def __init__(self, parent, effect_name: str, on_remove, on_select, on_move_up, on_move_down):
        super().__init__(parent, fg_color=("gray85", "gray25"), corner_radius=8)

        self.effect_name = effect_name
        self.on_remove = on_remove
        self.on_select = on_select
        self.parameters = {}

        # Initialize with default parameters
        for param, (min_val, max_val, default) in EFFECTS[effect_name].items():
            self.parameters[param] = default

        # Layout
        self.grid_columnconfigure(1, weight=1)

        # Effect name
        self.name_label = ctk.CTkLabel(
            self,
            text=effect_name,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.name_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")

        # Move up button
        self.up_btn = ctk.CTkButton(
            self,
            text="↑",
            width=30,
            command=lambda: on_move_up(self),
            fg_color=("gray75", "gray30")
        )
        self.up_btn.grid(row=1, column=0, padx=(10, 2), pady=5)

        # Move down button
        self.down_btn = ctk.CTkButton(
            self,
            text="↓",
            width=30,
            command=lambda: on_move_down(self),
            fg_color=("gray75", "gray30")
        )
        self.down_btn.grid(row=1, column=1, padx=2, pady=5)

        # Edit button
        edit_btn = ctk.CTkButton(
            self,
            text="Edit",
            width=60,
            command=lambda: on_select(self),
            fg_color=("#3b8ed0", "#1f6aa5")
        )
        edit_btn.grid(row=1, column=2, padx=5, pady=5)

        # Remove button
        remove_btn = ctk.CTkButton(
            self,
            text="Remove",
            width=70,
            command=lambda: on_remove(self),
            fg_color=("#d9534f", "#c9302c")
        )
        remove_btn.grid(row=1, column=3, padx=(5, 10), pady=5)


class AudioEffectChainGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Effect Chain Editor")
        self.geometry("1200x700")

        # Effect chain
        self.effect_chain: List[EffectCard] = []
        self.selected_card: EffectCard = None

        # Input file
        self.input_file = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ===== TOP BAR: File Selection =====
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        top_frame.grid_columnconfigure(1, weight=1)

        # Input file
        ctk.CTkLabel(top_frame, text="Input Audio:").grid(row=0, column=0, padx=5)
        self.input_label = ctk.CTkLabel(
            top_frame,
            text="No file selected - Click Browse to select your audio file",
            anchor="w",
            fg_color=("gray85", "gray25"),
            corner_radius=6
        )
        self.input_label.grid(row=0, column=1, padx=5, sticky="ew")
        ctk.CTkButton(
            top_frame,
            text="Browse Input",
            width=120,
            command=self.select_input_file
        ).grid(row=0, column=2, padx=5)

        # ===== LEFT PANEL: Available Effects =====
        left_panel = ctk.CTkFrame(self, width=200)
        left_panel.grid(row=1, column=0, padx=(10, 5), pady=(0, 10), sticky="nsew")
        left_panel.grid_propagate(False)

        ctk.CTkLabel(
            left_panel,
            text="Available Effects",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=10, pady=10)

        # Scrollable frame for effects
        scroll_frame = ctk.CTkScrollableFrame(left_panel)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Group effects by category
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
                scroll_frame,
                text=category,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w"
            ).pack(fill="x", padx=5, pady=(10, 5))

            for effect in effects:
                ctk.CTkButton(
                    scroll_frame,
                    text=f"+ {effect}",
                    command=lambda e=effect: self.add_effect(e),
                    anchor="w"
                ).pack(fill="x", padx=5, pady=2)

        # ===== MIDDLE PANEL: Effect Chain =====
        middle_panel = ctk.CTkFrame(self)
        middle_panel.grid(row=1, column=1, padx=5, pady=(0, 10), sticky="nsew")

        ctk.CTkLabel(
            middle_panel,
            text="Effect Chain (Signal Flow →)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=10, pady=10)

        self.chain_frame = ctk.CTkScrollableFrame(middle_panel)
        self.chain_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Empty state label
        self.empty_label = ctk.CTkLabel(
            self.chain_frame,
            text="No effects added yet.\nAdd effects from the left panel.",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.empty_label.pack(expand=True, pady=50)

        # ===== RIGHT PANEL: Parameters =====
        right_panel = ctk.CTkFrame(self, width=300)
        right_panel.grid(row=1, column=2, padx=(5, 10), pady=(0, 10), sticky="nsew")
        right_panel.grid_propagate(False)

        ctk.CTkLabel(
            right_panel,
            text="Effect Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=10, pady=10)

        self.params_frame = ctk.CTkScrollableFrame(right_panel)
        self.params_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.param_widgets = {}

        # Default message
        self.params_empty_label = ctk.CTkLabel(
            self.params_frame,
            text="Select an effect to edit\nits parameters",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.params_empty_label.pack(expand=True, pady=50)

        # ===== BOTTOM BAR: Process Button =====
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="ew")

        self.process_btn = ctk.CTkButton(
            bottom_frame,
            text="🎵 Process Audio",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            command=self.process_audio,
            fg_color=("#2ecc71", "#27ae60")
        )
        self.process_btn.pack(fill="x", padx=20)

    def select_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.aiff"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.input_file = filename
            self.input_label.configure(text=Path(filename).name)

    def add_effect(self, effect_name: str):
        """Add an effect to the chain"""
        card = EffectCard(
            self.chain_frame,
            effect_name,
            on_remove=self.remove_effect,
            on_select=self.select_effect,
            on_move_up=self.move_effect_up,
            on_move_down=self.move_effect_down
        )
        card.pack(fill="x", padx=5, pady=5)
        self.effect_chain.append(card)

        # Hide empty label
        if self.empty_label.winfo_exists():
            self.empty_label.pack_forget()

    def remove_effect(self, card: EffectCard):
        """Remove an effect from the chain"""
        if card in self.effect_chain:
            self.effect_chain.remove(card)
            card.destroy()

            # Clear parameters if this was selected
            if self.selected_card == card:
                self.clear_parameters()

            # Show empty label if no effects
            if not self.effect_chain:
                self.empty_label.pack(expand=True, pady=50)

    def move_effect_up(self, card: EffectCard):
        """Move effect up in the chain"""
        idx = self.effect_chain.index(card)
        if idx > 0:
            self.effect_chain[idx], self.effect_chain[idx-1] = self.effect_chain[idx-1], self.effect_chain[idx]
            self.refresh_chain_display()

    def move_effect_down(self, card: EffectCard):
        """Move effect down in the chain"""
        idx = self.effect_chain.index(card)
        if idx < len(self.effect_chain) - 1:
            self.effect_chain[idx], self.effect_chain[idx+1] = self.effect_chain[idx+1], self.effect_chain[idx]
            self.refresh_chain_display()

    def refresh_chain_display(self):
        """Refresh the visual order of effects"""
        for card in self.effect_chain:
            card.pack_forget()
        for card in self.effect_chain:
            card.pack(fill="x", padx=5, pady=5)

    def select_effect(self, card: EffectCard):
        """Select an effect to edit its parameters"""
        self.selected_card = card
        self.display_parameters(card)

    def clear_parameters(self):
        """Clear the parameters panel"""
        for widget in list(self.params_frame.winfo_children()):
            widget.destroy()
        self.param_widgets.clear()
        self.selected_card = None

        self.params_empty_label = ctk.CTkLabel(
            self.params_frame,
            text="Select an effect to edit\nits parameters",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.params_empty_label.pack(expand=True, pady=50)

    def display_parameters(self, card: EffectCard):
        """Display parameters for the selected effect"""
        # Clear existing parameters
        for widget in list(self.params_frame.winfo_children()):
            widget.destroy()
        self.param_widgets.clear()

        # Show effect name
        title = ctk.CTkLabel(
            self.params_frame,
            text=f"🎛️ {card.effect_name}",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(pady=(0, 20))

        # Create parameter controls
        params = EFFECTS[card.effect_name]

        for param_name, (min_val, max_val, default) in params.items():
            frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
            frame.pack(fill="x", pady=10)

            # Parameter name
            label = ctk.CTkLabel(
                frame,
                text=param_name.replace("_", " ").title(),
                font=ctk.CTkFont(size=12)
            )
            label.pack(anchor="w")

            # Value display
            value_label = ctk.CTkLabel(
                frame,
                text=f"{card.parameters[param_name]:.2f}",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            value_label.pack(anchor="w")

            # Slider
            slider = ctk.CTkSlider(
                frame,
                from_=min_val,
                to=max_val,
                command=lambda val, p=param_name, vl=value_label, c=card: self.update_parameter(c, p, val, vl)
            )
            slider.set(card.parameters[param_name])
            slider.pack(fill="x", pady=(5, 0))

            self.param_widgets[param_name] = {
                "frame": frame,
                "slider": slider,
                "value_label": value_label
            }

    def update_parameter(self, card: EffectCard, param_name: str, value: float, value_label: ctk.CTkLabel):
        """Update a parameter value"""
        card.parameters[param_name] = value
        value_label.configure(text=f"{value:.2f}")

    def process_audio(self):
        """Process the audio with the configured effect chain"""
        # Validation
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file")
            return

        if not self.effect_chain:
            messagebox.showwarning("Warning", "No effects in the chain. Audio will be copied as-is.")

        # Prepare configuration
        config = {
            "input_file": self.input_file,
            "output_file": str(Path(self.input_file).parent / f"{Path(self.input_file).stem}_processed{Path(self.input_file).suffix}"),
            "effects": []
        }

        for card in self.effect_chain:
            effect_config = {
                "name": card.effect_name,
                "parameters": card.parameters.copy()
            }
            config["effects"].append(effect_config)

        # Show configuration (in production, this would call your backend)
        print("\n" + "="*50)
        print("AUDIO PROCESSING CONFIGURATION")
        print("="*50)
        print(f"Input: {config['input_file']}")
        print(f"Output: {config['output_file']}")
        print(f"\nEffect Chain ({len(config['effects'])} effects):")
        for i, effect in enumerate(config['effects'], 1):
            print(f"\n{i}. {effect['name']}")
            for param, value in effect['parameters'].items():
                print(f"   - {param}: {value:.2f}")
        print("="*50 + "\n")

        # Show success message
        messagebox.showinfo(
            "Ready to Process",
            f"Configuration ready!\n\n"
            f"Input: {Path(self.input_file).name}\n"
            f"Output: {Path(config['output_file']).name}\n"
            f"Effects: {len(config['effects'])}\n\n"
            f"Check console for full configuration.\n"
            f"Now integrate with your Pedalboard backend!"
        )

        # TODO: Call your backend here
        # from audio_backend import AudioProcessor
        # AudioProcessor.process_audio(config)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = AudioEffectChainGUI()
    app.mainloop()