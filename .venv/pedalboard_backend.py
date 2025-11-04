"""
Pedalboard Backend Integration
Processes audio files using Pedalboard effect chain

Installation:
pip install pedalboard numpy
"""

from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from pedalboard import (
    Chorus, Distortion, Phaser, Clipping,
    Compressor, Gain, Limiter,
    HighpassFilter, LowpassFilter,
    Delay, Reverb,
    PitchShift,
    Bitcrush
)
import numpy as np
from typing import Dict, List, Any, Callable, Optional


class AudioProcessor:
    """Backend processor that uses Pedalboard to apply effects"""

    # Map effect names to Pedalboard classes
    EFFECT_MAP = {
        "Chorus": Chorus,
        "Distortion": Distortion,
        "Phaser": Phaser,
        "Clipping": Clipping,
        "Compressor": Compressor,
        "Gain": Gain,
        "Limiter": Limiter,
        "HighpassFilter": HighpassFilter,
        "LowpassFilter": LowpassFilter,
        "Delay": Delay,
        "Reverb": Reverb,
        "PitchShift": PitchShift,
        "Bitcrush": Bitcrush
    }

    @staticmethod
    def create_effect(effect_name: str, parameters: Dict[str, float]):
        """
        Create a Pedalboard effect with the given parameters

        Args:
            effect_name: Name of the effect
            parameters: Dictionary of parameter values

        Returns:
            Pedalboard effect instance
        """
        effect_class = AudioProcessor.EFFECT_MAP.get(effect_name)

        if not effect_class:
            raise ValueError(f"Unknown effect: {effect_name}")

        # Create effect with parameters
        return effect_class(**parameters)

    @staticmethod
    def process_audio(
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Process audio file with the configured effect chain

        Args:
            config: Dictionary containing:
                - input_file: Path to input audio file
                - output_file: Path to output audio file
                - effects: List of effect configurations
            progress_callback: Optional callback function(current, total, message)

        Returns:
            tuple: (success: bool, message: str, output_file: str)
        """
        try:
            input_file = config["input_file"]
            output_file = config["output_file"]
            effect_configs = config["effects"]

            total_steps = len(effect_configs) + 2

            if progress_callback:
                progress_callback(0, total_steps, "Loading audio file...")

            # Load the audio file
            with AudioFile(input_file) as f:
                audio = f.read(f.frames)
                samplerate = f.samplerate

            if progress_callback:
                progress_callback(1, total_steps, "Building effect chain...")

            # Create effect chain
            board = Pedalboard([])

            for i, effect_config in enumerate(effect_configs):
                effect_name = effect_config["name"]
                parameters = effect_config["parameters"]

                if progress_callback:
                    progress_callback(
                        i + 2,
                        total_steps,
                        f"Adding {effect_name}..."
                    )

                # Create and add effect
                effect = AudioProcessor.create_effect(effect_name, parameters)
                board.append(effect)

            if progress_callback:
                progress_callback(
                    total_steps - 1,
                    total_steps,
                    "Processing audio..."
                )

            # Process the audio
            processed_audio = board(audio, samplerate)

            if progress_callback:
                progress_callback(
                    len(effect_configs) + 2,
                    len(effect_configs) + 2,
                    "Saving output file..."
                )

            # Save the processed audio
            with AudioFile(output_file, 'w', samplerate, processed_audio.shape[0]) as f:
                f.write(processed_audio)

            if progress_callback:
                progress_callback(
                    len(effect_configs) + 2,
                    len(effect_configs) + 2,
                    "Complete!"
                )

            return True, "Processing completed successfully!", output_file

        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            print(error_msg)
            if progress_callback:
                progress_callback(-1, -1, error_msg)
            return False, error_msg, None


def example_usage():
    """Example of how to use the processor"""

    # Example configuration (this would come from your GUI)
    config = {
        "input_file": "input.wav",
        "output_file": "output.wav",
        "effects": [
            {
                "name": "Distortion",
                "parameters": {
                    "drive_db": 25.0
                }
            },
            {
                "name": "Chorus",
                "parameters": {
                    "rate_hz": 1.5,
                    "depth": 0.5,
                    "centre_delay_ms": 7.0,
                    "feedback": 0.2,
                    "mix": 0.5
                }
            },
            {
                "name": "Reverb",
                "parameters": {
                    "room_size": 0.7,
                    "damping": 0.5,
                    "wet_level": 0.33,
                    "dry_level": 0.4,
                    "width": 1.0
                }
            }
        ]
    }

    # Process with progress callback
    def progress(current, total, message):
        if total > 0:
            percent = (current / total) * 100
            print(f"[{percent:.0f}%] {message}")
        else:
            print(message)

    processor = AudioProcessor()
    success, message, output = processor.process_audio(config, progress_callback=progress)

    if success:
        print(f"✅ {message}")
        print(f"Output file: {output}")
    else:
        print(f"❌ {message}")


if __name__ == "__main__":
    example_usage()