from pedalboard import Pedalboard, Chorus, Compressor, Delay, Gain, Reverb, Phaser
from pedalboard.io import AudioStream

print(AudioStream.input_device_names)
print(AudioStream.output_device_names)



# Open up an audio stream:
with AudioStream(
  input_device_name=AudioStream.default_input_device_name,  # Guitar interface
  output_device_name=AudioStream.default_output_device_name
) as stream:
  # Audio is now streaming through this pedalboard and out of your speakers!
  stream.plugins = Pedalboard([
      Compressor(threshold_db=-50, ratio=25),
      Gain(gain_db=30),
      Chorus(),
      Phaser(),
      Reverb(room_size=0.55),
  ])
  input("Press enter to stop streaming...")

# The live AudioStream is now closed, and audio has stopped.