from pydub import AudioSegment
import base64
import io
# Load main audio and background noise
def add_noise(main_audio_path, noise_audio_path):
    main_audio = AudioSegment.from_file(main_audio_path)
    noise = AudioSegment.from_file(noise_audio_path)

    # Loop or trim noise to match main audio length
    if len(noise) < len(main_audio):
        # Repeat noise if it's shorter
        repeats = int(len(main_audio) / len(noise)) + 1
        noise = noise * repeats
    noise = noise[:len(main_audio)]

    # Lower the volume of the noise (optional, e.g., -20 dB)
    noise = noise - 20

    # Overlay noise onto main audio
    combined = main_audio.overlay(noise)
    #convert to base64 str and return
    
    # Export to a bytes buffer
    buffer = io.BytesIO()
    combined.export(buffer, format="wav")
    buffer.seek(0)

    # Encode as base64 string
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return audio_base64
    # Export the result
    # combined.export("output_with_noise.wav", format="wav")
