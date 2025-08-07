import os
import base64
import json
from openai import OpenAI
from add_noise import add_noise
client = OpenAI()  # Requires OPENAI_API_KEY in environment

ROOT_DIR = "./audio_files"
MODEL = "gpt-4o-audio-preview"

def encode_audio(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_audio(encoded_string, filename):
    print(type(encoded_string), type(filename))
    response = client.chat.completions.create(
        model=MODEL,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": 
                        "text", "text": "Carefully listen to the audio. Try to infer the content, and the characteristics of the speaker. Note down as many attrbutes as you can."},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
    )

    # print(response.choices[0].message.audio.transcript)
    return response.choices[0].message.audio.transcript

def process_actor_audio_files(root_dir):
    results = {}

    for actor_name in os.listdir(root_dir):
        actor_path = os.path.join(root_dir, actor_name)
        if not os.path.isdir(actor_path):
            continue

        results[actor_name] = {}

        for filename in os.listdir(actor_path):
            if filename.lower().endswith(".wav"):
                print(f"Processing {filename} for actor {actor_name}...")
                file_path = os.path.join(actor_path, filename)
                print(f"🎙️ {actor_name} - {filename}")
                try:
                    noised = add_noise(file_path, "whitenoise.wav")
                    # encoded = encode_audio(file_path)
                    result = analyze_audio(noised, filename)
                    print(result)
                    results[actor_name][filename] = result
                    print(f"✅ {filename}: {result[:100]}...\n")
                except Exception as e:
                    print(f"❌ Error with {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    print()

    return results

if __name__ == "__main__":
    all_results = process_actor_audio_files(ROOT_DIR)

    # Save results to JSON
    with open("gpt4o_audio_results_by_actor_noised.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("📝 Saved all results to gpt4o_audio_results_by_actor.json")
