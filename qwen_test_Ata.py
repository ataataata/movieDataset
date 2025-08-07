
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

DEFAULT_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
DEFAULT_INPUT_DIR = Path("/Users/atacinargenc/Desktop/github/movieDataset/Lines")
DEFAULT_OUTPUT_FILE = Path("qwen_test.json")

if torch.cuda.is_available():
    DEVICE = "cuda"  
elif torch.backends.mps.is_available():
    DEVICE = "mps" 
else:
    DEVICE = "cpu"

def load_model(model_name: str = DEFAULT_MODEL):
    print(f"Loading {model_name} on {DEVICE.upper()} …")

    processor = AutoProcessor.from_pretrained(model_name)

    dtype = torch.float16 if DEVICE == "mps" else None
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(DEVICE)
    model.eval()

    return processor, model


def analyse_clip(wav_path: Path, processor, model) -> str:
    """Run a single WAV through the model and return its answer."""
    target_sr = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(wav_path, sr=target_sr, mono=True)

    max_len = 30 * target_sr
    audio = audio[:max_len]

    print(f"   🔍 Audio shape: {audio.shape}, SR: {target_sr}")

    # Build multi-modal chat template
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Can you figure out who this speaker is?"},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    print(f"   🔍 Generated prompt length: {len(text_prompt)} characters")

    inputs = processor(
        text=text_prompt,
        audio=[audio],
        sampling_rate=target_sr, 
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    input_len = inputs.input_ids.size(1)
    print(f"   🔍 Input tokens: {input_len}")

    # Generate
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )

    generated = generated[:, input_len:]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print(f"   Generated: '{response[:100]}{'…' if len(response) > 100 else ''}'")
    return response or "No response generated"


def traverse_and_analyse(
    root_dir: Path,
    out_path: Path,
    processor,
    model,
) -> Dict[str, str]:
    """Walk the directory tree, run every WAV, and dump incremental JSON."""
    results: Dict[str, str] = {}

    if out_path.exists():
        try:
            with out_path.open() as f:
                results = json.load(f)
            print(f" Loaded {len(results)} existing results from {out_path}")
        except Exception as e:
            print(f"Could not read existing results: {e}")

    wav_paths = sorted(root_dir.rglob("*.wav"))
    print(f"🔍 Found {len(wav_paths)} .wav files under {root_dir}")

    for idx, wav_path in enumerate(wav_paths, 1):
        rel_path = wav_path.relative_to(root_dir).as_posix()

        if rel_path in results:
            print(f"[{idx}/{len(wav_paths)}] ⏭️  {rel_path} (already done)")
            continue

        print(f"[{idx}/{len(wav_paths)}] 🎧  {rel_path}")
        try:
            guess = analyse_clip(wav_path, processor, model)
        except Exception as exc:
            print(f" Error on '{rel_path}': {exc}")
            guess = f"ERROR: {exc}"

        results[rel_path] = guess
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    return results


# ─────────────── CLI ─────────────── #
def parse_args():
    p = argparse.ArgumentParser(
        description="Identify speakers with Qwen-2-Audio-7B-Instruct."
    )
    p.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help=f".wav root (default: {DEFAULT_INPUT_DIR})")
    p.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT_FILE,
                   help=f"JSON results file (default: {DEFAULT_OUTPUT_FILE})")
    p.add_argument("--model_name", default=DEFAULT_MODEL,
                   help="HF model name or local checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    processor, model = load_model(args.model_name)
    traverse_and_analyse(
        args.input_dir.expanduser(),
        args.output_file,
        processor,
        model,
    )
    print(f"\n  All done. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
