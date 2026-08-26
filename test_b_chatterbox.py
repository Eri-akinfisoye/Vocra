# VALIDATION TESTS B + E - Chatterbox load, API compat, timed generation
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()  # loads HF_TOKEN and GEMINI_API_KEY from .env

sys.stdout.reconfigure(line_buffering=True)

print("Importing chatterbox...")
t0 = time.time()
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio as ta
print(f"Import OK ({time.time() - t0:.1f}s)")

print("Loading ChatterboxMultilingualTTS on cpu...")
print("(First run downloads model from HuggingFace - can take a while)")
t1 = time.time()
cb = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
t2 = time.time()
print(f"Model load: {t2 - t1:.1f}s")

text = "Bonjour, nous devons approuver ce budget immediatement. Le rapport sera pret vendredi."
prompt_path = "audio_samples/speaker.wav"
print(f"Generating {len(text)} chars of French with prompt: {prompt_path}")
t3 = time.time()
try:
    wav = cb.generate(
        text,
        audio_prompt_path=prompt_path,
        language_id="fr",
        exaggeration=0.5,
        cfg_weight=0.5,
    )
except TypeError as e:
    print(f"API COMPAT ERROR (kwargs rejected): {e}")
    print("Retrying with minimal args only...")
    t3 = time.time()
    wav = cb.generate(text, audio_prompt_path=prompt_path, language_id="fr")
t4 = time.time()

os.makedirs("outputs", exist_ok=True)
out = "outputs/validation_test.wav"
ta.save(out, wav, cb.sr)

gen_secs = wav.shape[-1] / cb.sr
print(f"Generation time: {t4 - t3:.1f}s to produce {gen_secs:.1f}s of audio")
print(f"Speed factor: {(t4 - t3) / gen_secs:.2f}x real-time (lower is better)")
print(f"Saved: {out}")
print("DONE - now listen to outputs/validation_test.wav")
