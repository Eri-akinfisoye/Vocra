# VALIDATION TEST A - Whisper transcription speed on CPU
import time

t0 = time.time()
import whisper
model = whisper.load_model("medium")
t1 = time.time()
print(f"Model load: {t1 - t0:.1f}s")

result = model.transcribe("audio_samples/input.wav", word_timestamps=True, verbose=False)
t2 = time.time()

segments = result["segments"]
audio_dur = segments[-1]["end"] if segments else 0
print(f"Transcription time: {t2 - t1:.1f}s for {audio_dur:.1f}s of audio")
print(f"Real-time factor: {(t2 - t1) / audio_dur:.2f}x (lower is better)")
print(f"Detected language: {result['language']}")
print(f"Segments: {len(segments)}")
print("-" * 50)
for s in segments:
    print(f"  [{s['start']:6.1f} - {s['end']:6.1f}] {s['text'].strip()}")
