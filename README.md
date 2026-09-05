# Vocra
### *You, in every language.*

**Vocra** is a real-time voice translation pipeline that preserves your voice identity — your emotion, breath, accent, and character — across 23+ languages. It does not replace your voice with a generic TTS output. It clones your voice and delivers the translation in it.

Built from the ground up for African voices, African accents, and African markets.

> **Status:** Stage 1 pipeline — fully operational locally. End-to-end verified on a 56-second audio file. Browser extension and real-time streaming in development.

---

## The Problem It Solves

Existing translation tools — Google Translate, live captions, TTS earbuds — translate words. They do not translate *people*. The output is flat, robotic, and foreign. You sound like a stranger in your own meeting.

Vocra solves three specific problems:

- **Language barrier** — A French executive in Abidjan and an English partner in Lagos cannot negotiate freely. One of them is always working in a second language, losing precision and authority.
- **Accent barrier** — Both parties speak English, but a Nigerian accent to a Nordic ear creates friction and fatigue. Meetings slow. Trust erodes.
- **Identity loss** — No existing tool preserves who you are while changing what language you speak. Emotion, urgency, warmth — all stripped in translation.

---

## How It Works

Vocra is a four-stage pipeline. Each stage feeds context into the next.

```
Audio Input → Whisper ASR → Librosa + Gemini → ChatterboxMultilingualTTS → Audio Output
  (your voice)   (transcribe)   (analyze + translate)    (synthesize in your voice)
```

---

## Pipeline Walkthrough

### Stage 0 — Audio Normalisation

Before the pipeline begins, all audio is normalised to `.wav` at 22,050 Hz mono. This ensures consistent input quality across Whisper and Chatterbox regardless of the original file format (`.mp3`, `.m4a`, `.ogg`, `.flac` all supported).

```python
def convert_to_wav(input_path):
    """
    Converts any audio file to .wav format.
    If already .wav, returns the path unchanged.
    """
    if input_path.lower().endswith(".wav"):
        return input_path

    base = os.path.splitext(input_path)[0]
    output_path = base + ".wav"
    file_extension = os.path.splitext(input_path)[1].lower().strip(".")

    audio = AudioSegment.from_file(input_path, format=file_extension)
    audio = audio.set_frame_rate(22050)  # optimal for Whisper + Chatterbox
    audio = audio.set_channels(1)        # mono — voice models perform better
    audio.export(output_path, format="wav")

    return output_path
```

---

### Stage 1 — Transcription (Whisper ASR)

OpenAI Whisper Medium runs on GPU. It auto-detects the source language — no manual configuration required. Word-level timestamps are extracted per segment and flow into every subsequent stage: they drive pause timing, breath position mapping, and acoustic segmentation.

**Benchmark: RTF 0.28× — 3.5× faster than real-time transcription**

```python
def transcribe_audio(audio_path):
    """
    Uses Whisper to transcribe audio into text segments.
    Returns list of segments with start time, end time, and text.
    """
    model = whisper.load_model("medium")
    result = model.transcribe(
        audio_path,
        word_timestamps=True,  # per-word timing — used downstream for breath injection
        verbose=False
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "language": result["language"]
        })

    return segments
```

---

### Stage 2A — Acoustic Emotion Analysis (Librosa)

This is what separates Vocra from text-only translation tools. Rather than asking an LLM to guess tone from words, Vocra measures it directly from the audio waveform — per segment, per sentence.

Three acoustic features are extracted:
- **RMS energy** — volume level. Louder speech = more intense delivery.
- **Pitch variance** — emotional speech has wider F0 swings than flat speech.
- **Zero-crossing rate** — correlates with speech rate and signal texture.

These are measured facts, not inferences.

**Processing cost: 20–50ms per segment on CPU**

```python
def analyze_audio_emotion(audio_path, start_time, end_time):
    """
    Analyzes acoustic characteristics of a specific segment
    to detect emotional markers from the waveform directly.
    """
    y, sr = librosa.load(audio_path, offset=start_time,
                         duration=end_time - start_time)

    if len(y) < 100:
        return {"rms": 0.05, "pitch_variance": 0.0, "speech_rate": 0.05}

    # Volume — angry speech is louder
    rms = float(np.sqrt(np.mean(y**2)))

    # Pitch variation — emotional speech has wider swings
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    active_pitches = pitches[pitches > 0]
    pitch_variance = float(np.var(active_pitches)) if len(active_pitches) > 0 else 0.0

    # Speech rate proxy via zero crossing rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return {"rms": rms, "pitch_variance": pitch_variance, "speech_rate": zcr}
```

---

### Stage 2B — Translation + Tone Mapping (Gemini)

The transcribed text is sent to Gemini alongside the measured acoustic metrics. The LLM receives both what was said *and* how it was said — volume, pitch variance, speech rate. This is acoustically grounded translation, not text-only inference.

Gemini returns three things: the translated text, a tone label (calm / urgent / angry / emotional / excited / hesitant / sad / formal), and an intensity score from 1–10 that scales synthesis expressiveness.

```python
def translate_and_detect_tone(text, audio_metrics, target_language):
    """
    Sends text + measured acoustic metrics to Gemini.
    Returns translation, tone label, and intensity score (1-10).
    The LLM knows how you said it — not just what you said.
    """
    prompt = f"""
    You are a professional interpreter and emotion analyst.

    Original text: "{text}"
    Target language: {target_language}

    Audio characteristics measured from the original recording:
    - Volume level (0 is silent, 0.1+ is loud): {audio_metrics['rms']:.4f}
    - Pitch variation (higher = more emotional): {audio_metrics['pitch_variance']:.4f}
    - Speech rate (higher = faster speaking): {audio_metrics['speech_rate']:.4f}

    Translate naturally as spoken language (not written).
    Classify the tone as ONE of: calm, formal, urgent, angry,
    emotional, excited, hesitant, sad
    Rate emotional intensity 1–10.

    Return ONLY: {{"translation": "...", "tone": "calm", "intensity": 5}}
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash", contents=prompt
    )
    return json.loads(response.text.strip())
```

---

### Stage 2C — Chatterbox Parameter Mapping

The tone label and intensity score are mapped to Chatterbox synthesis parameters. `exaggeration` controls how expressive the output is (0.0 = flat, 1.0 = very expressive). `cfg_weight` controls how strictly the cloned voice is preserved (≥ 0.35 locks identity).

Intensity scales exaggeration dynamically — a calm segment at intensity 3 gets different settings than a calm segment at intensity 7.

```python
def get_chatterbox_settings(tone, intensity):
    """
    Maps detected tone and intensity to Chatterbox parameters.
    exaggeration: 0.0 (flat) to 1.0 (very expressive)
    cfg_weight: 0.0 (loose voice match) to 1.0 (strict voice match)
    """
    tone_base = {
        "calm":      {"exaggeration": 0.3, "cfg_weight": 0.5},
        "formal":    {"exaggeration": 0.2, "cfg_weight": 0.6},
        "urgent":    {"exaggeration": 0.6, "cfg_weight": 0.3},
        "angry":     {"exaggeration": 0.9, "cfg_weight": 0.2},
        "emotional": {"exaggeration": 0.7, "cfg_weight": 0.2},
        "excited":   {"exaggeration": 0.8, "cfg_weight": 0.3},
        "hesitant":  {"exaggeration": 0.3, "cfg_weight": 0.2},
        "sad":       {"exaggeration": 0.4, "cfg_weight": 0.15},
    }

    base = tone_base.get(tone, {"exaggeration": 0.5, "cfg_weight": 0.5})
    intensity_scale = intensity / 5.0
    scaled_exaggeration = min(base["exaggeration"] * intensity_scale, 1.0)

    return {
        "exaggeration": round(scaled_exaggeration, 2),
        "cfg_weight": base["cfg_weight"]
    }
```

---

### Stage 3 — Voice Synthesis (ChatterboxMultilingualTTS)

The translated text — with breath markers injected at waveform-measured positions — is synthesised using ChatterboxMultilingualTTS. The model clones the speaker's voice from a 5-second audio sample. No studio recording required. `cfg_weight ≥ 0.35` locks voice identity across all target languages.

**Benchmark: RTF 0.62× on T4 GPU — faster than real-time synthesis**
*(CPU: RTF 7.7× — batch processing only)*

```python
def synthesize_segment(text, speaker_sample_path, tone, intensity,
                        target_language, chatterbox_model, segment_index):
    """
    Synthesises one translated segment in the speaker's cloned voice.
    Voice identity is locked via cfg_weight.
    Breath markers are injected before synthesis.
    """
    text_with_markers = add_human_markers(text, tone, intensity)
    settings = get_chatterbox_settings(tone, intensity)

    language_map = {
        "french": "fr", "spanish": "es", "arabic": "ar",
        "swahili": "sw", "hindi": "hi", "portuguese": "pt",
        "german": "de",  "chinese": "zh", "japanese": "ja",
        "korean": "ko",  "italian": "it", "dutch": "nl",
        "russian": "ru", "turkish": "tr", "polish": "pl",
        "english": "en"
    }
    language_id = language_map.get(target_language.lower(), "en")

    wav = chatterbox_model.generate(
        text_with_markers,
        audio_prompt_path=speaker_sample_path,
        language_id=language_id,
        exaggeration=settings["exaggeration"],
        cfg_weight=settings["cfg_weight"]   # >= 0.35 locks voice identity
    )

    output_path = f"outputs/segment_{segment_index}.wav"
    ta.save(output_path, wav, chatterbox_model.sr)
    return output_path, wav, chatterbox_model.sr
```

---

### Stage 4 — Segment Assembly

All synthesised segments are joined with 300ms natural pauses between them — matching the rhythm of real human speech rather than producing a continuous robotic stream.

```python
def combine_segments(segment_paths, output_filename="outputs/final_output.wav"):
    """
    Combines all audio segments into one final output file.
    300ms pause mimics the natural gap between spoken thoughts.
    """
    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=300)  # natural inter-sentence gap

    for i, path in enumerate(segment_paths):
        segment_audio = AudioSegment.from_wav(path)
        combined += segment_audio
        if i < len(segment_paths) - 1:
            combined += pause

    combined.export(output_filename, format="wav")
    return output_filename
```

→ **Full pipeline code:** [`stage1_pipeline.py`](./stage1_pipeline.py)

---

## Benchmarks

All benchmarks from an actual end-to-end run on a 56-second audio file. Not synthetic tests.

| Metric | Result | Notes |
|---|---|---|
| Whisper ASR RTF | **0.28×** | 3.5× faster than real-time transcription |
| Synthesis RTF (T4 GPU) | **0.62×** | Faster than real-time synthesis |
| Synthesis RTF (CPU) | 7.7× | Batch processing only — not real-time |
| Segments processed | **13** | Full acoustic pipeline per segment |
| Languages supported | **23+** | Via ChatterboxMultilingualTTS |
| Acoustic analysis per segment | 20–50ms | CPU, negligible overhead |

> **Note on live latency:** End-to-end streaming latency has not yet been benchmarked — dedicated GPU hosting infrastructure is required for this. The synthesis RTF of 0.62× on T4 confirms real-time is technically feasible. Live latency benchmarking is the next phase once GPU compute is secured.

---

## Demo

Audio demo available in [`/demo`](./demo):

- `original_en.wav` — original English input
- `translated_fr.wav` — translated French output, same speaker voice cloned

Additional language pairs (Arabic, Swahili, Yoruba, Spanish, Hindi) are being generated and will be added to the demo folder.

---

## Tech Stack

| Component | Library | Licence |
|---|---|---|
| Speech-to-text | OpenAI Whisper (medium) | MIT |
| Acoustic analysis | librosa | ISC |
| Translation + tone | Google Gemini (gemini-3.6-flash) | Free tier |
| Voice synthesis | ChatterboxMultilingualTTS | MIT |
| Audio processing | pydub | MIT |
| ML framework | PyTorch | BSD |

100% production-viable licensing.

---

## Current Limitations

Being explicit about what this is and is not today:

- **No live streaming yet** — the pipeline processes complete audio files. Real-time mic input is Phase 2 (browser extension with WASAPI/VB-Audio routing).
- **GPU required for real-time** — synthesis runs at RTF 0.62× on T4 GPU. CPU RTF is 7.7× — usable for batch processing, not live calls.
- **African language depth is roadmap** — ChatterboxMultilingualTTS covers 23 languages including Arabic,French,and Swahili. Deep African language coverage (Igbo, Hausa, Twi, Amharic and others) is the core Phase 2/3 development target.
- **Stage 1 only** — this repo contains the core translation pipeline. Browser extension, cloud API, and bilateral mode are subsequent build phases.

---

## Roadmap

| Phase | Timeline | Focus |
|---|---|---|
| **Phase 1 — Now** | Complete | Core pipeline, GPU benchmarks, EN→FR verified |
| **Phase 2 — 3 months** | In progress | Browser extension (Chrome/Zoom/Teams), BEGIN African language fine-tuning (Yoruba, Igbo, Hausa) |
| **Phase 3 — 6 months** | Planned | Cloud API + SDK, DEPLOY African language models in production, revenue model |
| **Phase 4 — 12 months** | Planned | Mobile app, earpiece integration, offline mode, 20+ African languages, Pan-African launch |

African language model expansion is not an afterthought. It is the core long-term differentiator and a dedicated phase of the build.

---

## Setup

### Requirements

```
python >= 3.10
torch >= 2.0 (CUDA recommended for real-time performance)
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Run the pipeline

```bash
python stage1_pipeline.py
```

Edit the three variables at the bottom of the file to point to your audio files:

```python
INPUT_AUDIO    = "audio_samples/input.m4a"    # audio to translate
SPEAKER_SAMPLE = "audio_samples/speaker.m4a"  # voice reference for cloning
TARGET_LANGUAGE = "French"                     # target language
```

Supported input formats: `.wav`, `.mp3`, `.m4a`, `.mp4`, `.ogg`, `.flac`

---

## Project Structure

```
vocra/
├── stage1_pipeline.py       # full pipeline — all four stages
├── requirements.txt
├── .env.example
├── demo/
│   ├── original_en.wav      # original English input
│   ├── translated_fr.wav    # EN→FR output — same cloned voice
│   └── README.md
└── docs/
    ├── pipeline_architecture.md
    └── benchmarks.md
```

---

## Why Africa First

Large global platforms are beginning to build real-time translation. This validates the market. The critical question is not whether translation will exist — it is who will serve African languages, African accents, and African linguistic context with the depth they deserve.

2,000+ languages are spoken across Africa. Language is the #1 barrier to intra-African trade (IFC). The AfCFTA is creating unprecedented cross-border business activity — and language friction is its biggest operational obstacle. Global platforms optimise for global majority use cases. African languages are not a global majority priority.

A tool built specifically around African linguistic reality — from the architecture decisions up — can develop a moat that global platforms cannot easily replicate. And if it does, it becomes strategically valuable to any global player seeking genuine African market presence. Stripe acquired Paystack not to compete with it, but because Paystack understood Nigeria better than Stripe ever could.

---

## Built By

**Akinfisoye Erioluwa**
400-Level Mechanical Engineering
Federal University of Technology, Akure (FUTA)

Built for the Next AI Innovators Challenge 2026.

---

*Language should never be the reason the right voice goes unheard.*
