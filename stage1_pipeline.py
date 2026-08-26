# ============================================================
# VOICEBRIDGE — STAGE 1 PIPELINE
# Flow: Audio File → Whisper → Gemini → Chatterbox → Output
# ============================================================

import os
import json
import time
import numpy as np
import librosa
from pydub import AudioSegment
from dotenv import load_dotenv
import whisper
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from google import genai

# ============================================================
# LOAD ENVIRONMENT VARIABLES (your API keys from .env file)
# ============================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ============================================================
# CONFIGURE GEMINI
# ============================================================
client = genai.Client(api_key=GEMINI_API_KEY)
# ============================================================
# STEP 0 — AUDIO FORMAT CONVERSION
# Converts any audio format to .wav before pipeline begins
# This ensures uniform audio quality throughout
# ============================================================
def convert_to_wav(input_path):
    """
    Converts any audio file to .wav format.
    If already .wav, returns the path unchanged.
    """

    print(f"  Checking format of: {input_path}")

    # If already .wav, do nothing
    if input_path.lower().endswith(".wav"):
        print(f"  Already .wav — no conversion needed")
        return input_path

    # Build the output path — same folder, same name, .wav extension
    base = os.path.splitext(input_path)[0]
    output_path = base + ".wav"

    # Detect format from file extension
    file_extension = os.path.splitext(input_path)[1].lower().strip(".")
    print(f"  Converting .{file_extension} → .wav ...")

    # Load the audio file
    audio = AudioSegment.from_file(input_path, format=file_extension)

    # Set sample rate to 22050Hz — optimal balance for Whisper and Chatterbox
    audio = audio.set_frame_rate(22050)

    # Convert to mono — voice models perform better on single channel audio
    audio = audio.set_channels(1)

    # Export as .wav
    audio.export(output_path, format="wav")
    print(f"  Conversion complete — saved as: {output_path}")

    return output_path


# ============================================================
# STEP 1 — SPEECH TO TEXT (WHISPER)
# Transcribes audio and splits into segments with timestamps
# ============================================================
def transcribe_audio(audio_path):
    """
    Uses Whisper to transcribe audio into text segments.
    Returns list of segments with start time, end time, and text.
    """

    print(f"  Loading Whisper model...")
    model = whisper.load_model("medium")

    print(f"  Transcribing audio...")
    result = model.transcribe(
        audio_path,
        word_timestamps=True,   # gives us timing per word
        verbose=False
    )

    # Extract segments
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "language": result["language"]
        })

    print(f"  Transcription complete — {len(segments)} segments detected")
    print(f"  Detected language: {result['language']}")

    return segments


# ============================================================
# STEP 2A — AUDIO EMOTION ANALYSIS (LIBROSA)
# Analyzes acoustic properties of each audio segment
# Volume, pitch variance, and speech rate are measured
# ============================================================
def analyze_audio_emotion(audio_path, start_time, end_time):
    """
    Analyzes the acoustic characteristics of a specific
    segment of audio to detect emotional markers.
    """

    # Load just the segment we care about
    y, sr = librosa.load(
        audio_path,
        offset=start_time,
        duration=end_time - start_time
    )

    # If segment is too short, return neutral defaults
    if len(y) < 100:
        return {"rms": 0.05, "pitch_variance": 0.0, "speech_rate": 0.05}

    # Measure volume (RMS energy) — angry speech is louder
    rms = float(np.sqrt(np.mean(y**2)))

    # Measure pitch variation — emotional speech has wider swings
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    active_pitches = pitches[pitches > 0]
    pitch_variance = float(np.var(active_pitches)) if len(active_pitches) > 0 else 0.0

    # Measure speech rate via zero crossing rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return {
        "rms": rms,
        "pitch_variance": pitch_variance,
        "speech_rate": zcr
    }


# ============================================================
# STEP 2B — TONE DETECTION + TRANSLATION (GEMINI)
# Sends text and audio metrics to Gemini
# Gets back translation AND tone with intensity score
# ============================================================
def translate_and_detect_tone(text, audio_metrics, target_language):
    """
    Uses Gemini to translate text and detect emotional tone.
    Audio metrics from librosa give Gemini acoustic context.
    Returns translation, tone label, and intensity (1-10).
    """

    prompt = f"""
    You are a professional interpreter and emotion analyst.
    
    Analyze this spoken segment and translate it.
    
    Original text: "{text}"
    Target language: {target_language}
    
    Audio characteristics measured from the original recording:
    - Volume level (0 is silent, 0.1+ is loud): {audio_metrics['rms']:.4f}
    - Pitch variation (higher = more emotional): {audio_metrics['pitch_variance']:.4f}
    - Speech rate (higher = faster speaking): {audio_metrics['speech_rate']:.4f}
    
    Use these guidelines for tone detection:
    - High volume + high pitch variation = angry or very excited
    - Low volume + low variation = calm or sad
    - High rate + high volume = urgent or angry
    - Low rate + low volume = emotional, tired, or grieving
    - Moderate everything = formal or neutral
    
    Translate naturally as spoken language (not written).
    Where natural in {target_language}, add conversational 
    markers a native speaker would use.
    Preserve the speaker's urgency and emotional weight.
    
    Classify the tone as ONE of:
    calm, formal, urgent, angry, emotional, excited, hesitant, sad
    
    Rate emotional intensity from 1 to 10.
    1 = completely flat/monotone
    10 = extremely intense emotion
    
    Return ONLY this exact JSON format, nothing else:
    {{"translation": "translated text here", "tone": "calm", "intensity": 5}}
    """

    response = client.models.generate_content(model="gemini-3.6-flash",contents=prompt)
    # Clean response and parse JSON
    response_text = response.text.strip()

    # Remove markdown code blocks if Gemini adds them
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    response_text = response_text.strip()

    result = json.loads(response_text)
    return result


# ============================================================
# STEP 2C — PARALINGUISTIC MARKER INJECTION
# Adds human sound markers based on detected tone
# Makes output sound natural rather than robotic
# ============================================================
def add_human_markers(text, tone, intensity):
    """
    Injects paralinguistic tags into translated text.
    These tell Chatterbox to add natural human sounds
    like breathing, sighs, etc. at appropriate points.
    """

    if tone == "angry" and intensity > 7:
        # High anger — sharp delivery, forceful breaths
        text = text.replace(". ", ". [breath] ")
        text = text.replace("! ", "! [breath] ")

    elif tone == "emotional" and intensity > 6:
        # High emotion — broken delivery with sighs
        text = text.replace(", ", ", [breath] ")
        text = text.replace(". ", ". [sigh] ")

    elif tone == "sad":
        # Sadness — slow, sighing delivery
        text = text.replace(". ", ". [sigh] ")

    elif tone in ["calm", "formal"]:
        # Measured delivery — steady controlled breaths
        text = text.replace(". ", ". [breath] ")

    elif tone == "excited":
        # Energetic — quick breaths between thoughts
        text = text.replace("! ", "! [breath] ")
        text = text.replace(", ", ", [breath] ")

    elif tone == "hesitant":
        # Uncertain — slight pause markers
        text = text.replace(", ", ", [breath] ")

    return text


# ============================================================
# STEP 2D — CHATTERBOX PARAMETER MAPPING
# Maps tone label + intensity to Chatterbox settings
# Controls how expressive and faithful the output is
# ============================================================
def get_chatterbox_settings(tone, intensity):
    """
    Maps detected tone and intensity to Chatterbox parameters.
    exaggeration: 0.0 (flat) to 1.0 (very expressive)
    cfg_weight: 0.0 (loose) to 1.0 (strict voice match)
    """

    # Base settings per tone
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

    # Scale exaggeration by intensity
    # intensity 5 = base value, intensity 10 = maximum
    intensity_scale = intensity / 5.0
    scaled_exaggeration = min(base["exaggeration"] * intensity_scale, 1.0)

    return {
        "exaggeration": round(scaled_exaggeration, 2),
        "cfg_weight": base["cfg_weight"]
    }


# ============================================================
# STEP 3 — VOICE SYNTHESIS (CHATTERBOX)
# Synthesizes each translated segment in the speaker's
# cloned voice with the correct tone settings
# ============================================================
def synthesize_segment(
    text,
    speaker_sample_path,
    tone,
    intensity,
    target_language,
    chatterbox_model,
    segment_index
):
    """
    Uses Chatterbox to synthesize a single text segment
    in the speaker's cloned voice with tone-matched delivery.
    Saves segment audio to outputs/ folder.
    """

    # Add human breath and sound markers
    text_with_markers = add_human_markers(text, tone, intensity)

    # Get Chatterbox settings for this tone and intensity
    settings = get_chatterbox_settings(tone, intensity)

    print(f"  Synthesizing segment {segment_index + 1}...")
    print(f"    Tone: {tone} | Intensity: {intensity}/10")
    print(f"    Exaggeration: {settings['exaggeration']} | CFG: {settings['cfg_weight']}")
    print(f"    Text: {text[:60]}...")

    # Map target language to Chatterbox language ID
    language_map = {
        "french": "fr",
        "spanish": "es",
        "arabic": "ar",
        "swahili": "sw",
        "hindi": "hi",
        "portuguese": "pt",
        "german": "de",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "italian": "it",
        "dutch": "nl",
        "russian": "ru",
        "turkish": "tr",
        "polish": "pl",
        "english": "en"
    }

    language_id = language_map.get(target_language.lower(), "en")

    # Generate audio with Chatterbox
    wav = chatterbox_model.generate(
        text_with_markers,
        audio_prompt_path=speaker_sample_path,
        language_id=language_id,
        exaggeration=settings["exaggeration"],
        cfg_weight=settings["cfg_weight"]
    )

    # Save segment to outputs folder
    output_path = f"outputs/segment_{segment_index}.wav"
    ta.save(output_path, wav, chatterbox_model.sr)

    return output_path, wav, chatterbox_model.sr


# ============================================================
# STEP 4 — COMBINE SEGMENTS
# Joins all synthesized audio segments with natural
# micro-pauses between them — mimics human speech rhythm
# ============================================================
def combine_segments(segment_paths, output_filename="outputs/final_output.wav"):
    """
    Combines all audio segments into one final output file.
    Adds small natural pauses between segments to mimic
    the rhythm of real human speech.
    """

    print(f"  Combining {len(segment_paths)} segments...")

    # Start with silence
    combined = AudioSegment.empty()

    # Natural pause between segments (milliseconds)
    # 300ms mimics the natural gap between spoken thoughts
    pause = AudioSegment.silent(duration=300)

    for i, path in enumerate(segment_paths):
        segment_audio = AudioSegment.from_wav(path)
        combined += segment_audio
        # Add pause between segments but not after the last one
        if i < len(segment_paths) - 1:
            combined += pause

    # Export final combined audio
    combined.export(output_filename, format="wav")
    print(f"  Final output saved: {output_filename}")

    return output_filename


# ============================================================
# MAIN PIPELINE FUNCTION
# Orchestrates all steps in sequence
# ============================================================
def run_pipeline(audio_input_path, speaker_sample_path, target_language="French"):
    """
    Main pipeline function. Takes an audio file, transcribes it,
    translates it with tone detection, and synthesizes the output
    in the original speaker's cloned voice.

    Parameters:
    - audio_input_path: path to the audio you want translated
    - speaker_sample_path: path to the voice reference sample
    - target_language: language to translate into (default French)
    """

    start_time = time.time()

    print("\n" + "="*55)
    print("   VOICEBRIDGE PIPELINE — STARTING")
    print("="*55)
    print(f"  Input audio:    {audio_input_path}")
    print(f"  Voice sample:   {speaker_sample_path}")
    print(f"  Target language: {target_language}")
    print("="*55)

    # ----------------------------------------------------------
    # STEP 0 — Convert audio files to .wav if needed
    # ----------------------------------------------------------
    print("\n[STEP 0] Checking and converting audio formats...")
    audio_input_path = convert_to_wav(audio_input_path)
    speaker_sample_path = convert_to_wav(speaker_sample_path)
    print("  Formats confirmed ✓")

    # ----------------------------------------------------------
    # STEP 1 — Transcribe with Whisper
    # ----------------------------------------------------------
    print("\n[STEP 1] Transcribing with Whisper...")
    segments = transcribe_audio(audio_input_path)

    # Print what Whisper found
    print("\n  Transcription results:")
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1} [{seg['start']:.1f}s → {seg['end']:.1f}s]: {seg['text']}")

    # ----------------------------------------------------------
    # STEP 2 — Load Chatterbox model (load once, use for all segments)
    # ----------------------------------------------------------
    print("\n[STEP 2] Loading Chatterbox Multilingual model...")
    print("  This may take a moment on first run...")
    chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(
        device="cpu"   # change to "cuda" if you have NVIDIA GPU
    )
    print("  Chatterbox loaded ✓")

    # ----------------------------------------------------------
    # STEP 3 — Process each segment: analyze, translate, synthesize
    # ----------------------------------------------------------
    print(f"\n[STEP 3] Processing {len(segments)} segments...")
    segment_output_paths = []

    for i, segment in enumerate(segments):
        print(f"\n  --- Segment {i+1} of {len(segments)} ---")

        # Skip empty segments
        if not segment["text"].strip():
            print("  Empty segment — skipping")
            continue

        # Analyze audio emotion for this segment
        audio_metrics = analyze_audio_emotion(
            audio_input_path,
            segment["start"],
            segment["end"]
        )

        # Translate and detect tone with Gemini
        print(f"  Sending to Gemini for translation + tone detection...")
        gemini_result = translate_and_detect_tone(
            segment["text"],
            audio_metrics,
            target_language
        )

        translation = gemini_result["translation"]
        tone = gemini_result["tone"]
        intensity = gemini_result["intensity"]

        print(f"  Original:    {segment['text']}")
        print(f"  Translation: {translation}")
        print(f"  Tone:        {tone} (intensity {intensity}/10)")

        # Synthesize this segment in the speaker's cloned voice
        output_path, _, _ = synthesize_segment(
            translation,
            speaker_sample_path,
            tone,
            intensity,
            target_language,
            chatterbox_model,
            i
        )

        segment_output_paths.append(output_path)
        print(f"  Segment {i+1} complete ✓")

    # ----------------------------------------------------------
    # STEP 4 — Combine all segments into final output
    # ----------------------------------------------------------
    print("\n[STEP 4] Combining all segments...")
    final_output = combine_segments(segment_output_paths)

    # ----------------------------------------------------------
    # DONE
    # ----------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "="*55)
    print("   VOICEBRIDGE PIPELINE — COMPLETE")
    print("="*55)
    print(f"  Total time:     {elapsed:.1f} seconds")
    print(f"  Segments:       {len(segment_output_paths)}")
    print(f"  Output file:    {final_output}")
    print("="*55)
    print("\n  Open outputs/final_output.wav to hear the result.")

    return final_output


# ============================================================
# RUN THE PIPELINE
# Change these three values to match your files
# ============================================================
if __name__ == "__main__":

    # Path to the audio you want to translate
    # Can be .wav, .mp3, .m4a, .mp4, .ogg, .flac — any format
    INPUT_AUDIO = "audio_samples/input.m4a"

    # Path to your voice reference sample for cloning
    # Clean recording, at least 10 seconds, ideally 30-60 seconds
    SPEAKER_SAMPLE = "audio_samples/speaker.m4a"

    # Language to translate into
    # Options: French, Spanish, Arabic, Swahili, Hindi,
    # Portuguese, German, Chinese, Japanese, Korean, Italian
    TARGET_LANGUAGE = "French"

    run_pipeline(INPUT_AUDIO, SPEAKER_SAMPLE, TARGET_LANGUAGE)