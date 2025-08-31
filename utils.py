# utils.py (Final corrected version)
import os
import wave
import torch
import gradio as gr
from pydub import AudioSegment
from piper import PiperVoice
from pyannote.audio import Pipeline
from nemo.collections.asr.models import ASRModel

# --- Configuration and loading of "lightweight" models ---
print("Loading Canary-1b-v2 model, please wait...")

# MODIFIED HERE: We create a torch.device object, not just a string.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

canary_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2").to(DEVICE)

# Initialize the pipeline to None. It will be loaded on the first call.
diarization_pipeline = None

# --- Utility Functions ---
def get_supported_languages():
    return { "English": "en", "French": "fr", "German": "de", "Spanish": "es", "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Polish": "pl", "Russian": "ru", "Swedish": "sv", "Ukrainian": "uk", "Czech": "cs", "Danish": "da", "Finnish": "fi", "Greek": "el", "Hungarian": "hu", "Latvian": "lv", "Romanian": "ro", "Slovak": "sk", "Slovenian": "sl" }

def get_piper_voices(voices_dir="voices"):
    voices_by_lang = {}
    if not os.path.exists(voices_dir): return voices_by_lang
    for filename in os.listdir(voices_dir):
        if filename.endswith(".onnx"):
            lang_code = filename.split('-')[0][:2]
            voice_name = os.path.splitext(filename)[0]
            if lang_code not in voices_by_lang: voices_by_lang[lang_code] = []
            voices_by_lang[lang_code].append(voice_name)
    return voices_by_lang

# --- Main Processing Logic ---
def process_diarization_and_translation(audio_path, num_speakers, source_lang, target_lang):
    global diarization_pipeline

    if diarization_pipeline is None:
        print("Loading diarization pipeline (first call)...")
        try:
            diarization_pipeline = Pipeline.from_pretrained("ivrit-ai/pyannote-speaker-diarization-3.1").to(DEVICE)
            print("Pyannote model loaded successfully.")
        except Exception as e:
            error_message = (
                f"Failed to load the Pyannote diarization model. "
                f"Check your internet connection and ensure you are authenticated with Hugging Face "
                f"(run this command in your terminal: 'huggingface-cli login'). Error: {e}"
            )
            print(error_message)
            raise gr.Error(error_message)

    print("Preprocessing audio (16kHz, mono)...")
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        raise gr.Error(f"Cannot read the audio file. Make sure FFmpeg is installed. Error: {e}")
        
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    temp_audio_for_pyannote = "temp_pyannote_input.wav"
    audio.export(temp_audio_for_pyannote, format="wav")

    print("Diarization in progress...")
    diarization = diarization_pipeline(temp_audio_for_pyannote, num_speakers=num_speakers)
    os.remove(temp_audio_for_pyannote)

    print("Translating segments...")
    segments_data = []
    
    temp_chunk_dir = "temp_chunks_for_canary"
    os.makedirs(temp_chunk_dir, exist_ok=True)

    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        chunk = audio[start_ms:end_ms]

        chunk_path = os.path.join(temp_chunk_dir, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")

        output = canary_model.transcribe([chunk_path], source_lang=source_lang, target_lang=target_lang)
        translated_text = output[0].text if output and output[0] else ""

        segments_data.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "translated_text": translated_text,
            "original_duration": turn.end - turn.start
        })

    for file_name in os.listdir(temp_chunk_dir):
        os.remove(os.path.join(temp_chunk_dir, file_name))
    os.rmdir(temp_chunk_dir)
    
    return segments_data

# --- Synthesis Logic (MODIFIED) ---
def synthesize_and_combine(segments_data, voice_mapping, voices_dir="voices"):
    print("Synthesizing speech and combining audio...")
    final_audio = AudioSegment.empty()
    last_segment_end_time = 0.0

    piper_voices = {}
    for _, voice_name in voice_mapping.items():
        if voice_name and voice_name not in piper_voices:
            model_path = os.path.join(voices_dir, f"{voice_name}.onnx")
            if os.path.exists(model_path):
                # MODIFIED HERE: We check the type of the DEVICE object, not its string value.
                use_cuda_flag = (DEVICE.type == "cuda")
                piper_voices[voice_name] = PiperVoice.load(model_path, use_cuda=use_cuda_flag)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    temp_synthesis_path = os.path.join(output_dir, "temp_synth.wav")

    for segment in segments_data:
        speaker = segment["speaker"]
        voice_name = voice_mapping.get(speaker)
        voice_model = piper_voices.get(voice_name)

        if not voice_model:
            continue

        silence_duration = (segment["start"] - last_segment_end_time) * 1000
        if silence_duration > 0:
            final_audio += AudioSegment.silent(duration=silence_duration)

        with wave.open(temp_synthesis_path, "wb") as wav_file:
            voice_model.synthesize_wav(segment["translated_text"], wav_file)
        
        synthesized_chunk = AudioSegment.from_wav(temp_synthesis_path)
        final_audio += synthesized_chunk
        last_segment_end_time = segment["start"] + synthesized_chunk.duration_seconds

    final_output_path = os.path.join(output_dir, "translated_conversation.wav")
    final_audio.export(final_output_path, format="wav")
    
    if os.path.exists(temp_synthesis_path):
        os.remove(temp_synthesis_path)

    return final_output_path