# app.py (Final version with voice manager)
import gradio as gr
import pandas as pd
from utils import (
    get_supported_languages,
    get_piper_voices, # Gets locally installed voices
    process_diarization_and_translation,
    synthesize_and_combine
)
from downloader import get_all_piper_voice_names, download_voice_if_needed

# --- Load data once on startup ---
supported_langs = get_supported_languages()
ALL_PIPER_VOICES = get_all_piper_voice_names()

def get_all_voices_for_lang(lang_code):
    """Filters the complete list of voices for a given language code."""
    if not lang_code:
        return []
    return [voice for voice in ALL_PIPER_VOICES if voice.startswith(f"{lang_code}_")]

# NEW: Function to display installed voices in a readable format
def get_installed_voices_df():
    """Returns a DataFrame of locally installed voices, grouped by language."""
    voices_by_lang = get_piper_voices()
    if not voices_by_lang:
        return pd.DataFrame(columns=["Language", "Installed Voices"])
    
    data = []
    for lang, voices in sorted(voices_by_lang.items()):
        data.append({"Language": lang, "Installed Voices": ", ".join(voices)})
    return pd.DataFrame(data)


# --- Voice Manager Functions (NEW) ---

def update_voice_dl_options(lang_name):
    """Updates the voice download choices based on the selected language."""
    if not lang_name:
        return gr.update(choices=[], value=None)
    lang_code = supported_langs.get(lang_name)
    voices = get_all_voices_for_lang(lang_code)
    return gr.update(choices=voices, value=voices[0] if voices else None)

def handle_voice_download(voice_name):
    """Handles downloading a voice and updates the display."""
    if not voice_name:
        return "Please select a voice.", get_installed_voices_df()
    
    success = download_voice_if_needed(voice_name)
    
    if success:
        status_message = f"Voice '{voice_name}' is ready."
    else:
        status_message = f"Download failed for '{voice_name}'. Check the console."
        
    return status_message, get_installed_voices_df()


# --- Gradio Interface Functions (MODIFIED) ---

def step1_process_audio(audio_file, num_speakers, source_lang_name, target_lang_name):
    """
    First step: Diarization and Translation.
    """
    if audio_file is None:
        raise gr.Error("Please provide an audio file.")
    
    source_lang_code = supported_langs[source_lang_name]
    target_lang_code = supported_langs[target_lang_name]

    if source_lang_code != 'en' and target_lang_code != 'en':
        raise gr.Error("Translation is only supported to/from English with Canary-1b-v2.")

    segments_data = process_diarization_and_translation(
        audio_file, int(num_speakers), source_lang_code, target_lang_code
    )

    if not segments_data:
        gr.Warning("No speech segments were detected.")
        return None, [], gr.update(visible=False), *(gr.Dropdown(visible=False) for _ in range(10))

    df = pd.DataFrame(segments_data)
    df_display = df[['start', 'end', 'speaker', 'translated_text']].copy()
    df_display['start'] = df_display['start'].round(2)
    df_display['end'] = df_display['end'].round(2)

    unique_speakers = sorted(df['speaker'].unique())
    voice_assignment_components = []
    
    # MODIFIED: Load only the LOCALLY available voices for the target language
    locally_available_target_voices = get_piper_voices().get(target_lang_code, [])

    if not locally_available_target_voices:
        gr.Warning(
            f"No voices are installed locally for the language '{target_lang_name}' ({target_lang_code}). "
            f"Please download one using the 'Piper Voice Manager' at the top of the page."
        )
        # We still display the dropdowns to avoid crashing the UI, but they will be empty.
    
    default_voice = locally_available_target_voices[0] if locally_available_target_voices else None
    
    for speaker in unique_speakers:
        label = f"Voice for {speaker}"
        dropdown = gr.Dropdown(
            choices=locally_available_target_voices, # MODIFIED: Use only local voices
            label=label,
            value=default_voice
        )
        voice_assignment_components.append(dropdown)
    
    # Fill the remaining components (up to 10) with empty, invisible dropdowns
    for _ in range(len(unique_speakers), 10):
        voice_assignment_components.append(gr.Dropdown(visible=False))

    return df_display, segments_data, gr.update(visible=True), *voice_assignment_components

def step2_generate_audio(segments_data_json, *voice_choices):
    """
    Second step: Speech Synthesis.
    """
    if not segments_data_json:
        raise gr.Error("Segment data is missing.")

    # The download check is no longer critical here as voices are already local,
    # but we keep it for safety.
    unique_speakers = sorted(list(set(seg['speaker'] for seg in segments_data_json)))
    for voice in voice_choices[:len(unique_speakers)]:
        if not download_voice_if_needed(voice): # This will just check for existence
             raise gr.Error(f"The required voice '{voice}' could not be found. Synthesis canceled.")

    voice_mapping = {speaker: voice for speaker, voice in zip(unique_speakers, voice_choices)}
    
    final_audio_path = synthesize_and_combine(segments_data_json, voice_mapping)
    
    return final_audio_path

# --- Build Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Voice-to-Voice Translation System")
    gr.Markdown("Uses `pyannote`, `Canary-1b-v2`, and `Piper-TTS`.")

    # NEW: The voice manager
    with gr.Accordion("Piper Voice Manager (Open to download/view voices)", open=False):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Download a new voice")
                lang_select_for_dl = gr.Dropdown(
                    choices=list(supported_langs.keys()), 
                    label="1. Choose language"
                )
                voice_select_for_dl = gr.Dropdown(label="2. Choose voice")
                download_button = gr.Button("Download selected voice", variant="secondary")
                download_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=3):
                gr.Markdown("### Currently Installed Voices")
                installed_voices_df = gr.DataFrame(
                    value=get_installed_voices_df, 
                    headers=["Language", "Installed Voices"],
                    interactive=False,
                    every=5 # Refreshes every 5 seconds
                )

    gr.Markdown("---") # Visual separator

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Step 1: Configuration and Processing")
            audio_input = gr.Audio(type="filepath", label="Input audio file (.wav)")
            num_speakers_input = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Number of speakers")
            
            with gr.Row():
                source_lang_dropdown = gr.Dropdown(choices=list(supported_langs.keys()), label="Source language", value="English")
                target_lang_dropdown = gr.Dropdown(choices=list(supported_langs.keys()), label="Target language", value="French")
            
            process_button = gr.Button("1. Start Diarization and Translation", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Translation Results")
            results_df = gr.DataFrame(headers=["Start (s)", "End (s)", "Speaker", "Translated Text"], interactive=False)
            segments_state = gr.State([])

    with gr.Group(visible=False) as assignment_group:
        gr.Markdown("## Step 2: Voice Assignment and Generation")
        gr.Markdown("Assign a voice **from your installed voices**.")
        
        voice_assignment_inputs = []
        with gr.Row():
            for i in range(10):
                voice_assignment_inputs.append(gr.Dropdown(label=f"Voice for Speaker {i+1}", visible=(i < num_speakers_input.value)))
        
        generate_button = gr.Button("2. Generate final audio", variant="primary")

    with gr.Row():
        output_audio = gr.Audio(label="Final translated audio", type="filepath")

    # --- Connect Events ---

    # NEW: Events for the voice manager
    lang_select_for_dl.change(
        fn=update_voice_dl_options,
        inputs=lang_select_for_dl,
        outputs=voice_select_for_dl
    )
    download_button.click(
        fn=handle_voice_download,
        inputs=voice_select_for_dl,
        outputs=[download_status, installed_voices_df]
    )

    # Main workflow events
    process_button.click(
        fn=step1_process_audio,
        inputs=[audio_input, num_speakers_input, source_lang_dropdown, target_lang_dropdown],
        outputs=[results_df, segments_state, assignment_group] + voice_assignment_inputs
    )
    
    def update_voice_dropdowns_visibility(num_speakers):
        return [gr.update(visible=(i < num_speakers)) for i in range(10)]

    num_speakers_input.change(
        fn=update_voice_dropdowns_visibility,
        inputs=num_speakers_input,
        outputs=voice_assignment_inputs
    )
    
    # REMOVED: We no longer need to download voices upon selection
    # for dropdown in voice_assignment_inputs:
    #     dropdown.change(...)

    generate_button.click(
        fn=step2_generate_audio,
        inputs=[segments_state] + voice_assignment_inputs,
        outputs=output_audio
    )

if __name__ == "__main__":
    demo.launch(share=False)