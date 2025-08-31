# app.py (version modifiée)
import gradio as gr
import pandas as pd
from utils import (
    get_supported_languages,
    get_piper_voices, # On le garde pour trouver une voix par défaut
    process_diarization_and_translation,
    synthesize_and_combine
)
from downloader import get_all_piper_voice_names, download_voice_if_needed

# --- Charger les données une fois au démarrage ---
supported_langs = get_supported_languages()
ALL_PIPER_VOICES = get_all_piper_voice_names()

def get_all_voices_for_lang(lang_code):
    """Filtre la liste complète des voix pour un code langue donné."""
    return [voice for voice in ALL_PIPER_VOICES if voice.startswith(f"{lang_code}_")]

# --- Fonctions de l'interface Gradio ---

def on_voice_select(voice_name):
    """
    Fonction déclenchée lorsqu'une voix est sélectionnée.
    Elle télécharge la voix si nécessaire et retourne un message de statut.
    """
    if not voice_name:
        return "Aucune voix sélectionnée."
    
    success = download_voice_if_needed(voice_name)
    if success:
        return f"Voix '{voice_name}' prête à l'emploi."
    else:
        # Gradio ne gère pas bien les gr.Error dans les events .change, un message est plus sûr
        return f"Échec du téléchargement pour la voix '{voice_name}'. Vérifiez la console."

def step1_process_audio(audio_file, num_speakers, source_lang_name, target_lang_name):
    """
    Première étape: Diarisation et Traduction.
    """
    if audio_file is None:
        raise gr.Error("Veuillez fournir un fichier audio.")
    
    source_lang_code = supported_langs[source_lang_name]
    target_lang_code = supported_langs[target_lang_name]

    if source_lang_code != 'en' and target_lang_code != 'en':
        raise gr.Error("La traduction n'est supportée que depuis/vers l'anglais avec Canary-1b-v2.")

    segments_data = process_diarization_and_translation(
        audio_file, int(num_speakers), source_lang_code, target_lang_code
    )

    if not segments_data:
        gr.Warning("Aucun segment de parole n'a été détecté.")
        return None, [], gr.update(visible=False), None

    df = pd.DataFrame(segments_data)
    df_display = df[['start', 'end', 'speaker', 'translated_text']].copy()
    df_display['start'] = df_display['start'].round(2)
    df_display['end'] = df_display['end'].round(2)

    unique_speakers = sorted(df['speaker'].unique())
    voice_assignment_components = []
    
    all_target_voices = get_all_voices_for_lang(target_lang_code)
    locally_available_target_voices = get_piper_voices().get(target_lang_code, [])

    if not all_target_voices:
        gr.Warning(f"Aucune voix Piper connue pour la langue cible '{target_lang_code}'.")

    # Déterminer une voix par défaut (la première disponible localement, ou la première de la liste)
    default_voice = locally_available_target_voices[0] if locally_available_target_voices else (all_target_voices[0] if all_target_voices else None)
    
    # Pré-télécharger la voix par défaut si elle n'est pas locale
    if default_voice:
        download_voice_if_needed(default_voice)

    for speaker in unique_speakers:
        label = f"Voix pour {speaker}"
        dropdown = gr.Dropdown(
            choices=all_target_voices,
            label=label,
            value=default_voice
        )
        voice_assignment_components.append(dropdown)
    
    # Remplir les composants restants (jusqu'à 10) avec des dropdowns vides et invisibles
    for _ in range(len(unique_speakers), 10):
        voice_assignment_components.append(gr.Dropdown(visible=False))

    return df_display, segments_data, gr.update(visible=True), *voice_assignment_components

def step2_generate_audio(segments_data_json, *voice_choices):
    """
    Deuxième étape: Synthèse vocale.
    """
    if not segments_data_json:
        raise gr.Error("Les données des segments sont manquantes.")

    segments_data = segments_data_json
    unique_speakers = sorted(list(set(seg['speaker'] for seg in segments_data)))
    
    # Assurer que toutes les voix nécessaires sont téléchargées avant la synthèse
    for voice in voice_choices[:len(unique_speakers)]:
        if not download_voice_if_needed(voice):
             raise gr.Error(f"Impossible de télécharger la voix requise '{voice}'. La synthèse est annulée.")

    voice_mapping = {speaker: voice for speaker, voice in zip(unique_speakers, voice_choices)}
    
    final_audio_path = synthesize_and_combine(segments_data, voice_mapping)
    
    return final_audio_path

# --- Construction de l'interface Gradio ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Système de Traduction Voice-to-Voice (avec téléchargement dynamique)")
    gr.Markdown("Utilise `pyannote`, `Canary-1b-v2`, et `Piper-TTS`. Les voix Piper sont téléchargées à la demande.")

    status_textbox = gr.Textbox(label="Statut du téléchargement", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            # ... (identique à avant)
            gr.Markdown("## Étape 1 : Configuration et Traitement")
            audio_input = gr.Audio(type="filepath", label="Fichier audio d'entrée (.wav)")
            num_speakers_input = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Nombre de locuteurs")
            
            with gr.Row():
                source_lang_dropdown = gr.Dropdown(choices=list(supported_langs.keys()), label="Langue source", value="Anglais")
                target_lang_dropdown = gr.Dropdown(choices=list(supported_langs.keys()), label="Langue cible", value="Français")
            
            process_button = gr.Button("1. Lancer la Diarisation et Traduction", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Résultats de la Traduction")
            results_df = gr.DataFrame(headers=["Début (s)", "Fin (s)", "Locuteur", "Texte Traduit"], interactive=False)
            segments_state = gr.State([])

    with gr.Group(visible=False) as assignment_group:
        gr.Markdown("## Étape 2 : Assignation des Voix et Génération")
        gr.Markdown("Assignez une voix. Si une voix n'est pas locale, elle sera téléchargée automatiquement.")
        
        voice_assignment_inputs = []
        with gr.Row():
            for i in range(10):
                voice_assignment_inputs.append(gr.Dropdown(label=f"Voix pour Locuteur {i+1}", visible=(i < num_speakers_input.value)))
        
        generate_button = gr.Button("2. Générer l'audio final", variant="primary")

    with gr.Row():
        output_audio = gr.Audio(label="Audio traduit final", type="filepath")

    # --- Connexion des événements ---

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
    
    # Lier l'événement de changement pour chaque dropdown de voix à la fonction de téléchargement
    for dropdown in voice_assignment_inputs:
        dropdown.change(
            fn=on_voice_select,
            inputs=dropdown,
            outputs=status_textbox
        )

    generate_button.click(
        fn=step2_generate_audio,
        inputs=[segments_state] + voice_assignment_inputs,
        outputs=output_audio
    )

if __name__ == "__main__":
    demo.launch(share=False)