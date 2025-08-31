# app.py (Version finale avec gestionnaire de voix)
import gradio as gr
import pandas as pd
from utils import (
    get_supported_languages,
    get_piper_voices, # Devient la source principale pour les voix disponibles
    process_diarization_and_translation,
    synthesize_and_combine
)
from downloader import get_all_piper_voice_names, download_voice_if_needed

# --- Charger les données une fois au démarrage ---
supported_langs = get_supported_languages()
ALL_PIPER_VOICES = get_all_piper_voice_names()

def get_all_voices_for_lang(lang_code):
    """Filtre la liste complète des voix pour un code langue donné."""
    if not lang_code:
        return []
    return [voice for voice in ALL_PIPER_VOICES if voice.startswith(f"{lang_code}_")]

# NOUVEAU: Fonction pour afficher les voix installées dans un format lisible
def get_installed_voices_df():
    """Retourne un DataFrame des voix installées localement, groupées par langue."""
    voices_by_lang = get_piper_voices()
    if not voices_by_lang:
        return pd.DataFrame(columns=["Langue", "Voix Installées"])
    
    data = []
    for lang, voices in sorted(voices_by_lang.items()):
        data.append({"Langue": lang, "Voix Installées": ", ".join(voices)})
    return pd.DataFrame(data)


# --- Fonctions pour le Gestionnaire de Voix (NOUVEAU) ---

def update_voice_dl_options(lang_name):
    """Met à jour les choix de voix à télécharger en fonction de la langue sélectionnée."""
    if not lang_name:
        return gr.update(choices=[], value=None)
    lang_code = supported_langs.get(lang_name)
    voices = get_all_voices_for_lang(lang_code)
    return gr.update(choices=voices, value=voices[0] if voices else None)

def handle_voice_download(voice_name):
    """Gère le téléchargement d'une voix et met à jour l'affichage."""
    if not voice_name:
        return "Veuillez sélectionner une voix.", get_installed_voices_df()
    
    success = download_voice_if_needed(voice_name)
    
    if success:
        status_message = f"Voix '{voice_name}' prête."
    else:
        status_message = f"Échec du téléchargement pour '{voice_name}'. Vérifiez la console."
        
    return status_message, get_installed_voices_df()


# --- Fonctions de l'interface Gradio (MODIFIÉES) ---

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
        return None, [], gr.update(visible=False), *(gr.Dropdown(visible=False) for _ in range(10))

    df = pd.DataFrame(segments_data)
    df_display = df[['start', 'end', 'speaker', 'translated_text']].copy()
    df_display['start'] = df_display['start'].round(2)
    df_display['end'] = df_display['end'].round(2)

    unique_speakers = sorted(df['speaker'].unique())
    voice_assignment_components = []
    
    # MODIFIÉ: On ne charge que les voix LOCALEMENT disponibles pour la langue cible
    locally_available_target_voices = get_piper_voices().get(target_lang_code, [])

    if not locally_available_target_voices:
        gr.Warning(
            f"Aucune voix n'est installée localement pour la langue '{target_lang_name}' ({target_lang_code}). "
            f"Veuillez en télécharger une via le 'Gestionnaire de Voix Piper' en haut de la page."
        )
        # On affiche quand même les dropdowns pour ne pas crasher l'UI, mais ils seront vides.
    
    default_voice = locally_available_target_voices[0] if locally_available_target_voices else None
    
    for speaker in unique_speakers:
        label = f"Voix pour {speaker}"
        dropdown = gr.Dropdown(
            choices=locally_available_target_voices, # MODIFIÉ: On n'utilise que les voix locales
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

    # La vérification de téléchargement n'est plus critique ici car les voix sont déjà locales,
    # mais on la garde par sécurité.
    unique_speakers = sorted(list(set(seg['speaker'] for seg in segments_data_json)))
    for voice in voice_choices[:len(unique_speakers)]:
        if not download_voice_if_needed(voice): # Ceci vérifiera juste l'existence
             raise gr.Error(f"La voix requise '{voice}' est introuvable. La synthèse est annulée.")

    voice_mapping = {speaker: voice for speaker, voice in zip(unique_speakers, voice_choices)}
    
    final_audio_path = synthesize_and_combine(segments_data_json, voice_mapping)
    
    return final_audio_path

# --- Construction de l'interface Gradio ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Système de Traduction Voice-to-Voice")
    gr.Markdown("Utilise `pyannote`, `Canary-1b-v2`, et `Piper-TTS`.")

    # NOUVEAU: Le gestionnaire de voix
    with gr.Accordion("Gestionnaire de Voix Piper (Ouvrir pour télécharger/voir les voix)", open=False):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Télécharger une nouvelle voix")
                lang_select_for_dl = gr.Dropdown(
                    choices=list(supported_langs.keys()), 
                    label="1. Choisir la langue"
                )
                voice_select_for_dl = gr.Dropdown(label="2. Choisir la voix")
                download_button = gr.Button("Télécharger la voix sélectionnée", variant="secondary")
                download_status = gr.Textbox(label="Statut", interactive=False)
            with gr.Column(scale=3):
                gr.Markdown("### Voix actuellement installées")
                installed_voices_df = gr.DataFrame(
                    value=get_installed_voices_df, 
                    headers=["Langue", "Voix Installées"],
                    interactive=False,
                    every=5 # Rafraîchit toutes les 5 secondes
                )

    gr.Markdown("---") # Séparateur visuel

    with gr.Row():
        with gr.Column(scale=1):
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
        gr.Markdown("Assignez une voix **parmi celles que vous avez installées**.")
        
        voice_assignment_inputs = []
        with gr.Row():
            for i in range(10):
                voice_assignment_inputs.append(gr.Dropdown(label=f"Voix pour Locuteur {i+1}", visible=(i < num_speakers_input.value)))
        
        generate_button = gr.Button("2. Générer l'audio final", variant="primary")

    with gr.Row():
        output_audio = gr.Audio(label="Audio traduit final", type="filepath")

    # --- Connexion des événements ---

    # NOUVEAU: Événements pour le gestionnaire de voix
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

    # Événements du workflow principal
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
    
    # SUPPRIMÉ: On n'a plus besoin de télécharger les voix au moment de la sélection
    # for dropdown in voice_assignment_inputs:
    #     dropdown.change(...)

    generate_button.click(
        fn=step2_generate_audio,
        inputs=[segments_state] + voice_assignment_inputs,
        outputs=output_audio
    )

if __name__ == "__main__":
    demo.launch(share=False)