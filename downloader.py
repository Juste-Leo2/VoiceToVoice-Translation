# downloader.py (Version corrigée et robuste)
import os
import sys
import subprocess
# shutil n'est plus nécessaire !

VOICES_DIR = "voices"

def get_all_piper_voice_names():
    """Retourne une liste statique de tous les noms de voix Piper connus."""
    # Cette liste est basée sur la sortie que vous avez fournie.
    # On pourrait la rendre dynamique en scrappant une page, mais c'sest plus simple et fiable.
    return [
        "ar_JO-kareem-low", "ar_JO-kareem-medium", "ca_ES-upc_ona-medium", "ca_ES-upc_ona-x_low",
        "ca_ES-upc_pau-x_low", "cs_CZ-jirka-low", "cs_CZ-jirka-medium", "cy_GB-bu_tts-medium",
        "cy_GB-gwryw_gogleddol-medium", "da_DK-talesyntese-medium", "de_DE-eva_k-x_low",
        "de_DE-karlsson-low", "de_DE-kerstin-low", "de_DE-mls-medium", "de_DE-pavoque-low",
        "de_DE-ramona-low", "de_DE-thorsten-high", "de_DE-thorsten-low", "de_DE-thorsten-medium",
        "de_DE-thorsten_emotional-medium", "el_GR-rapunzelina-low", "en_GB-alan-low",
        "en_GB-alan-medium", "en_GB-alba-medium", "en_GB-aru-medium", "en_GB-cori-high",
        "en_GB-cori-medium", "en_GB-jenny_dioco-medium", "en_GB-northern_english_male-medium",
        "en_GB-semaine-medium", "en_GB-southern_english_female-low", "en_GB-vctk-medium",
        "en_US-amy-low", "en_US-amy-medium", "en_US-arctic-medium", "en_US-bryce-medium",
        "en_US-danny-low", "en_US-hfc_female-medium", "en_US-hfc_male-medium", "en_US-joe-medium",
        "en_US-john-medium", "en_US-kathleen-low", "en_US-kristin-medium", "en_US-kusal-medium",
        "en_US-l2arctic-medium", "en_US-lessac-high", "en_US-lessac-low", "en_US-lessac-medium",
        "en_US-libritts-high", "en_US-libritts_r-medium", "en_US-ljspeech-high",
        "en_US-ljspeech-medium", "en_US-norman-medium", "en_US-reza_ibrahim-medium",
        "en_US-ryan-high", "en_US-ryan-low", "en_US-ryan-medium", "en_US-sam-medium",
        "es_AR-daniela-high", "es_ES-carlfm-x_low", "es_ES-davefx-medium", "es_ES-mls_10246-low",
        "es_ES-mls_9972-low", "es_ES-sharvard-medium", "es_MX-ald-medium", "es_MX-claude-high",
        "fa_IR-amir-medium", "fa_IR-ganji-medium", "fa_IR-ganji_adabi-medium", "fa_IR-gyro-medium",
        "fa_IR-reza_ibrahim-medium", "fi_FI-harri-low", "fi_FI-harri-medium", "fr_FR-gilles-low",
        "fr_FR-mls-medium", "fr_FR-mls_1840-low", "fr_FR-siwis-low", "fr_FR-siwis-medium",
        "fr_FR-tom-medium", "fr_FR-upmc-medium", "he_IL-motek-medium", "hi_IN-pratham-medium",
        "hi_IN-priyamvada-medium", "hi_IN-rohan-medium", "hu_HU-anna-medium", "hu_HU-berta-medium",
        "hu_HU-imre-medium", "id_ID-news_tts-medium", "is_IS-bui-medium", "is_IS-salka-medium",
        "is_IS-steinn-medium", "is_IS-ugla-medium", "it_IT-paola-medium", "it_IT-riccardo-x_low",
        "ka_GE-natia-medium", "kk_KZ-iseke-x_low", "kk_KZ-issai-high", "kk_KZ-raya-x_low",
        "lb_LU-marylux-medium", "lv_LV-aivars-medium", "ml_IN-arjun-medium", "ml_IN-meera-medium",
        "ne_NP-chitwan-medium", "ne_NP-google-medium", "ne_NP-google-x_low", "nl_BE-nathalie-medium",
        "nl_BE-nathalie-x_low", "nl_BE-rdh-medium", "nl_BE-rdh-x_low", "nl_NL-mls-medium",
        "nl_NL-mls_5809-low", "nl_NL-mls_7432-low", "nl_NL-pim-medium", "nl_NL-ronnie-medium",
        "no_NO-talesyntese-medium", "pl_PL-darkman-medium", "pl_PL-gosia-medium",
        "pl_PL-mc_speech-medium", "pl_PL-mls_6892-low", "pt_BR-cadu-medium", "pt_BR-edresson-low",
        "pt_BR-faber-medium", "pt_BR-jeff-medium", "pt_PT-tugão-medium", "ro_RO-mihai-medium",
        "ru_RU-denis-medium", "ru_RU-dmitri-medium", "ru_RU-irina-medium", "ru_RU-ruslan-medium",
        "sk_SK-lili-medium", "sl_SI-artur-medium", "sr_RS-serbski_institut-medium",
        "sv_SE-lisa-medium", "sv_SE-nst-medium", "sw_CD-lanfrica-medium", "te_IN-maya-medium",
        "te_IN-padmavathi-medium", "te_IN-venkatesh-medium", "tr_TR-dfki-medium",
        "tr_TR-fahrettin-medium", "tr_TR-fettah-medium", "uk_UA-lada-x_low",
        "uk_UA-ukrainian_tts-medium", "vi_VN-25hours_single-low", "vi_VN-vais1000-medium",
        "vi_VN-vivos-x_low", "zh_CN-huayan-medium", "zh_CN-huayan-x_low"
    ]

def download_voice_if_needed(voice_name: str) -> bool:
    """
    Vérifie si une voix existe localement. Si non, la télécharge directement
    dans le bon répertoire en utilisant le module de téléchargement de Piper.
    Retourne True en cas de succès ou si la voix existe déjà, False en cas d'échec.
    """
    if not voice_name:
        return False # On considère que "pas de voix" n'est pas un succès

    os.makedirs(VOICES_DIR, exist_ok=True)

    onnx_path = os.path.join(VOICES_DIR, f"{voice_name}.onnx")
    json_path = os.path.join(VOICES_DIR, f"{voice_name}.onnx.json")

    if os.path.exists(onnx_path) and os.path.exists(json_path):
        print(f"La voix '{voice_name}' existe déjà localement.")
        return True

    print(f"Téléchargement de la voix '{voice_name}'...")
    try:
        # MODIFICATION CLÉ : On dit à Piper où télécharger les fichiers
        command = [
            sys.executable, "-m", "piper.download_voices",
            "--voices-dir", VOICES_DIR,  # <- L'ajout magique
            voice_name
        ]
        
        subprocess.run(command, check=True, capture_output=True, text=True)

        # Les fichiers sont maintenant directement dans VOICES_DIR.
        # On vérifie juste qu'ils sont bien arrivés.
        if os.path.exists(onnx_path) and os.path.exists(json_path):
            print(f"La voix '{voice_name}' a été téléchargée avec succès dans '{VOICES_DIR}'.")
            return True
        else:
            # Ce cas est peu probable mais c'est une sécurité
            print(f"Erreur : Le téléchargement semble avoir réussi mais les fichiers de la voix '{voice_name}' sont introuvables.")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Erreur lors du téléchargement de la voix '{voice_name}'.")
        print(f"Commande : {' '.join(e.cmd)}")
        print(f"Sortie : {e.stdout}")
        print(f"Erreur : {e.stderr}")
        return False
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors du téléchargement : {e}")
        return False