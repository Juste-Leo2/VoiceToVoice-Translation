<div align="left">
  <a href="README.md" target="_blank"><img src="https://img.shields.io/badge/🇬🇧-Back%20to%20English-555555?style=flat&labelColor=333" alt="Back to English" /></a>
</div>

# Système de Traduction Voice-to-Voice

Une application basée sur Gradio pour la traduction vocale de bout en bout. Elle utilise `pyannote` pour la diarisation des locuteurs, `Canary-1b-v2` de NVIDIA pour la traduction, et `Piper-TTS` pour la synthèse vocale.

---

## Installation

#### Prérequis
-   Un GPU NVIDIA avec le **CUDA Toolkit 12.8** installé.
-   Python 3.10 (assurez-vous de l'ajouter au PATH de votre système).

#### Étapes d'installation

1.  **Clonez le dépôt :**
    ```sh
    git clone https://github.com/Juste-Leo2/VoiceToVoice-Translation.git
    cd VoiceToVoice-Translation
    ```

2.  **Créez et activez un environnement virtuel :**
    ```sh
    python -m venv .venv
    
    # Sous Windows
    .venv\Scripts\activate
    
    # Sous Linux / macOS
    # source .venv/bin/activate
    ```

3.  **Installez les dépendances avec `uv` :**
    ```sh
    python -m pip install --upgrade pip
    pip install uv
    uv pip install -r requirements.txt --no-deps --index-strategy unsafe-best-match
    ```

## Lancement

1.  **Lancez l'application :**
    ```sh
    python app.py
    ```
2.  Ouvrez l'URL locale indiquée dans le terminal (ex: `http://127.0.0.1:7860`) dans votre navigateur.