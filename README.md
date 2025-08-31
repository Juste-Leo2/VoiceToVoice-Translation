<div align="left">
  <a href="README_FR.md" target="_blank"><img src="https://img.shields.io/badge/🇫🇷-Version%20Française-0073E6?style=flat&labelColor=333" alt="Version Française" /></a>
</div>

# Voice-to-Voice Translation System

A Gradio-based application for end-to-end voice translation. It uses `pyannote` for speaker diarization, NVIDIA `Canary-1b-v2` for translation, and `Piper-TTS` for voice synthesis.

---

## Installation

#### Prerequisites
-   NVIDIA GPU with **CUDA Toolkit 12.8** installed.
-   Python 3.10 (ensure it's added to your system's PATH).

#### Setup Steps

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Juste-Leo2/VoiceToVoice-Translation.git
    cd VoiceToVoice-Translation
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv .venv
    
    # On Windows
    .venv\Scripts\activate
    
    # On Linux / macOS
    # source .venv/bin/activate
    ```

3.  **Install dependencies using `uv`:**
    ```sh
    python -m pip install --upgrade pip
    pip install uv
    uv pip install -r requirements.txt --no-deps --index-strategy unsafe-best-match
    ```

## Usage

1.  **Run the application:**
    ```sh
    python app.py
    ```
2.  Open the local URL provided in the terminal (e.g., `http://127.0.0.1:7860`) in your browser.