import os
import re
import tempfile
import time

import ollama
import pygame
import torch

from TTS.TTS.api import TTS

TEMP_DIR = tempfile.gettempdir()
PROMPT_FILE = os.path.join(TEMP_DIR, "prompt.txt")
STATUS_FILE = os.path.join(TEMP_DIR, "status.txt")
AUDIO_DIR = "audio_responses"

# Ensure the audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

def remove_emojis(text):
    """
    Removes emojis and other non-standard characters from the text.
    """
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F]|"  # Emoticons
        "[\U0001F300-\U0001F5FF]|"  # Symbols & pictographs
        "[\U0001F680-\U0001F6FF]|"  # Transport & map symbols
        "[\U0001F1E0-\U0001F1FF]",  # Flags
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def text_to_speech_ai(text, audio_file_path):
    """
    Converts text to speech using an AI-based TTS model and saves it as a WAV file.
    """
    try:
        # Initialize the TTS model (you can specify a different model if needed)
        tts = TTS(model_name="tts_models/en/ljspeech/speedy-speech", progress_bar=False, gpu=False)

        # Generate the WAV file
        tts.tts_to_file(text=text, file_path=audio_file_path)
        return True
    except Exception as e:
        print(f"Error generating audio: {e}")
        return False


def play_audio(audio_file_path):
    """
    Plays the audio file using pygame.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file_path)
        pygame.mixer.music.play()

        # Wait until the audio finishes playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.quit()
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False


def process_prompt():
    """
    Reads the prompt from the prompt file, generates a response, converts it to speech,
    plays the audio, and writes the status to the status file.
    """
    try:
        # Check if the prompt file exists
        if not os.path.exists(PROMPT_FILE):
            return

        # Read the prompt from the file
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        if not prompt:
            return

        print(f"Processing prompt: {prompt}")

        # Send the prompt to Ollama
        response = ollama.generate(model="gemma3:4b", prompt=prompt)

        # Extract the response text
        response_text = response["response"].strip()

        # Remove emojis from the response
        response_text = remove_emojis(response_text)

        print(f"Generated response: {response_text}")

        # Generate the audio file
        audio_file_path = os.path.join(AUDIO_DIR, "response.wav")
        if not text_to_speech_ai(response_text, audio_file_path):
            # Write failure status
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                f.write("false")
            return

        # Play the audio file
        print(f"Playing audio: {audio_file_path}")
        if not play_audio(audio_file_path):
            # Write failure status
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                f.write("false")
            return

        # Write success status
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write("true")

        # Clear the prompt file after processing
        os.remove(PROMPT_FILE)

    except Exception as e:
        print(f"Error processing prompt: {e}")
        # Write failure status
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write("false")


def main():
    """
    Continuously monitors the prompt file for new prompts and processes them.
    """
    print("Starting Ollama file processor...")
    while True:
        process_prompt()
        time.sleep(0.5)  # Check for new prompts every 500ms


if __name__ == "__main__":
    main()
