import os
import re
import tempfile
import time
import uuid

import ollama
import pygame
import torch

from TTS.TTS.api import TTS

TEMP_DIR = tempfile.gettempdir()
PROMPT_FILE = os.path.join(TEMP_DIR, "yaboai_prompt.txt")
STATUS_FILE = os.path.join(TEMP_DIR, "yaboai_status.txt")
AUDIO_DIR = "audio_responses"

# Ensure the audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a persistent session ID
SESSION_ID = str(uuid.uuid4())

# Define the system prompt
SYSTEM_PROMPT = (
    "You are an F1 commentator with the personality of Jeremy Clarkson. "
    "Provide humorous and insightful commentary about F1 races, drivers, and events. "
    "Do not include any sound effects, stage directions, or descriptions of sounds or vocal effects. "
    "Limit your response to a few sentences. "
    "Here is the latest event that you need to comment on: "
)

tts = TTS(
    model_name="tts_models/en/ljspeech/speedy-speech",
    progress_bar=False,
    gpu=True if device == "cuda" else False,
)


def remove_emojis(text):
    """
    Removes emojis and other non-standard characters from the text.
    """
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f]|"  # Emoticons
        "[\U0001f300-\U0001f5ff]|"  # Symbols & pictographs
        "[\U0001f680-\U0001f6ff]|"  # Transport & map symbols
        "[\U0001f1e0-\U0001f1ff]",  # Flags
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def text_to_speech_ai(text, audio_file_path):
    """
    Converts text to speech using an AI-based TTS model and saves it as a WAV file.
    """
    try:
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
    Reads the prompt from the prompt file, generates a response in a persistent session,
    converts it to speech, plays the audio, and writes the status to the status file.
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

        response = ollama.chat(
            model="gemma3:4b",
            messages=[{"role": "user", "content": SYSTEM_PROMPT + prompt}],
        )

        # Extract the response text
        response_text = response["message"]["content"].strip()

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
    except Exception as e:
        print(f"Error processing prompt: {e}")
        # Write failure status
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write("false")
    finally:
        # Clean up the prompt file after processing
        if os.path.exists(PROMPT_FILE):
            os.remove(PROMPT_FILE)


def main():
    """
    Continuously monitors the prompt file for new prompts and processes them.
    """
    print("Starting Ollama file processor...")
    if os.path.exists(PROMPT_FILE):
        os.remove(PROMPT_FILE)
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)

    while True:
        process_prompt()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
