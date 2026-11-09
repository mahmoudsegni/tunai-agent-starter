"""Step 5: speech to text, so customers can send a voice note instead of typing.

/v1/audio/transcriptions is OpenAI-compatible. Whisper handles Arabic, French
and English; pass `language` when you know it to skip detection.

    python examples/05_transcribe.py path/to/voice-note.m4a [language]
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if len(sys.argv) < 2:
    raise SystemExit("usage: python examples/05_transcribe.py AUDIO_FILE [language]")

client = OpenAI(
    base_url=os.getenv("TUNAI_BASE_URL", "https://api.tunai.tn/v1"),
    api_key=os.environ["TUNAI_API_KEY"],
)

options = {"language": sys.argv[2]} if len(sys.argv) > 2 else {}
with open(sys.argv[1], "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model=os.getenv("TUNAI_TRANSCRIPTION_MODEL", "openai/whisper-large-v3-turbo"),
        file=audio,
        **options,
    )

print(transcript.text)
