"""Step 2: stream tokens as they are generated.

With stream=True the response is a sequence of Server-Sent Events. Ask for
stream_options.include_usage to get the token counts in a final chunk.

    python examples/02_streaming.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("TUNAI_BASE_URL", "https://api.tunai.tn/v1"),
    api_key=os.environ["TUNAI_API_KEY"],
)

stream = client.chat.completions.create(
    model=os.getenv("TUNAI_CHAT_MODEL", "gpt-5.4-mini"),
    messages=[{"role": "user", "content": "Describe Sidi Bou Said in three short sentences."}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.usage:  # the last chunk, with empty choices
        print(f"\n\ntokens: {chunk.usage.prompt_tokens} in, {chunk.usage.completion_tokens} out")
