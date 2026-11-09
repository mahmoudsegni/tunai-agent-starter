"""Step 1: one chat completion.

TunAI speaks OpenAI's API, so the official SDK works unchanged: point base_url
at TunAI and pass your TunAI key.

    python examples/01_first_request.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("TUNAI_BASE_URL", "https://api.tunai.tn/v1"),
    api_key=os.environ["TUNAI_API_KEY"],
)

response = client.chat.completions.create(
    model=os.getenv("TUNAI_CHAT_MODEL", "gpt-5.4-mini"),
    messages=[
        {"role": "system", "content": "You answer in one sentence."},
        {"role": "user", "content": "What is the capital of Tunisia?"},
    ],
)

print(response.choices[0].message.content)
print(f"\ntokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
