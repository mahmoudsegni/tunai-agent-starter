"""Step 3: one tool call, by hand. This is the whole idea behind an agent.

1. Send the question with a list of tools (name, description, JSON Schema).
2. The model replies with a tool call instead of text.
3. You run the function and send the result back as a "tool" message.
4. The model answers using that result.

The agent in src/tunai_agent/agent.py is this, in a loop, with error handling.

    python examples/03_tool_call.py
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("TUNAI_BASE_URL", "https://api.tunai.tn/v1"),
    api_key=os.environ["TUNAI_API_KEY"],
)
model = os.getenv("TUNAI_CHAT_MODEL", "gpt-5.4-mini")

SHIPPING_FEES_TND = {"tunis": 7, "ariana": 7, "ben arous": 7, "manouba": 7}


def shipping_fee(governorate: str) -> dict:
    fee = SHIPPING_FEES_TND.get(governorate.strip().lower(), 9)
    return {"governorate": governorate, "fee_tnd": fee}


tools = [
    {
        "type": "function",
        "function": {
            "name": "shipping_fee",
            "description": "Shipping fee in TND for delivery to a Tunisian governorate.",
            "parameters": {
                "type": "object",
                "properties": {"governorate": {"type": "string", "description": "e.g. Sfax"}},
                "required": ["governorate"],
            },
        },
    }
]

messages = [{"role": "user", "content": "How much is delivery to Sfax?"}]

# 1-2. The model decides to call the tool.
first = client.chat.completions.create(model=model, messages=messages, tools=tools)
reply = first.choices[0].message
if not reply.tool_calls:
    raise SystemExit(f"The model answered without the tool: {reply.content}")

call = reply.tool_calls[0]
print(f"model asked for: {call.function.name}({call.function.arguments})")

# 3. Run it, and send back both the model's tool call and our result.
result = shipping_fee(**json.loads(call.function.arguments))
messages.append(
    {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
        ],
    }
)
messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

# 4. The model answers with the result in hand.
final = client.chat.completions.create(model=model, messages=messages, tools=tools)
print(f"answer: {final.choices[0].message.content}")
