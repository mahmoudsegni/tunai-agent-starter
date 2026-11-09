# TunAI API cheatsheet

Every endpoint the agent uses, as `curl` commands you can paste. Set your key first:

```bash
export TUNAI_API_KEY="sk-tunai-..."   # from https://console.tunai.tn
```

- **Base URL:** `https://api.tunai.tn/v1`
- **Auth:** `Authorization: Bearer $TUNAI_API_KEY` on every request
- **Format:** OpenAI's request and response shapes, so any OpenAI SDK works with `base_url` set to TunAI

---

## List models

```bash
curl https://api.tunai.tn/v1/models -H "Authorization: Bearer $TUNAI_API_KEY"
```

```json
{"object": "list", "data": [{"id": "gpt-5.4-mini", "object": "model", "created": 0, "owned_by": "openai"}, ...]}
```

Chat, embedding, rerank and transcription models all appear in this list. Ids are case-sensitive.

---

## Chat completions

```bash
curl https://api.tunai.tn/v1/chat/completions \
  -H "Authorization: Bearer $TUNAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [{"role": "user", "content": "Bonjour !"}],
    "max_tokens": 200
  }'
```

Supported parameters include `messages`, `stream`, `stream_options`, `temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `stop`, `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, `seed`, `presence_penalty` and `frequency_penalty`. Parameters a model does not support are dropped (for example `temperature` on reasoning models). `tools` is the exception: a model that cannot take tools returns `400 unsupported_parameter` rather than ignoring them.

### Streaming

Add `"stream": true` (and `"stream_options": {"include_usage": true}` for token counts). The response is Server-Sent Events: `data: {chat.completion.chunk}` lines, then `data: [DONE]`. During a long silence (a reasoning model thinking) the stream sends `: keepalive` comment lines, which SSE clients ignore.

### Function calling

```bash
curl https://api.tunai.tn/v1/chat/completions \
  -H "Authorization: Bearer $TUNAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [{"role": "user", "content": "Where is order DJ-1002?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_order_status",
        "description": "Look up an order by its number.",
        "parameters": {
          "type": "object",
          "properties": {"order_id": {"type": "string"}},
          "required": ["order_id"]
        }
      }
    }]
  }'
```

The reply's `choices[0].message.tool_calls` holds `{id, type: "function", function: {name, arguments}}`, where `arguments` is a JSON *string*. Send back the assistant message unchanged, then one `{"role": "tool", "tool_call_id": "<id>", "content": "<result>"}` per call, and request again. See [`examples/03_tool_call.py`](../examples/03_tool_call.py).

---

## Embeddings

```bash
curl https://api.tunai.tn/v1/embeddings \
  -H "Authorization: Bearer $TUNAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-3-small", "input": ["first text", "second text"]}'
```

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, ...]}, ...],
 "model": "text-embedding-3-small",
 "usage": {"prompt_tokens": 4, "total_tokens": 4}}
```

`input` takes a string or a list of strings. `dimensions` shortens the vectors on models that support it. Other embedding models include `text-embedding-3-large`, `BAAI/bge-m3` and `Qwen/Qwen3-Embedding-8B`.

---

## Rerank

Not an OpenAI endpoint: it takes the request shape of Cohere's and Jina's rerank APIs. Call it with any HTTP client.

```bash
curl https://api.tunai.tn/v1/rerank \
  -H "Authorization: Bearer $TUNAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-4B",
    "query": "Is delivery free above 150 TND?",
    "documents": ["Shipping is free on orders of 150 TND or more.", "Jars keep water cool."],
    "top_n": 1,
    "instruction": "Find passages that answer the question."
  }'
```

```json
{"id": "...", "model": "Qwen/Qwen3-Reranker-4B",
 "results": [{"index": 0, "relevance_score": 0.998}],
 "usage": {"prompt_tokens": 31, "total_tokens": 31}}
```

| Field | |
|---|---|
| `documents` | 1 to 1000 strings (or `{"text": ...}` objects) |
| `top_n` | Return only the best N. Omit for all, most relevant first |
| `return_documents` | `true` adds each document's text to its result |
| `instruction` | Optional task description for the Qwen3 rerankers |

`index` points into the `documents` you sent. Models: `Qwen/Qwen3-Reranker-0.6B`, `-4B`, `-8B` (larger is more accurate, slower and dearer).

---

## Speech to text

```bash
curl https://api.tunai.tn/v1/audio/transcriptions \
  -H "Authorization: Bearer $TUNAI_API_KEY" \
  -F model="openai/whisper-large-v3-turbo" \
  -F file="@voice-note.m4a" \
  -F language="ar"
```

```json
{"text": "..."}
```

One file of up to 25 MB: flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, opus, wav or webm. `response_format` is `json` (default), `text`, `srt`, `vtt` or `verbose_json`. `language` (ISO 639-1, e.g. `ar`, `fr`, `en`) and `prompt` are optional. Billed per second of audio.

---

## Headers worth reading

| Header | Meaning |
|---|---|
| `x-request-id` | Unique id of the request. Quote it to support. |
| `x-ratelimit-limit-requests` | Requests per minute this key allows (when the key has a limit) |
| `x-ratelimit-remaining-requests` | Requests left in the current minute |
| `x-ratelimit-reset-requests` | When the window resets, e.g. `42s` |

## Errors

Errors use OpenAI's envelope. Branch on `error.code`:

```json
{"error": {"message": "Insufficient balance. Top up your account to continue.",
           "type": "insufficient_quota", "param": null, "code": "insufficient_quota"}}
```

| Status | `code` | Meaning |
|---|---|---|
| 400 | `unsupported_parameter`, `invalid_value` | The request is not valid for this model |
| 401 | `invalid_api_key` | Missing, wrong, revoked or expired key |
| 402 | `insufficient_quota` | Balance too low: top up |
| 402 | `budget_exceeded` | The key reached the monthly budget set on it |
| 403 | `insufficient_scope` | The key is restricted away from this endpoint |
| 404 | `model_not_found` | Unknown model id, or wrong model type for the endpoint |
| 429 | `rate_limit_exceeded` | Over the key's requests per minute |
| 5xx | `upstream_error`, `model_unavailable` | The upstream provider failed. Retry with backoff |

Retry `429` and `5xx` with exponential backoff (the OpenAI SDKs do this for you). Do not retry `400`, `401`, `402`, `403` or `404`.

## Key scopes

A key can be limited to some endpoints when you create it. A key with no scopes can call everything.

| Scope | Endpoints |
|---|---|
| `chat` | `/v1/chat/completions` |
| `embeddings` | `/v1/embeddings` |
| `rerank` | `/v1/rerank` |
| `audio` | `/v1/audio/transcriptions` |
| `systemone` | `/v1/systemone` (Jev document understanding, not used here) |
