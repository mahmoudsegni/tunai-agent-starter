"""Step 4: find the right passage with embeddings, then order the shortlist with a reranker.

/v1/embeddings is OpenAI-compatible, so it goes through the SDK. /v1/rerank is
not an OpenAI endpoint (it takes Cohere's and Jina's request shape), so it is a
plain HTTPS POST.

    python examples/04_embeddings_and_rerank.py
"""

import math
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

base_url = os.getenv("TUNAI_BASE_URL", "https://api.tunai.tn/v1")
api_key = os.environ["TUNAI_API_KEY"]
client = OpenAI(base_url=base_url, api_key=api_key)

documents = [
    "Shipping is free on orders of 150 TND or more, anywhere in Tunisia.",
    "Guellala jars are unglazed and keep water cool by evaporation.",
    "Grand Tunis deliveries take 1 to 3 business days with Aramex.",
    "Card payments go through ClicToPay; we never see your card number.",
]
query = "Do I pay for delivery if my order is 200 dinars?"

# Embed the documents and the query in one request, then compare by cosine similarity.
embedding_model = os.getenv("TUNAI_EMBEDDING_MODEL", "text-embedding-3-small")
response = client.embeddings.create(model=embedding_model, input=[*documents, query])
*doc_vectors, query_vector = [item.embedding for item in response.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


print("embedding similarity:")
for score, doc in sorted(
    ((cosine(query_vector, v), d) for v, d in zip(doc_vectors, documents, strict=True)),
    reverse=True,
):
    print(f"  {score:.3f}  {doc}")

# A reranker reads the query and each document together, which judges relevance better.
rerank = httpx.post(
    f"{base_url}/rerank",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": os.getenv("TUNAI_RERANK_MODEL") or "Qwen/Qwen3-Reranker-4B",
        "query": query,
        "documents": documents,
        "top_n": 2,
    },
    timeout=60,
)
rerank.raise_for_status()

print("\nreranked (top 2):")
for result in rerank.json()["results"]:
    print(f"  {result['relevance_score']:.3f}  {documents[result['index']]}")
