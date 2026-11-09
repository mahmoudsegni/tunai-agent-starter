"""Retrieval over a folder of Markdown files: embeddings to recall, a reranker to order.

Two stages, because each is good at a different thing:

1. **Recall** with ``/v1/embeddings``. Every chunk is embedded once (and cached
   on disk); a query is embedded and compared by cosine similarity. Cheap and
   fast, but similarity is a blunt measure of relevance.
2. **Precision** with ``/v1/rerank``. A cross-encoder reads the query and each
   candidate *together* and scores how well the candidate answers it. Slower
   per document, so it only sees the handful of candidates stage 1 kept.

If the reranker is disabled or fails, results fall back to stage 1's order, so
search degrades instead of breaking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .client import TunAIClient
from .errors import TunAIError

logger = logging.getLogger(__name__)

RERANK_INSTRUCTION = "Given a customer question, find the passages that answer it."
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    source: str
    """File name, e.g. ``shipping.md``."""
    heading: str
    """The section the text sits under."""
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source} › {self.heading}" if self.heading else self.source


@dataclass(frozen=True)
class Passage:
    chunk: Chunk
    score: float
    ranked_by: str
    """``"rerank"`` or ``"embedding"``: which stage produced the final order."""


def split_markdown(source: str, markdown: str, max_chars: int = 900) -> list[Chunk]:
    """Split a document at its headings, then split long sections at paragraph breaks.

    Each chunk keeps its heading, so a passage still makes sense out of context
    and can be cited precisely.
    """
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING.finditer(markdown))
    if not matches or matches[0].start() > 0:
        preamble = markdown[: matches[0].start()] if matches else markdown
        sections.append(("", preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((match.group(2).strip(), markdown[match.end() : end]))

    chunks: list[Chunk] = []
    for heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        window: list[str] = []
        for paragraph in paragraphs:
            if window and len("\n\n".join([*window, paragraph])) > max_chars:
                chunks.append(Chunk(source, heading, "\n\n".join(window)))
                window = []
            window.append(paragraph)
        if window:
            chunks.append(Chunk(source, heading, "\n\n".join(window)))
    return chunks


def load_chunks(directory: str | Path) -> list[Chunk]:
    paths = sorted(Path(directory).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no .md files in {directory}")
    return [
        chunk
        for path in paths
        for chunk in split_markdown(path.name, path.read_text(encoding="utf-8"))
    ]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class KnowledgeBase:
    """Embedded chunks plus the search pipeline described in the module docstring."""

    def __init__(
        self, client: TunAIClient, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk is required")
        self.client = client
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def build(
        cls, client: TunAIClient, directory: str | Path, cache_dir: str | Path | None = ".cache"
    ) -> KnowledgeBase:
        """Load and embed every chunk, reusing the on-disk cache when nothing changed.

        The cache key covers the embedding model and every chunk's text, so
        editing a document or switching models re-embeds; anything else is free.
        """
        chunks = load_chunks(directory)
        model = client.settings.embedding_model
        key = _fingerprint(model, chunks)
        cache_file = Path(cache_dir) / f"index-{key[:16]}.json" if cache_dir else None

        if cache_file and cache_file.is_file():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                logger.info("Loaded %d embedded chunks from %s", len(chunks), cache_file)
                return cls(client, chunks, cached["vectors"])

        logger.info("Embedding %d chunks with %s", len(chunks), model)
        vectors = client.embed([f"{c.citation}\n\n{c.text}" for c in chunks])
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"key": key, "model": model, "vectors": vectors}))
        return cls(client, chunks, vectors)

    def search(self, query: str, top_k: int = 3, candidates: int = 8) -> list[Passage]:
        """The ``top_k`` passages that best answer ``query``."""
        query_vector = self.client.embed([query])[0]
        scored = sorted(
            (
                (cosine(query_vector, vector), chunk)
                for vector, chunk in zip(self.vectors, self.chunks, strict=True)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[: max(candidates, top_k)]

        if self.client.settings.rerank_model and len(scored) > 1:
            shortlist = [chunk for _, chunk in scored]
            try:
                ranked = self.client.rerank(
                    query,
                    [f"{c.citation}\n\n{c.text}" for c in shortlist],
                    top_n=top_k,
                    instruction=RERANK_INSTRUCTION,
                )
                return [Passage(shortlist[r.index], r.relevance_score, "rerank") for r in ranked]
            except (TunAIError, httpx.HTTPError) as exc:
                logger.warning("Rerank failed, using embedding order instead: %s", exc)

        return [Passage(chunk, score, "embedding") for score, chunk in scored[:top_k]]


def _fingerprint(model: str, chunks: Sequence[Chunk]) -> str:
    digest = hashlib.sha256(model.encode())
    for chunk in chunks:
        digest.update(b"\0" + chunk.citation.encode() + b"\0" + chunk.text.encode())
    return digest.hexdigest()


def passages_as_json(passages: Sequence[Passage]) -> dict[str, Any]:
    return {
        "results": [
            {"source": p.chunk.citation, "text": p.chunk.text, "score": round(p.score, 4)}
            for p in passages
        ]
    }
