import re
from pathlib import Path

import pytest

from tunai_agent.client import RerankResult
from tunai_agent.config import Settings
from tunai_agent.errors import TunAIError
from tunai_agent.retrieval import KnowledgeBase, cosine, load_chunks, split_markdown

from .conftest import TEST_KEY

KB_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"
VOCABULARY = ["shipping", "free", "fee", "refund", "card", "jar", "water", "damaged", "cash"]


class KeywordEmbeddingClient:
    """Embeds text as keyword counts, so similarity is predictable in tests."""

    def __init__(self, rerank_model: str | None = None, rerank=None) -> None:
        self.settings = Settings(api_key=TEST_KEY, rerank_model=rerank_model)
        self.embed_calls = 0
        self._rerank = rerank

    def embed(self, texts):
        self.embed_calls += 1
        return [
            [len(re.findall(word, text.lower())) + 0.01 for word in VOCABULARY] for text in texts
        ]

    def rerank(self, query, documents, **kwargs):
        return self._rerank(query, documents, **kwargs)


def test_split_markdown_keeps_headings_and_respects_size():
    text = "# Title\n\nIntro.\n\n## Fees\n\n" + "\n\n".join(["word " * 50] * 6)
    chunks = split_markdown("doc.md", text, max_chars=600)

    assert chunks[0].heading == "Title" and chunks[0].text == "Intro."
    fee_chunks = [c for c in chunks if c.heading == "Fees"]
    assert len(fee_chunks) > 1
    assert all(len(c.text) <= 600 for c in fee_chunks)
    assert fee_chunks[0].citation == "doc.md › Fees"


def test_loads_the_bundled_knowledge_base():
    chunks = load_chunks(KB_DIR)
    assert {c.source for c in chunks} == {
        "catalog.md",
        "payments.md",
        "product-care.md",
        "returns.md",
        "shipping.md",
    }


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine([0, 0], [1, 1]) == 0


def test_embedding_only_search_when_no_reranker(tmp_path):
    client = KeywordEmbeddingClient(rerank_model=None)
    kb = KnowledgeBase.build(client, KB_DIR, cache_dir=tmp_path)

    passages = kb.search("free shipping fee", top_k=2)

    assert len(passages) == 2
    assert all(p.ranked_by == "embedding" for p in passages)
    assert "shipping.md" in {p.chunk.source for p in passages}


def test_reranker_decides_the_final_order(tmp_path):
    def rerank(query, documents, top_n, **kwargs):
        # Reverse the shortlist, to prove the reranker's order wins.
        return [
            RerankResult(index=i, relevance_score=1.0 - n / 10)
            for n, i in enumerate(reversed(range(len(documents))))
        ][:top_n]

    client = KeywordEmbeddingClient(rerank_model="Qwen/Qwen3-Reranker-4B", rerank=rerank)
    kb = KnowledgeBase.build(client, KB_DIR, cache_dir=tmp_path)

    plain = KnowledgeBase(KeywordEmbeddingClient(), kb.chunks, kb.vectors).search("refund card", 8)
    reranked = kb.search("refund card", top_k=2, candidates=8)

    assert [p.ranked_by for p in reranked] == ["rerank", "rerank"]
    assert reranked[0].chunk == plain[-1].chunk


def test_falls_back_to_embedding_order_when_rerank_fails(tmp_path, caplog):
    def rerank(*args, **kwargs):
        raise TunAIError(503, "upstream unavailable")

    client = KeywordEmbeddingClient(rerank_model="Qwen/Qwen3-Reranker-4B", rerank=rerank)
    passages = KnowledgeBase.build(client, KB_DIR, cache_dir=tmp_path).search("damaged jar")

    assert passages and all(p.ranked_by == "embedding" for p in passages)
    assert "Rerank failed" in caplog.text


def test_index_is_cached_until_the_documents_change(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "a.md").write_text("# A\n\nShipping is free.")
    cache = tmp_path / "cache"

    first = KeywordEmbeddingClient()
    KnowledgeBase.build(first, kb_dir, cache_dir=cache)
    second = KeywordEmbeddingClient()
    KnowledgeBase.build(second, kb_dir, cache_dir=cache)
    assert (first.embed_calls, second.embed_calls) == (1, 0)

    (kb_dir / "a.md").write_text("# A\n\nShipping costs 7 TND.")
    third = KeywordEmbeddingClient()
    KnowledgeBase.build(third, kb_dir, cache_dir=cache)
    assert third.embed_calls == 1
