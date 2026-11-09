"""Settings, read from the environment (and a local .env file, if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.tunai.tn/v1"
DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-4B"
DEFAULT_TRANSCRIPTION_MODEL = "openai/whisper-large-v3-turbo"

KEY_PREFIX = "sk-tunai-"


class ConfigError(RuntimeError):
    """The environment is missing something the agent needs."""


@dataclass(frozen=True)
class Settings:
    """Everything the agent reads from the environment.

    The API key is excluded from ``repr`` so it never ends up in a log line or a
    traceback by accident.
    """

    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rerank_model: str | None = DEFAULT_RERANK_MODEL
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL

    @property
    def masked_key(self) -> str:
        """The key with everything but its prefix and last four characters hidden."""
        return f"{self.api_key[: len(KEY_PREFIX)]}…{self.api_key[-4:]}"

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        """Build settings from environment variables.

        A ``.env`` file in the working directory is loaded first if it exists.
        Variables already set in the environment win over the file.

        Raises:
            ConfigError: if ``TUNAI_API_KEY`` is missing or malformed.
        """
        if env_file and Path(env_file).is_file():
            load_dotenv(env_file, override=False)

        api_key = os.environ.get("TUNAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "TUNAI_API_KEY is not set. Create a key at https://console.tunai.tn, "
                "then copy .env.example to .env and paste it there."
            )
        if not api_key.startswith(KEY_PREFIX):
            raise ConfigError(f"TUNAI_API_KEY should start with '{KEY_PREFIX}'.")

        rerank = os.environ.get("TUNAI_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip()
        return cls(
            api_key=api_key,
            base_url=os.environ.get("TUNAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            chat_model=os.environ.get("TUNAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
            embedding_model=os.environ.get("TUNAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            rerank_model=rerank or None,
            transcription_model=os.environ.get(
                "TUNAI_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL
            ),
        )
