"""Core configuration and utilities for the QuranChatBot package.

This module exposes a global ``settings`` object that reads values from the
environment with sensible defaults. Centralising configuration in a single
location avoids scattering environment lookups throughout the codebase and
makes it easier to adapt the application to different environments (e.g. local
development vs. production).

The settings defined here should be considered the source of truth for
deployment parameters such as API keys, model identifiers, Qdrant details and
collection names. When adding new configuration values, prefer to declare them
here and document them in ``.env.example``.
"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables.

    Each attribute corresponds to one or more environment variables. When an
    environment variable is not defined, a sensible default is used. You can
    override these defaults by creating a ``.env`` file in your project root
    (see ``.env.example`` for guidance) or by exporting variables in your shell
    before starting the application.
    """

    # OpenAI API configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    openai_timeout_s: int = int(os.getenv("OPENAI_TIMEOUT_S", "60"))

    # Qdrant connection
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    # Collection names and vector dimensions
    qdrant_collection_ayahs: str = os.getenv(
        "QDRANT_COLLECTION_AYAHS", "quran_ayahs_openai_v1"
    )
    qdrant_collection_words: str = os.getenv(
        "QDRANT_COLLECTION_WORDS", "quran_words_openai_v1"
    )
    # Vector dimension for OpenAI text-embedding-3-small (per OpenAI docs)
    # Should only be changed if using a different embedding model.
    vector_dim: int = int(os.getenv("VECTOR_DIM", "1536"))

    # Search configuration
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "7"))

    # Miscellaneous
    # The path to the ayah JSON file used by the router. This defaults to the
    # location within the package but can be overridden when deploying
    # FastAPI on a remote server that has a different working directory.
    ayahs_json_path: str = os.getenv(
        "AYAHS_JSON_PATH",
        str(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "metadata", "ayahs_collection.json"
            )
        ),
    )


# Expose a singleton settings instance for convenience
settings = Settings()
