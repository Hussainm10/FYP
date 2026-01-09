"""
Runtime configuration for embedding + indexing.
"""
from __future__ import annotations
from dataclasses import dataclass
import os

from core import settings

@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for connecting to Qdrant and defining collections.

    Defaults are loaded from ``core.settings.Settings``. You can override
    any of these values by setting the corresponding environment variable as
    documented in ``.env.example``. The embedding dimension defaults to 1536
    which matches the OpenAI ``text-embedding-3-small`` model. If you use
    another embedding model, update ``VECTOR_DIM`` accordingly in the
    environment.
    """

    host: str = settings.qdrant_host
    port: int = settings.qdrant_port
    # Ayah collection name and vector dimension
    ayah_collection: str = settings.qdrant_collection_ayahs
    ayah_vector_dim: int = settings.vector_dim
    # Word collection name and vector dimension
    word_collection: str = settings.qdrant_collection_words
    word_vector_dim: int = settings.vector_dim
    # distance metric (cosine for OpenAI embeddings)
    distance: str = "COSINE"


@dataclass(frozen=True)
class EmbedConfig:
    """Embedding configuration.

    The ``model_name`` is retained for backward compatibility with the
    SentenceTransformers workflow but is not used when OpenAI embeddings are
    configured. ``batch_size`` still controls how many items are processed
    per request for both SentenceTransformers and OpenAI.
    """

    model_name: str = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
    device: str = os.getenv("EMBED_DEVICE", "cuda")  # switch to "cpu" on non-GPU servers
    batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "128"))


@dataclass(frozen=True)
class Paths:
    ayahs_json: str = "./metadata/ayahs_collection.json"
    words_json: str = "./metadata/words_collection.json"
