"""OpenAI embedding backend for Qur'an search.

This module wraps the OpenAI embeddings API to produce vector representations
for lists of strings. It handles batching, basic retry logic with
exponential backoff, and ordering of returned embeddings. The OpenAI API
requires an API key which should be provided via the environment variable
``OPENAI_API_KEY`` or via the ``api_key`` parameter when instantiating the
embedder.

Note: The embedding model ``text-embedding-3-small`` returns vectors of
dimension 1536. If you change the model to a different embedding model,
update the ``VECTOR_DIM`` in ``core/settings.py`` accordingly and ensure
your Qdrant collections are created with the matching dimension.
"""

from __future__ import annotations

import time
from typing import Iterable, List

import openai

from core import settings


class OpenAIEmbedder:
    """Batched embedder using OpenAI's embedding API.

    Args:
        model: The OpenAI embedding model identifier (defaults to
            ``settings.openai_embed_model``).
        api_key: Your OpenAI API key (defaults to ``settings.openai_api_key``).
        timeout: Per-request timeout in seconds.
        batch_size: Maximum number of strings to embed per API call.
        max_retries: Number of times to retry a failed request.

    Usage:

    ```python
    embedder = OpenAIEmbedder()
    vectors = embedder.embed_texts(["hello", "world"])
    ```
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        batch_size: int = 50,
        max_retries: int = 5,
    ) -> None:
        self.model = model or settings.openai_embed_model
        self.api_key = api_key or settings.openai_api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_retries = max_retries

        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key is missing. Please set OPENAI_API_KEY in your environment or .env file."
            )

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed a batch of texts into vectors.

        Returns a list of lists of floats corresponding to the embeddings of
        each input string. The order of the returned vectors matches the
        order of the input.
        """
        # Configure API key for the current thread
        openai.api_key = self.api_key
        items = list(texts)
        if not items:
            return []
        vectors: List[List[float]] = []

        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            # Retry with exponential backoff
            attempt = 0
            while True:
                try:
                    response = openai.embeddings.create(
                        input=batch,
                        model=self.model,
                        timeout=self.timeout,
                    )
                    # The API returns embeddings in the order of the inputs
                    # Each element in response.data has an ``embedding`` and an ``index``.
                    # Sort by index to ensure correct ordering.
                    sorted_data = sorted(response.data, key=lambda d: d.index)
                    vectors.extend([d.embedding for d in sorted_data])
                    break
                except Exception:
                    attempt += 1
                    if attempt >= self.max_retries:
                        raise
                    # Exponential backoff with jitter
                    sleep_seconds = (2 ** attempt) + (0.1 * attempt)
                    time.sleep(sleep_seconds)
        return vectors
