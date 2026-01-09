"""Embedding backends for the QuranChatBot.

Currently the project supports OpenAI's embedding API through the
``OpenAIEmbedder`` class. Additional embedders (e.g. SentenceTransformer) can
be added here in the future to allow pluggable embedding strategies.
"""

from .openai_embedder import OpenAIEmbedder  # noqa: F401
