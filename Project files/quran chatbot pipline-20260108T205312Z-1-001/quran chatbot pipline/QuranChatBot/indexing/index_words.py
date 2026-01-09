"""
One-time embedding + indexing for WORD-level entries.

- Loads words from JSON
- Embeds with OpenAI text-embedding-3-small (preferred if OPENAI_API_KEY is set)
- Creates Qdrant 'words' collection (if missing)
- Upserts vectors + payload (root/lemma/pos, token index, links)

Usage (from project root):
    python -m indexing.index_words --smoke
    python -m indexing.index_words
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.http.exceptions import ResponseHandlingException

from sentence_transformers import SentenceTransformer  # fallback only
from tqdm import tqdm

from core import settings
from embeddings.openai_embedder import OpenAIEmbedder

from .config import QdrantConfig, EmbedConfig, Paths
from .data_models import WordRecord


def batched(items: List[WordRecord], size: int) -> Iterable[List[WordRecord]]:
    """Yield fixed-size chunks from a list of WordRecord."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_words(path: str, limit: int | None = None) -> List[WordRecord]:
    """Load and validate words from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit is not None:
        raw = raw[:limit]
    return [WordRecord(**row) for row in raw]


def ensure_collection(client: QdrantClient, cfg: QdrantConfig) -> None:
    """Create the WORDS collection if missing."""
    names = {c.name for c in client.get_collections().collections}
    if cfg.word_collection not in names:
        client.create_collection(
            collection_name=cfg.word_collection,
            vectors_config=VectorParams(
                size=cfg.word_vector_dim,
                distance=Distance[cfg.distance],
            ),
        )


def to_point(w: WordRecord, vec: List[float]) -> PointStruct:
    """
    Convert a WordRecord + embedding into a Qdrant point.

    Qdrant IDs must be uint or UUID → we use stable UUIDv5 from "word:{word_id}".
    """
    payload = {
        "word_id": w.word_id,
        "surah": w.surah,
        "ayah": w.ayah,
        "token_index": w.token_index(),
        "arabic": w.arabic,
        "transliteration": w.transliteration,
        "segmented": w.Segmented_Word,
        "gloss": {
            "en": w.english,
            "ur": w.urdu,
            "fa": w.farsi,
            "ps": w.pashto,
        },
        "root": w.root,
        "lemma": w.lemma,
        "root_norm": w.norm_root(),
        "lemma_norm": w.norm_lemma(),
        "pos": w.pos,
    }
    qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"word:{w.word_id}"))
    return PointStruct(id=qdrant_id, vector=vec, payload=payload)


def upsert_with_retry(
    client: QdrantClient,
    collection: str,
    points: List[PointStruct],
    max_retries: int = 4,
) -> None:
    """
    Upsert with retry/backoff to survive occasional Qdrant timeouts.

    This is intentionally narrow (only wraps upsert) to keep changes minimal.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            client.upsert(collection_name=collection, points=points)
            return
        except (ResponseHandlingException, TimeoutError) as e:
            last_err = e
            # exponential-ish backoff: 2, 4, 8, 16 seconds
            sleep_s = min(2 ** (attempt + 1), 16)
            print(f"[warn] Qdrant upsert timed out (attempt {attempt+1}/{max_retries+1}). "
                  f"Sleeping {sleep_s}s then retrying...")
            time.sleep(sleep_s)

    # If we exhausted retries, raise the last error
    raise last_err if last_err else RuntimeError("Unknown upsert failure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Index first ~1k word tokens")
    args = parser.parse_args()

    qcfg = QdrantConfig()

    # IMPORTANT: never default to cuda on Windows unless you explicitly set it.
    # If you really want CUDA later, set: $env:EMBED_DEVICE="cuda"
    ecfg = EmbedConfig(device=os.getenv("EMBED_DEVICE", "cpu"))

    paths = Paths()

    limit = 1000 if args.smoke else None
    print("[load] reading words JSON ...")
    words = load_words(paths.words_json, limit=limit)
    print(f"[load] total words loaded: {len(words)}")

    # Embedder
    openai_key = settings.openai_api_key
    if openai_key:
        embedder: object = OpenAIEmbedder(
            model=settings.openai_embed_model,
            api_key=openai_key,
            timeout=settings.openai_timeout_s,
            batch_size=ecfg.batch_size,
        )
        print(f"[embed] Using OpenAI model: {settings.openai_embed_model}")
    else:
        embedder = SentenceTransformer(ecfg.model_name, device=ecfg.device)
        print(f"[embed] Using SentenceTransformer: {ecfg.model_name} on {ecfg.device}")

    # Qdrant client: longer timeout + skip compatibility check (your versions mismatch)
    client = QdrantClient(
        host=qcfg.host,
        port=qcfg.port,
        timeout=180,                 # increased from default
        check_compatibility=False,   # avoids warning / strict checks
    )
    ensure_collection(client, qcfg)

    # Key fix for your timeout: upsert smaller batches than embedding batches.
    # Embedding can be 128/256, but upserts should be smaller on Windows/docker.
    UPSERT_BATCH = int(os.getenv("QDRANT_UPSERT_BATCH", "32"))

    print(f"[index] embedding + upserting words ... (embed_batch={ecfg.batch_size}, upsert_batch={UPSERT_BATCH})")
    total = len(words)

    # We embed in ecfg.batch_size but upsert in UPSERT_BATCH
    for embed_chunk in tqdm(batched(words, ecfg.batch_size), total=(total // ecfg.batch_size + 1)):
        texts = [w.embedding_text() for w in embed_chunk]

        if isinstance(embedder, SentenceTransformer):
            vectors = embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
        else:
            vectors = embedder.embed_texts(texts)

        # split the points into smaller upsert payloads
        points_all = [to_point(w, v) for w, v in zip(embed_chunk, vectors)]
        for upsert_chunk in batched(points_all, UPSERT_BATCH):
            upsert_with_retry(client, qcfg.word_collection, upsert_chunk)

    print(f"[done] indexed {total} word tokens into '{qcfg.word_collection}' ✅")


if __name__ == "__main__":
    main()
