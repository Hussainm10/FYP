"""
One-time GPU embedding + indexing script for AYAH-level vectors.

- Loads ayahs from JSON
- Embeds with multilingual-e5-base (GPU if available)
- Creates Qdrant collection (if missing)
- Upserts vectors + rich payload

Usage (from project root):
    python -m indexing.index_ayahs --smoke
    python -m indexing.index_ayahs
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from typing import Optional
from sentence_transformers import SentenceTransformer  # fallback for non-OpenAI embeddings
import os
from core import settings

from embeddings.openai_embedder import OpenAIEmbedder

from tqdm import tqdm

from .config import QdrantConfig, EmbedConfig, Paths
from .data_models import AyahRecord


def batched(items: List[AyahRecord], size: int) -> Iterable[List[AyahRecord]]:
    """Yield lists of length `size` from a list of AyahRecord."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_ayahs(path: str) -> List[AyahRecord]:
    """Load and validate ayahs from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [AyahRecord(**row) for row in raw]


def ensure_collection(client: QdrantClient, cfg: QdrantConfig) -> None:
    """Create the AYAH collection if missing (idempotent)."""
    names = {c.name for c in client.get_collections().collections}
    if cfg.ayah_collection not in names:
        client.create_collection(
            collection_name=cfg.ayah_collection,
            vectors_config=VectorParams(
                size=cfg.ayah_vector_dim,
                distance=Distance[cfg.distance],
            ),
        )


def to_point(r: AyahRecord, vec: List[float]) -> PointStruct:
    """
    Convert an AyahRecord + embedding into a Qdrant point.

    Qdrant IDs must be uint or UUID → we use stable UUIDv5 from human id "ayah:{chunk_id}".
    """
    payload = {
        "chunk_id": r.chunk_id,                     # keep original
        "surah_num": r.surah_num,
        "ayah_num": r.ayah_num,
        "surah_name_ar": r.surah_info.name_ar,
        "surah_name_en": r.surah_info.name_en,
        "revealed_in": r.surah_info.revealed_in,
        "arabic": r.arabic,
        "transliteration": r.transliteration,
        "translations": r.translations,
        "tafsirs": r.tafsirs,
    }
    qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ayah:{r.chunk_id}"))
    return PointStruct(id=qdrant_id, vector=vec, payload=payload)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Index first 20 ayahs only")
    args = parser.parse_args()

    qcfg = QdrantConfig()
    # Use environment overrides for embedding device; defaults defined in EmbedConfig
    ecfg = EmbedConfig(device=os.getenv("EMBED_DEVICE", "cuda"))  # switch to "cpu" on non-GPU servers
    paths = Paths()

    print("[load] reading JSON ...")
    ayahs = load_ayahs(paths.ayahs_json)
    if args.smoke:
        ayahs = ayahs[:20]
        print(f"[load] SMOKE MODE: {len(ayahs)} ayahs")

    # Initialise the embedder. Prefer OpenAI embeddings when an API key is provided.
    openai_key = settings.openai_api_key
    if openai_key:
        embedder: Optional[object] = OpenAIEmbedder(
            model=settings.openai_embed_model,
            api_key=openai_key,
            timeout=settings.openai_timeout_s,
            batch_size=ecfg.batch_size,
        )
        print(f"[embed] Using OpenAI model: {settings.openai_embed_model}")
    else:
        # Fall back to SentenceTransformers for local/offline embedding
        embedder = SentenceTransformer(ecfg.model_name, device=ecfg.device)
        print(f"[embed] Using SentenceTransformer: {ecfg.model_name} on {ecfg.device}")

    client = QdrantClient(host=qcfg.host, port=qcfg.port)
    ensure_collection(client, qcfg)

    print("[index] embedding + upserting ...")
    total = len(ayahs)
    for chunk in tqdm(batched(ayahs, ecfg.batch_size), total=(total // ecfg.batch_size + 1)):
        texts = [a.embedding_text() for a in chunk]
        # Generate embeddings depending on backend
        if isinstance(embedder, SentenceTransformer):
            vectors = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
        else:
            vectors = embedder.embed_texts(texts)
        points = [to_point(a, v) for a, v in zip(chunk, vectors)]
        client.upsert(collection_name=qcfg.ayah_collection, points=points)

    print(f"[done] indexed {total} ayahs into '{qcfg.ayah_collection}' ✅")


if __name__ == "__main__":
    main()
