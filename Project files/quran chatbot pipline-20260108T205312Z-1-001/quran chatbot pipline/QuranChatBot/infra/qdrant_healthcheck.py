"""
Qdrant health check — tolerant to all /healthz formats.

Run this after starting Qdrant with Docker to confirm:
1. The REST API is reachable.
2. Collections can be created and deleted.
"""

from __future__ import annotations

from typing import Dict, Any
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def health_status(url: str = "http://localhost:6333/healthz") -> Dict[str, Any]:
    """
    Call Qdrant's /healthz endpoint and handle any text/JSON variation.

    Args:
        url: health endpoint URL.

    Returns:
        Dict containing at least {"status": "..."}.
    """
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    try:
        data = resp.json()
        if isinstance(data, dict) and "status" in data:
            return {"status": str(data["status"])}
    except Exception:
        pass
    # fallback: plain text body
    text = (resp.text or "").strip()
    return {"status": text or f"http {resp.status_code}"}


def main() -> None:
    """Check Qdrant liveness and basic collection lifecycle."""
    print("[1] Checking Qdrant health via /healthz ...")
    status = health_status()
    print(status)

    # Accept both common variants
    status_text = status.get("status", "").strip().lower()
    if status_text not in ("ok", "healthz check passed"):
        raise SystemExit(
            "Qdrant not healthy. Check Docker logs: `docker compose logs -f qdrant`"
        )

    client = QdrantClient(host="localhost", port=6333)

    print("[2] Creating temporary collection...")
    client.recreate_collection(
        collection_name="tmp_check",
        vectors_config=VectorParams(size=8, distance=Distance.COSINE),
    )

    print("[3] Listing collections...")
    cols = client.get_collections()
    print([c.name for c in cols.collections])

    print("[4] Deleting temporary collection...")
    client.delete_collection("tmp_check")

    print("✅ Qdrant is healthy and ready to use!")


if __name__ == "__main__":
    main()
