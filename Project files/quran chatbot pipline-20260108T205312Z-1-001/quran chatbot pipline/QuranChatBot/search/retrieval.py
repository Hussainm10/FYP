"""
CPU-only retrieval utilities for Qur'an RAG.

Compatibility target:
- qdrant-client==1.9.1
- qdrant server == 1.9.x (docker image)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core import settings  # absolute import (Windows-safe)

from rapidfuzz import fuzz
from search.textnorm import is_arabic, normalize_arabic, normalize_latin

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from rerank.cross_encoder import CrossEncoderReranker

# Local embedding fallback (only used if OpenAI key not set)
from sentence_transformers import SentenceTransformer

# OpenAI embedder (your intended production path)
from embeddings.openai_embedder import OpenAIEmbedder


@dataclass(frozen=True)
class SearchConfig:
    qdrant_host: str = settings.qdrant_host
    qdrant_port: int = settings.qdrant_port
    ayah_collection: str = settings.qdrant_collection_ayahs
    word_collection: str = settings.qdrant_collection_words
    top_k: int = settings.default_top_k


class QuranSearcher:
    def __init__(self, cfg: Optional[SearchConfig] = None) -> None:
        self.cfg = cfg or SearchConfig()

        # IMPORTANT:
        # qdrant-client==1.9.1 does NOT support check_compatibility param.
        self.client = QdrantClient(host=self.cfg.qdrant_host, port=self.cfg.qdrant_port)

        # Prefer OpenAI embeddings when key is set; else fallback to local ST.
        if settings.openai_api_key:
            self.embedder = OpenAIEmbedder(
                model=settings.openai_embed_model,
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_s,
                batch_size=64,
            )
            print(f"[embed] Retrieval using OpenAI embeddings: {settings.openai_embed_model}")
        else:
            # fallback: local CPU embeddings
            model = "intfloat/multilingual-e5-small"
            self.embedder = SentenceTransformer(model, device="cpu")
            print(f"[embed] Retrieval using local SentenceTransformer: {model} (cpu)")

        # ---------------------------------------------------------
        # OpenAI block notes:
        # - This file expects OpenAI embeddings through OpenAIEmbedder.
        # - If you ever want to disable OpenAI temporarily: unset OPENAI_API_KEY
        # - If you later switch embedding models, you MUST reindex into a new collection.
        # ---------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        if isinstance(self.embedder, SentenceTransformer):
            vec = self.embedder.encode(
                [text], convert_to_numpy=True, normalize_embeddings=True
            ).tolist()[0]
            return vec
        return self.embedder.embed_texts([text])[0]

    def _qdrant_search(self, collection: str, vec: List[float], limit: int):
        """
        qdrant-client==1.9.1 uses `.search(...)` not `.query_points(...)`.
        """
        return self.client.search(
            collection_name=collection,
            query_vector=vec,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    # ---------- Ayah search ----------
    def search_ayahs(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        k = top_k or self.cfg.top_k
        vec = self._embed(query)

        res = self._qdrant_search(self.cfg.ayah_collection, vec, k)

        hits: List[Dict] = []
        for p in res:
            payload = p.payload or {}
            hits.append(
                {
                    "score": float(p.score),
                    "surah_num": payload.get("surah_num"),
                    "ayah_num": payload.get("ayah_num"),
                    "surah_name_en": payload.get("surah_name_en"),
                    "surah_name_ar": payload.get("surah_name_ar"),
                    "arabic": payload.get("arabic"),
                    "transliteration": payload.get("transliteration"),
                    "translations": payload.get("translations") or {},
                    "tafsirs": payload.get("tafsirs") or {},
                }
            )
        return hits

    # ---------- Word search ----------
    def search_words(self, query: str, top_k: int = 10) -> List[Dict]:
        vec = self._embed(query)
        res = self._qdrant_search(self.cfg.word_collection, vec, top_k)
        return [p.payload or {} for p in res]

    def fetch_words(self, surah: int, ayah: int) -> List[Dict]:
        flt = Filter(
            must=[
                FieldCondition(key="surah", match=MatchValue(value=surah)),
                FieldCondition(key="ayah", match=MatchValue(value=ayah)),
            ]
        )
        scrolled, _ = self.client.scroll(
            collection_name=self.cfg.word_collection,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=1000,
        )
        tokens = [p.payload or {} for p in scrolled]
        tokens.sort(key=lambda t: int(t.get("token_index", 0)))
        return tokens

    def pronounce(
        self,
        term: str,
        cutoff: int = 85,
        candidate_k: int = 60,
        max_return: int = 5,
    ) -> List[Dict]:
        term = (term or "").strip()
        if not term:
            return []

        # Arabic exact match
        if is_arabic(term):
            flt = Filter(must=[FieldCondition(key="arabic", match=MatchValue(value=term))])
            page, _ = self.client.scroll(
                collection_name=self.cfg.word_collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=256,
            )
            results: List[Dict] = []
            for p in page:
                payload = p.payload or {}
                ar = (payload.get("arabic") or "").strip()
                tr = (payload.get("transliteration") or "").strip()
                if ar and tr:
                    results.append({"arabic": ar, "transliteration": tr, "score": 100})
            if results:
                seen = set()
                deduped = []
                for r in results:
                    key = normalize_latin(r["transliteration"])
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(r)
                return deduped[:max_return]

        # Semantic candidates + fuzzy re-rank
        qvec = self._embed(term)
        candidates = self._qdrant_search(self.cfg.word_collection, qvec, candidate_k)

        nl_term = normalize_latin(term)
        na_term = normalize_arabic(term)

        scored: List[Dict] = []
        for c in candidates:
            payload = c.payload or {}
            ar = (payload.get("arabic") or "").strip()
            tr = (payload.get("transliteration") or "").strip()
            gloss_en = ""
            gloss = payload.get("gloss") or {}
            if isinstance(gloss, dict):
                gloss_en = (gloss.get("en") or "").strip()

            best_score = 0
            if tr:
                best_score = max(best_score, fuzz.WRatio(nl_term, normalize_latin(tr)))
            if gloss_en:
                best_score = max(best_score, fuzz.WRatio(nl_term, normalize_latin(gloss_en)))
            if ar:
                best_score = max(best_score, fuzz.WRatio(na_term, normalize_arabic(ar)))

            scored.append({"arabic": ar, "transliteration": tr, "score": int(best_score)})

        if not scored:
            return []

        scored.sort(key=lambda x: x["score"], reverse=True)

        seen = set()
        deduped: List[Dict] = []
        for r in scored:
            tr = (r.get("transliteration") or "").strip()
            if not tr:
                continue
            key = normalize_latin(tr)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)

        if not deduped:
            return []

        if deduped[0]["score"] < cutoff:
            return []

        return deduped[:max_return]

    _reranker: CrossEncoderReranker | None = None

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(device="cpu")
        return self._reranker

    def _candidate_text(self, hit: Dict, lang: str) -> str:
        tr = (hit.get("translations") or {}).get(lang) or (hit.get("translations") or {}).get("en") or ""
        ar = hit.get("arabic") or ""
        return f"{ar}\n\n{tr}".strip()

    def _deterministic_reason(self, query: str, best: Dict, lang: str) -> str:
        parts: List[str] = []
        sname = best.get("surah_name_en")
        ref = f"{best.get('surah_num')}:{best.get('ayah_num')}"
        parts.append(f"Selected the most relevant ayah by semantic search and cross-encoder reranking: {sname} ({ref}).")
        has_tafsir = bool((best.get("tafsirs") or {}).get(lang) or (best.get("tafsirs") or {}).get("en"))
        parts.append("Tafsir is available for additional context." if has_tafsir else "Tafsir was not required to select this result.")
        q_tokens = set([t for t in query.lower().split() if len(t) > 2])
        tr_text = (best.get("translations") or {}).get(lang) or (best.get("translations") or {}).get("en") or ""
        overlap = q_tokens.intersection(set([t for t in tr_text.lower().split() if len(t) > 2]))
        if overlap:
            parts.append(f"Overlapping terms in translation: {', '.join(sorted(overlap))}.")
        return " ".join(parts)

    def search_ayahs_best(
        self,
        query: str,
        lang: str,
        top_k: int = 7,
        with_alternatives: bool = True,
    ) -> Tuple[Dict | None, List[Dict], str]:
        hits = self.search_ayahs(query, top_k=top_k)
        if not hits:
            return None, [], "> No results."

        texts = [self._candidate_text(h, lang=lang) for h in hits]
        reranker = self._get_reranker()
        scores = reranker.score_pairs(query, texts)
        scored = list(zip(hits, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        best_hit = scored[0][0]
        alts = [h for (h, _) in scored[1:]] if with_alternatives else []
        reason = self._deterministic_reason(query, best_hit, lang)
        return best_hit, alts, reason


def render_hit(hit: Dict, lang: str = "en", show_tafsir: bool = False, max_tafsir_chars: int = 300) -> str:
    tr = (hit.get("translations") or {}).get(lang) or (hit.get("translations") or {}).get("en") or ""
    tafsir = (hit.get("tafsirs") or {}).get(lang) or ""
    if show_tafsir and tafsir:
        tafsir = (tafsir[:max_tafsir_chars] + "…") if len(tafsir) > max_tafsir_chars else tafsir
    header = f"{hit.get('surah_name_en')} ({hit.get('surah_num')}:{hit.get('ayah_num')}) — score={hit.get('score'):.3f}"
    body = [header, hit.get("arabic") or "", tr or ""]
    if show_tafsir and tafsir:
        body.append(f"[tafsir] {tafsir}")
    return "\n".join([line for line in body if line])
