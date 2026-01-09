"""
Data-driven Qur'an catalog (no hardcoding).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rapidfuzz import fuzz, process
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

_ARTICLE = re.compile(r"^(al[\-\s]+)", re.I)


def _normalize_name(s: str) -> str:
    """
    Lowercase, strip punctuation, fold hyphens, remove leading 'al ', trim variants like final 'h'.
    Examples:
      'Al-Baqarah' -> 'baqara'
      'Ikhlas' -> 'ikhlas'
    """
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"[-_]", " ", s)
    s = _ARTICLE.sub("", s)  # drop leading 'al-'
    s = re.sub(r"\s+", " ", s).strip()
    # collapse common trailing variants: baqara(h) -> baqara
    s = re.sub(r"(ah|a)$", "a", s)
    return s


@dataclass(frozen=True)
class Catalog:
    surah_by_en: Dict[str, int]
    surah_by_ar: Dict[str, int]
    name_variants: List[str]
    ayah_bounds: Dict[int, Tuple[int, int]]

    @classmethod
    def from_ayahs_json(cls, ayahs_json_path: str) -> "Catalog":
        data = json.loads(Path(ayahs_json_path).read_text(encoding="utf-8"))
        surah_by_en: Dict[str, int] = {}
        surah_by_ar: Dict[str, int] = {}
        bounds: Dict[int, Tuple[int, int]] = {}

        for row in data:
            sn = int(row["surah_num"])
            en_raw = row["surah_info"]["name_en"]
            ar_raw = row["surah_info"]["name_ar"]

            en = _normalize_name(en_raw)
            ar = _normalize_name(ar_raw)

            # populate dictionaries
            surah_by_en.setdefault(en, sn)
            surah_by_ar.setdefault(ar, sn)

            # bounds
            ay = int(row["ayah_num"])
            lo, hi = bounds.get(sn, (ay, ay))
            bounds[sn] = (min(lo, ay), max(hi, ay))

            # add extra EN alias without spaces for robustness (e.g., 'baqarah'/'baqara')
            surah_by_en.setdefault(en.replace(" ", ""), sn)

        name_variants = list(set([*surah_by_en.keys(), *surah_by_ar.keys()]))
        return cls(surah_by_en, surah_by_ar, name_variants, bounds)

    def fuzzy_surah(self, text: str, score_cutoff: int = 70) -> Optional[Tuple[int, str, int]]:
        """
        Fuzzy map user text to a surah number.
        Lower cutoff to 70 so typos like 'bagarah' still work.
        """
        q = _normalize_name(text)
        # try direct key first (O(1))
        if q in self.surah_by_en:
            return self.surah_by_en[q], q, 100
        if q in self.surah_by_ar:
            return self.surah_by_ar[q], q, 100
        # fuzzy
        best = process.extractOne(q, self.name_variants, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
        if not best:
            return None
        key, score, _ = best
        if key in self.surah_by_en:
            return self.surah_by_en[key], key, int(score)
        if key in self.surah_by_ar:
            return self.surah_by_ar[key], key, int(score)
        return None

    def clamp_range(self, surah_num: int, start: int, end: int) -> Tuple[int, int]:
        lo, hi = self.ayah_bounds.get(surah_num, (start, end))
        return max(start, lo), min(end, hi)


def fetch_ayah_range(client: QdrantClient, collection: str, surah_num: int, start: int, end: int) -> List[dict]:
    flt = Filter(
        must=[
            FieldCondition(key="surah_num", match=MatchValue(value=surah_num)),
            FieldCondition(key="ayah_num", range=Range(gte=start, lte=end)),
        ]
    )
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        with_payload=True,
        with_vectors=False,
        limit=1000,
    )
    rows = [p.payload for p in points]
    rows.sort(key=lambda r: int(r["ayah_num"]))
    return rows
