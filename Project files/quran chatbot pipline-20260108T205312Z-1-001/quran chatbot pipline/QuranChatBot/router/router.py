"""
Regex-only intent detection + execution via Catalog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from qdrant_client import QdrantClient
from .catalog import Catalog, fetch_ayah_range

SURAH_WORDS = r"(?:surah|sura|سور[ۃه])"
AYAH_WORDS  = r"(?:ayah|ayat|verse|آیت|آیات)"

# pronounce patterns
PRONOUNCE_RE = re.compile(r"(?:pronounce|pronunciation|how\s+do\s+i\s+pronounce)\s+([A-Za-zء-ي]+)", re.I | re.UNICODE)


@dataclass(frozen=True)
class RouteResult:
    intent: str
    results: Optional[List[Dict]] = None
    info: Optional[Dict] = None


class DataRouter:
    RE_SURAH_NAME = re.compile(rf"{SURAH_WORDS}\s+([A-Za-zء-ي\- ]+)", re.I | re.UNICODE)
    RE_RANGE      = re.compile(rf"{AYAH_WORDS}\s+(\d+)\s*(?:to|-|–|—)\s*(\d+)", re.I | re.UNICODE)
    RE_SINGLE     = re.compile(rf"{AYAH_WORDS}\s+(\d+)\b", re.I | re.UNICODE)
    RE_FIRSTN     = re.compile(rf"first\s+(\d+)\s+{AYAH_WORDS}", re.I | re.UNICODE)

    def __init__(self, ayahs_json_path: str, qdrant_host: str = "localhost", qdrant_port: int = 6333) -> None:
        self.catalog = Catalog.from_ayahs_json(ayahs_json_path)
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def route(self, text: str, ayah_collection: str) -> RouteResult:
        t = text.strip()

        # 0) Pronunciation intent
        pm = PRONOUNCE_RE.search(t)
        if pm:
            term = pm.group(1)
            return RouteResult(intent="PRONUNCIATION", info={"term": term})

        # 1) Surah name present?
        name_m = self.RE_SURAH_NAME.search(t)
        if not name_m:
            return RouteResult(intent="SEMANTIC_FALLBACK")

        raw_name = name_m.group(1)
        surah_hit = self.catalog.fuzzy_surah(raw_name)
        if not surah_hit:
            return RouteResult(intent="SEMANTIC_FALLBACK")
        surah_num, _matched_key, _score = surah_hit

        # 2) Ranges / first-N / single
        r_m = self.RE_RANGE.search(t)
        if r_m:
            s, e = int(r_m.group(1)), int(r_m.group(2))
            s, e = self.catalog.clamp_range(surah_num, s, e)
            rows = fetch_ayah_range(self.client, ayah_collection, surah_num, s, e)
            return RouteResult(intent="SURAH_RANGE", results=rows, info={"surah_num": surah_num, "start": s, "end": e})

        f_m = self.RE_FIRSTN.search(t)
        if f_m:
            n = int(f_m.group(1))
            lo, hi = self.catalog.ayah_bounds.get(surah_num, (1, n))
            s, e = lo, min(lo + n - 1, hi)
            rows = fetch_ayah_range(self.client, ayah_collection, surah_num, s, e)
            return RouteResult(intent="SURAH_RANGE", results=rows, info={"surah_num": surah_num, "start": s, "end": e})

        s_m = self.RE_SINGLE.search(t)
        if s_m:
            ix = int(s_m.group(1))
            s, e = self.catalog.clamp_range(surah_num, ix, ix)
            rows = fetch_ayah_range(self.client, ayah_collection, surah_num, s, e)
            return RouteResult(intent="SINGLE_AYAH", results=rows, info={"surah_num": surah_num, "ayah": s})

        # 3) Surah only → first 10 ayahs
        lo, hi = self.catalog.ayah_bounds.get(surah_num, (1, 7))
        s, e = lo, min(hi, lo + 10 - 1)
        rows = fetch_ayah_range(self.client, ayah_collection, surah_num, s, e)
        return RouteResult(intent="SURAH_FULL", results=rows, info={"surah_num": surah_num, "start": s, "end": e})
