"""
Pydantic v2-safe models for ayah/word JSON rows with robust helpers.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field, ConfigDict


class SurahInfo(BaseModel):
    """Basic surah metadata bundled with each ayah."""
    model_config = ConfigDict(extra="ignore")

    name_ar: str
    name_en: str
    total_ayahs: int
    revealed_in: str
    meaning_en: Optional[str] = None


class AyahRecord(BaseModel):
    """
    Represents one ayah entry from ayahs_collection.json.
    Only core fields are modeled; translations/tafsirs are dicts.
    """
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="e.g., '001:001'")
    surah_num: int
    ayah_num: int
    surah_info: SurahInfo
    arabic: str
    transliteration: Optional[str] = None
    segmented_words: Optional[str] = None
    translations: Dict[str, str] = Field(default_factory=dict)
    tafsirs: Dict[str, str] = Field(default_factory=dict)

    def embedding_text(self) -> str:
        """
        Build text for embedding:
        Arabic + transliteration + English translation (if present).
        Keeping it focused avoids topic drift in retrieval.
        """
        dump = self.model_dump()
        parts = [
            dump.get("arabic", "") or "",
            dump.get("transliteration", "") or "",
        ]
        en = (dump.get("translations") or {}).get("en") or (dump.get("translations") or {}).get("EN") or ""
        if en:
            parts.append(en)
        return " \n".join([p for p in parts if p])


class WordRecord(BaseModel):
    """
    Represents one token entry from words_collection.json.
    Useful for word-by-word view, lemma/root searches, etc.
    """
    model_config = ConfigDict(extra="ignore")

    word_id: str = Field(..., description="e.g., '1:1:1' (surah:ayah:token)")
    surah: int
    ayah: int
    arabic: str
    transliteration: Optional[str] = None
    Segmented_Word: Optional[str] = None  # keep source casing
    english: Optional[str] = None
    urdu: Optional[str] = None
    farsi: Optional[str] = None
    pashto: Optional[str] = None
    root: Optional[str] = None
    lemma: Optional[str] = None
    pos: Optional[str] = None

    def token_index(self) -> int:
        """
        Extract 1-based token index from word_id formatted as 'surah:ayah:token'.
        Falls back to 0 if parsing fails.
        """
        try:
            return int(self.word_id.split(":")[2])
        except Exception:
            return 0

    def norm_root(self) -> Optional[str]:
        """Lowercased root for filtering (does not modify original)."""
        raw = (self.root or "").strip().lower()
        return raw or None

    def norm_lemma(self) -> Optional[str]:
        """Lowercased, brace/tilde/backtick-stripped lemma for filtering."""
        raw = (self.lemma or "").lower().strip()
        for ch in "{}~`":
            raw = raw.replace(ch, "")
        return raw or None

    def embedding_text(self) -> str:
        """
        Build text for word embeddings:
        Arabic + transliteration + short glosses (EN/UR/FA/PS).
        This supports cross-lingual word lookups.
        """
        dump = self.model_dump()
        parts = [
            dump.get("arabic", "") or "",
            dump.get("transliteration", "") or "",
            dump.get("english", "") or "",
            dump.get("urdu", "") or "",
            dump.get("farsi", "") or "",
            dump.get("pashto", "") or "",
        ]
        return " \n".join([p for p in parts if p])
