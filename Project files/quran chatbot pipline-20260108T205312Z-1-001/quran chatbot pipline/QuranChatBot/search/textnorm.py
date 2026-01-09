"""
Lightweight text normalization utilities for pronunciation matching.
No external deps beyond stdlib.
"""

from __future__ import annotations
import re

# Arabic diacritics (tashkeel) and tatweel
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u0640]")
# Anything that's not a letter or digit
_NON_ALNUM = re.compile(r"[^0-9A-Za-z\u0600-\u06FF]+")


def is_arabic(text: str) -> bool:
    """Return True if the string contains Arabic script codepoints."""
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def normalize_arabic(text: str) -> str:
    """
    Strip Arabic diacritics/tatweel and collapse whitespace/punct.
    Useful for diacritic-insensitive equality or fuzzy checks.
    """
    if not text:
        return ""
    s = _ARABIC_DIACRITICS.sub("", text)
    s = _NON_ALNUM.sub(" ", s)
    return " ".join(s.split()).strip()


def normalize_latin(text: str) -> str:
    """
    Lowercase, remove punctuation/hyphens/apostrophes/spaces.
    'al-lahi' -> 'allahi', "bismillah" -> 'bismillah'
    """
    if not text:
        return ""
    s = text.lower()
    s = re.sub(r"[’'`\-_]", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s.strip()
