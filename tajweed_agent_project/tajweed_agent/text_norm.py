"""
Text normalisation and similarity utilities for ASR gating.

This module provides:
  - Basic Arabic normalisation (remove diacritics, unify characters)
  - Basic Latin normalisation (lowercase, strip)
  - A simple character-level similarity score in [0, 1]
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


# -----------------------------
# Arabic normalisation
# -----------------------------

_ARABIC_DIACRITICS = re.compile(
    "[" +
    "\u0610-\u061A" +  # Quranic annotations
    "\u064B-\u065F" +  # tashkeel
    "\u0670" +
    "\u06D6-\u06ED" +
    "]"
)


def normalize_arabic(text: str) -> str:
    """
    Normalise Arabic for robust string comparison:

      - Remove diacritics (harakat, Quranic marks)
      - Normalise common character variants (e.g. ى→ي, ة→ه, etc.)
      - Remove non-letter characters except spaces
      - Collapse multiple spaces
      - Strip leading/trailing spaces
    """
    if not text:
        return ""

    # Unicode normalisation
    t = unicodedata.normalize("NFC", text)

    # Remove diacritics
    t = _ARABIC_DIACRITICS.sub("", t)

    # Common character normalisations
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي")
    t = t.replace("ة", "ه")
    t = t.replace("ؤ", "و").replace("ئ", "ي")

    # Keep only letters and spaces
    cleaned_chars = []
    for ch in t:
        if ch.isspace():
            cleaned_chars.append(" ")
        elif "\u0600" <= ch <= "\u06FF":  # Arabic block
            cleaned_chars.append(ch)
        # else: drop non-Arabic symbols
    t = "".join(cleaned_chars)

    # Collapse spaces and strip
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -----------------------------
# Latin normalisation
# -----------------------------

def normalize_latin(text: str) -> str:
    """
    Normalise Latin-script text for comparison:

      - Unicode NFC
      - Lowercase
      - Remove non alphanumeric characters except spaces
      - Collapse spaces
    """
    if not text:
        return ""

    t = unicodedata.normalize("NFC", text)
    t = t.lower()

    cleaned_chars = []
    for ch in t:
        if ch.isalnum():
            cleaned_chars.append(ch)
        elif ch.isspace():
            cleaned_chars.append(" ")
        # else: drop punctuation, etc.
    t = "".join(cleaned_chars)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -----------------------------
# Similarity
# -----------------------------

def char_similarity(a: str, b: str) -> float:
    """
    Character-level similarity in [0, 1] using SequenceMatcher ratio.
    Empty strings return 0.0.
    """
    a = a or ""
    b = b or ""
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())
