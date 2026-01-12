"""
quran_index.py

Index loader for DTW units (words / ayahs) defined in data/config/dtw_units.csv.

This module:
- Loads dtw_units.csv once at import time
- Builds a lookup dict keyed by (unit_type, surah_id, ayah_id, word_id)
- Exposes helper functions to:
    - get_unit_config(...)  → metadata for one unit
    - get_ref_audio_path(...) → absolute Path to reference WAV
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

# ---------------------------------------------------------------------------
# Config import (works both as package and when run directly)
# ---------------------------------------------------------------------------
try:
    # Normal case: imported as part of tajweed_agent package
    from . import config  # type: ignore[import]
except ImportError:
    # Fallback: if someone runs this file directly: python quran_index.py
    import config  # type: ignore[import]

# Root path to dtw_units.csv
# __file__ = .../tajweed_agent/quran_index.py
# parent      = .../tajweed_agent
# parent.parent = project root
CONFIG_DIR = Path(__file__).resolve().parent.parent / "data" / "config"
UNITS_CSV_PATH = CONFIG_DIR / "dtw_units.csv"

# Key type for the index dict
Key = Tuple[str, int, int, int]  # (unit_type, surah_id, ayah_id, word_id)


# ---------------------------------------------------------------------------
# Dataclass describing one unit row from the CSV
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitConfig:
    unit_id: str
    unit_type: str
    surah_id: int
    ayah_id: int
    word_id: int

    text_ar: str
    text_clean: str
    text_translit: str

    ref_audio_relpath: str

    expected_duration_sec: Optional[float]
    min_duration_sec: Optional[float]
    max_duration_sec: Optional[float]

    good_threshold: Optional[float]
    intermediate_threshold: Optional[float]
    max_good_duration_factor: Optional[float]

    difficulty: Optional[int]
    active: bool
    notes: str = ""


# In-memory index
_INDEX: Dict[Key, UnitConfig] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "t"):
        return True
    if s in ("0", "false", "no", "n", "f"):
        return False
    return default


def _load_index() -> None:
    """Load dtw_units.csv into the global _INDEX dict (only once)."""
    global _INDEX
    if _INDEX:
        return  # already loaded

    if not UNITS_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"dtw_units.csv not found at {UNITS_CSV_PATH}. "
            "Check that the file exists and the path is correct."
        )

    with UNITS_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit_type = row.get("unit_type", "").strip().lower()
            surah_id = _parse_int(row.get("surah_id"))
            ayah_id = _parse_int(row.get("ayah_id"))
            word_id = _parse_int(row.get("word_id"))

            if not unit_type or surah_id is None or ayah_id is None or word_id is None:
                # Skip incomplete rows
                continue

            key: Key = (unit_type, surah_id, ayah_id, word_id)

            cfg = UnitConfig(
                unit_id=row.get("unit_id", "").strip(),
                unit_type=unit_type,
                surah_id=surah_id,
                ayah_id=ayah_id,
                word_id=word_id,
                text_ar=row.get("text_ar", "").strip(),
                text_clean=row.get("text_clean", "").strip(),
                text_translit=row.get("text_translit", "").strip(),
                ref_audio_relpath=row.get("ref_audio_relpath", "").strip(),
                expected_duration_sec=_parse_float(row.get("expected_duration_sec")),
                min_duration_sec=_parse_float(row.get("min_duration_sec")),
                max_duration_sec=_parse_float(row.get("max_duration_sec")),
                good_threshold=_parse_float(row.get("good_threshold")),
                intermediate_threshold=_parse_float(row.get("intermediate_threshold")),
                max_good_duration_factor=_parse_float(row.get("max_good_duration_factor")),
                difficulty=_parse_int(row.get("difficulty")),
                active=_parse_bool(row.get("active"), default=True),
                notes=row.get("notes", "").strip(),
            )

            _INDEX[key] = cfg


# Load on import so it's ready to use
_load_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_unit_config(
    unit_type: str,
    surah_id: int,
    ayah_id: int,
    word_id: int,
    *,
    must_be_active: bool = True,
) -> UnitConfig:
    """
    Look up configuration for a given (unit_type, surah, ayah, word).

    unit_type: "word" or "ayah"
    word_id:   for ayah-level rows, this is usually 0.
    """
    unit_type = unit_type.strip().lower()
    key: Key = (unit_type, surah_id, ayah_id, word_id)

    if not _INDEX:
        _load_index()

    try:
        cfg = _INDEX[key]
    except KeyError:
        raise KeyError(
            f"No DTW unit found for key={key} in {UNITS_CSV_PATH}"
        )

    if must_be_active and not cfg.active:
        raise KeyError(
            f"DTW unit {key} is marked inactive in {UNITS_CSV_PATH}"
        )

    return cfg


def list_units(unit_type: str | None = None) -> Iterable[UnitConfig]:
    """
    Return all UnitConfig entries, optionally filtered by unit_type.
    """
    if not _INDEX:
        _load_index()

    if unit_type is None:
        return list(_INDEX.values())

    utype = unit_type.strip().lower()
    return [cfg for cfg in _INDEX.values() if cfg.unit_type == utype]


def get_ref_audio_path(cfg: UnitConfig) -> Path:
    """
    Get the absolute Path to the reference audio file for this unit,
    by joining config.DATA_ROOT with ref_audio_relpath.
    """
    return config.DATA_ROOT / cfg.ref_audio_relpath
