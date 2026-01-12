"""
Configuration constants for the Tajweed Agent.

These values define the expected audio processing parameters and
the default dataset location.  They can be overridden via
environment variables or changed programmatically by importing
this module and modifying the attributes.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# DATASET ROOT
# ---------------------------------------------------------------------

# Root directory for the QuranAudio dataset.
# If the environment variable ``TAJWEED_DATA_ROOT`` is set, it overrides this.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "QuranAudio"
DATA_ROOT = Path(os.environ.get("TAJWEED_DATA_ROOT", DEFAULT_DATA_ROOT))

# ---------------------------------------------------------------------
# AUDIO PROCESSING PARAMETERS
# ---------------------------------------------------------------------

# Target sample rate for all audio loaded by the pipeline.
SAMPLE_RATE: int = 22_050

# STFT window size.
N_FFT: int = 1024

# Hop length for STFT frames.
HOP_LENGTH: int = 256

# Mel bands — None means infer from reference mel file.
N_MELS: int | None = None

# Silence trimming threshold.
TRIM_TOP_DB: int = 30

# ---------------------------------------------------------------------
# SCORING & LABEL THRESHOLDS (word-level defaults)
# ---------------------------------------------------------------------

"""
IMPORTANT:
Scores returned by `dtw_similarity.score_from_distance` are in [0, 1].

Meaning:
- Lower distance → score closer to 1 (Good)
- Higher distance → score decays toward 0 (Wrong)
"""

# Word-level thresholds (used when level="word").
WRONG_THRESHOLD: float = 0.40
INTERMEDIATE_THRESHOLD: float = 0.70

LABELS = {
    "wrong": "Wrong",
    "intermediate": "Intermediate",
    "good": "Good",
}

# UI colors
COLORS = {
    "wrong": "#dc2626",         # red
    "intermediate": "#f97316",  # orange
    "good": "#16a34a",          # green
}

# ---------------------------------------------------------------------
# FEATURE SOURCE / CACHE FLAGS
# ---------------------------------------------------------------------

# If True, even word-level will use live Mel extraction from reference WAV.
# If False, word-level will try to use cached mel .npy (if available).
USE_LIVE_WORD_MEL: bool = True

# Enable an in-memory cache for reference Mels (word/ayah/surah) to avoid
# recomputing features for the same unit during a session.
ENABLE_MEL_CACHE: bool = True
