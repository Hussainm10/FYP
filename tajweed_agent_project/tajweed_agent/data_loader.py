"""
Utilities for locating and loading reference Mel spectrograms and WAVs.

The dataset is expected to have this structure under DATA_ROOT:

- wav/words_wav/sura_{surah}/ayah_{ayah:03d}/{surah:03d}_{ayah:03d}_{word:03d}.wav
- wav/ayahs_wav/sura_{surah}/{surah:03d}_{ayah:03d}.wav
- wav/surahs_wav/sura_{surah}/{surah:03d}.wav   (adjust if your layout differs)

Cached Mel files (older .mel.npy) are *optional* and used only if
USE_LIVE_WORD_MEL=False for word-level. Ayah/surah-level now always use
live Mel extraction from WAV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import config, mel_cache


# ---------------------------------------------------------------------
# WAV PATH HELPERS
# ---------------------------------------------------------------------

def get_reference_wav_path(
    surah: int,
    ayah: Optional[int] = None,
    word: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> Path:
    """
    Construct the path to a reference WAV recording.

    Word-level:
        wav/words_wav/sura_{surah}/ayah_{ayah:03d}/{surah:03d}_{ayah:03d}_{word:03d}.wav

    Ayah-level:
        wav/ayahs_wav/sura_{surah}/{surah:03d}_{ayah:03d}.wav

    Surah-level (flexible):
        Prefer:
            wav/surahs_wav/sura_{surah}/{surah:03d}.wav
        Fallback:
            wav/surahs_wav/{surah:03d}.wav
    """
    root = Path(data_root or config.DATA_ROOT)

    # ----------------- WORD -----------------
    if word is not None:
        if ayah is None:
            raise ValueError("word specified but ayah not provided")
        wav_dir = root / "wav" / "words_wav" / f"sura_{surah}" / f"ayah_{ayah:03d}"
        fname = f"{surah:03d}_{ayah:03d}_{word:03d}.wav"
        return wav_dir / fname

    # ----------------- AYAH -----------------
    if ayah is not None:
        wav_dir = root / "wav" / "ayahs_wav" / f"sura_{surah}"
        fname = f"{surah:03d}_{ayah:03d}.wav"
        return wav_dir / fname

    # ----------------- SURAH -----------------
    # Try folder-based layout first: wav/surahs_wav/sura_{surah}/{surah:03d}.wav
    folder_dir = root / "wav" / "surahs_wav" / f"sura_{surah}"
    folder_file = folder_dir / f"{surah:03d}.wav"

    # Then try flat layout: wav/surahs_wav/{surah:03d}.wav
    flat_file = root / "wav" / "surahs_wav" / f"{surah:03d}.wav"

    if folder_file.exists():
        return folder_file
    if flat_file.exists():
        return flat_file

    # If neither exists, raise a clear error so callers can report it.
    raise FileNotFoundError(
        f"No Surah WAV found for surah {surah}. Checked:\n"
        f"  {folder_file}\n"
        f"  {flat_file}"
    )


# ---------------------------------------------------------------------
# OPTIONAL: CACHED MEL (WORD-LEVEL ONLY)
# ---------------------------------------------------------------------

def _get_cached_mel_path(
    surah: int,
    ayah: Optional[int] = None,
    word: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> Path:
    """
    Construct the path to a cached Mel spectrogram file.

    Word-level:
        cache/mel/word/S{surah}_A{ayah:03d}_W{word:03d}.mel.npy

    (Ayah/Surah-level cached Mels are not used anymore.)
    """
    root = Path(data_root or config.DATA_ROOT)

    if word is not None:
        if ayah is None:
            raise ValueError("word specified but ayah not provided")
        fname = f"S{surah}_A{ayah:03d}_W{word:03d}.mel.npy"
        mel_dir = root / "cache" / "mel" / "word"
    else:
        raise ValueError("Cached Mels are only supported for word-level in this design.")

    return mel_dir / fname


def _load_cached_mel(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Cached MEL not found at {path}")
    mel = np.load(path)
    if mel.ndim == 3:
        mel = mel.squeeze(axis=0)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel, got {mel.shape}")
    return mel


# ---------------------------------------------------------------------
# UNIFIED REFERENCE MEL ACCESS
# ---------------------------------------------------------------------

def get_reference_mel(
    level: str,
    surah: int,
    ayah: Optional[int] = None,
    word: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> np.ndarray:
    """
    Return the reference Mel spectrogram for the given unit.

    Logic:
    - For level == "word":
        - If USE_LIVE_WORD_MEL=True:
            compute Mel from the reference word WAV (via mel_cache).
        - Else:
            load cached Mel from cache/mel/word/.
    - For level == "ayah" or "surah":
        - Always compute Mel from the reference WAV (via mel_cache).
    """
    # Word-level
    if level == "word":
        if not config.USE_LIVE_WORD_MEL:
            path = _get_cached_mel_path(surah, ayah, word, data_root=data_root)
            return _load_cached_mel(path)

        wav_path = get_reference_wav_path(surah, ayah, word, data_root=data_root)
        return mel_cache.get_or_compute_ref_mel(
            "word", surah, ayah, word, str(wav_path)
        )

    # Ayah-level
    if level == "ayah":
        if ayah is None:
            raise ValueError("Ayah-level comparison requires an ayah index.")
        wav_path = get_reference_wav_path(surah, ayah, None, data_root=data_root)
        return mel_cache.get_or_compute_ref_mel(
            "ayah", surah, ayah, None, str(wav_path)
        )

    # Surah-level
    if level == "surah":
        wav_path = get_reference_wav_path(surah, None, None, data_root=data_root)
        return mel_cache.get_or_compute_ref_mel(
            "surah", surah, None, None, str(wav_path)
        )

    raise ValueError(f"Unknown level '{level}'. Expected 'word', 'ayah', or 'surah'.")
