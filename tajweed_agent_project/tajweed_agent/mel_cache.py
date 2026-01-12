"""
Simple in-memory cache for reference Mel spectrograms.

This is purely an optimisation layer. The ground truth for reference
features is always computed from the reference WAV via
`features.extract_mel_from_audio`. Cached Mels can be safely discarded
whenever the feature pipeline changes.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np

from . import config, features


# Internal cache: maps (level, surah, ayah, word) -> mel_ref (np.ndarray)
_CACHE: Dict[Tuple[str, int, Optional[int], Optional[int]], np.ndarray] = {}


def _make_key(level: str, surah: int,
              ayah: Optional[int], word: Optional[int]) -> Tuple[str, int, Optional[int], Optional[int]]:
    return (level, int(surah), ayah if ayah is None else int(ayah),
            word if word is None else int(word))


def get_or_compute_ref_mel(
    level: str,
    surah: int,
    ayah: Optional[int],
    word: Optional[int],
    wav_path: str,
    *,
    n_mels: Optional[int] = None,
) -> np.ndarray:
    """
    Return a reference Mel spectrogram, computing it from WAV if needed.

    Parameters
    ----------
    level : {"word", "ayah", "surah"}
        Granularity of the comparison.
    surah, ayah, word : int or None
        Indices identifying the unit.
    wav_path : str
        Path to the reference WAV file.
    n_mels : int, optional
        If provided, override the default number of mel bands.

    Returns
    -------
    mel : np.ndarray
        Mel spectrogram of shape (n_mels, T).
    """
    key = _make_key(level, surah, ayah, word)

    if config.ENABLE_MEL_CACHE and key in _CACHE:
        return _CACHE[key]

    mel = features.extract_mel_from_audio(wav_path, n_mels=n_mels)

    if config.ENABLE_MEL_CACHE:
        _CACHE[key] = mel

    return mel
