"""
ASR gate: runs ASR and computes similarity vs expected text.

This module MUST be stable because other files call it.

Design goals:
- Provide ASRResult with consistent fields.
- Backward-compatible aliases for older attribute names:
    norm_ar, norm_tr, sim_max, normalized_text, raw_text, etc.
- recognise_text() should not crash if extra kwargs are passed.
- Default backend = Habib HF Quranic Whisper (Transformers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import numpy as np

from tajweed_agent import text_norm

# Our new HF backend
from tajweed_agent.habib_whisper_asr import transcribe_numpy


@dataclass
class ASRResult:
    # Canonical fields
    raw_text: str
    normalized_text: str

    # Similarity vs expected (0..1)
    sim_ar: float = 0.0
    sim_tr: float = 0.0
    sim_max_value: float = 0.0

    # ---- Backward-compatible aliases ----
    @property
    def norm_ar(self) -> str:
        # older code expects Arabic normalized string
        return self.normalized_text

    @property
    def norm_tr(self) -> str:
        # if you later add transliteration normalization, return it here
        return ""

    @property
    def sim_max(self) -> float:
        return self.sim_max_value


def recognise_text(
    audio: np.ndarray,
    sr: int,
    language: str = "ar",
    **kwargs: Any,
) -> ASRResult:
    """
    Run ASR on the provided audio and return ASRResult.

    Note:
    - Accepts **kwargs to remain compatible with older calls
      (e.g., model_name=..., backend=..., etc.). They are ignored safely.
    """
    raw = transcribe_numpy(audio, sr, language=language, task="transcribe")
    norm = text_norm.normalize_arabic(raw)
    return ASRResult(raw_text=raw, normalized_text=norm)


def content_similarity(
    asr_res: ASRResult,
    expected_ar: Optional[str] = None,
    expected_tr: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Compute similarity of ASR output vs expected Arabic and transliteration.
    Returns: (sim_ar, sim_tr, sim_max)
    """
    sim_ar = 0.0
    sim_tr = 0.0

    if expected_ar:
        exp_ar_norm = text_norm.normalize_arabic(expected_ar)
        sim_ar = text_norm.char_similarity(asr_res.norm_ar, exp_ar_norm)

    # Transliteration similarity (optional)
    if expected_tr:
        exp_tr_norm = text_norm.normalize_latin(expected_tr)
        # You can implement Arabic->Latin transliteration later.
        sim_tr = 0.0

    sim_max = max(sim_ar, sim_tr)
    return sim_ar, sim_tr, sim_max


def recognise_and_score(
    audio: np.ndarray,
    sr: int,
    expected_ar: Optional[str],
    expected_tr: Optional[str] = None,
    language: str = "ar",
    **kwargs: Any,
) -> ASRResult:
    """
    Convenience wrapper: run ASR then compute similarity fields.
    """
    res = recognise_text(audio, sr, language=language, **kwargs)
    sim_ar, sim_tr, sim_max = content_similarity(res, expected_ar, expected_tr)

    res.sim_ar = float(sim_ar)
    res.sim_tr = float(sim_tr)
    res.sim_max_value = float(sim_max)
    return res
