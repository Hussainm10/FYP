"""
Realtime word-level evaluation helpers for microphone audio.

Pipeline:

    mic audio (np.ndarray, sr_in)
      → basic checks (RMS, duration)
      → resample to config.SAMPLE_RATE
      → optional cropping to loudest word-sized window
      → Mel extraction from reference WAV + user array
      → DTW vs reference word Mel
      → distance, score, label

Word configuration (expected durations, thresholds, paths, etc.) comes
from the CSV-driven quran_index module (dtw_units.csv).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import resample

from . import config, features, dtw_similarity, quran_index

# ASR is optional. We only use it for logging, never to block DTW.
try:
    from . import asr_gate  # type: ignore[import-not-found]

    _ASR_AVAILABLE = True
except Exception:  # pragma: no cover
    asr_gate = None  # type: ignore[assignment]
    _ASR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration constants for mic evaluation
# ---------------------------------------------------------------------------

# Anything below this RMS is treated as "no attempt / silence"
SILENCE_RMS_THRESHOLD: float = 0.02

# Minimum window size (in seconds) when cropping to the loudest region
MIN_CROP_WINDOW_SEC: float = 0.35

# Factor for choosing crop window relative to expected duration:
#   crop_window ≈ min(duration, max(MIN_CROP_WINDOW_SEC, expected * CROP_DUR_FACTOR))
CROP_DUR_FACTOR: float = 2.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class WordEvalResult:
    surah: int
    ayah: int
    word: int

    rms: float
    duration_sec: float

    distance: Optional[float]
    avg_cost: Optional[float]
    score: float
    label: str

    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_mic_audio(
    audio: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, int, float, float, str]:
    """
    Convert raw mic signal into float32 mono at config.SAMPLE_RATE.

    Returns:
        y_resampled : np.ndarray
            Audio signal resampled to config.SAMPLE_RATE.
        sr_target   : int
            Target sample rate (config.SAMPLE_RATE).
        rms         : float
            RMS of the *resampled* audio.
        duration_sec: float
            Duration (seconds) at sr_target.
        notes       : str
            Debug string with original sr, rms, and duration.
    """
    y = np.asarray(audio, dtype=np.float32)

    # If multi-channel, convert to mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    orig_sr = int(sr)
    if y.size == 0:
        return y, config.SAMPLE_RATE, 0.0, 0.0, f"orig_sr={orig_sr}, empty_signal"

    # Original stats (for notes)
    rms_orig = float(np.sqrt(np.mean(y**2)))
    dur_orig = float(len(y) / orig_sr)

    # Resample if needed
    if orig_sr != config.SAMPLE_RATE:
        target_len = int(round(len(y) * config.SAMPLE_RATE / orig_sr))
        if target_len <= 0:
            return np.zeros(0, dtype=np.float32), config.SAMPLE_RATE, 0.0, 0.0, (
                f"orig_sr={orig_sr}, target_len<=0"
            )
        y_resampled = resample(y, target_len).astype(np.float32)
        sr_target = config.SAMPLE_RATE
    else:
        y_resampled = y
        sr_target = orig_sr

    # Compute RMS + duration on resampled signal
    if y_resampled.size == 0:
        rms = 0.0
        duration_sec = 0.0
    else:
        rms = float(np.sqrt(np.mean(y_resampled**2)))
        duration_sec = float(len(y_resampled) / sr_target)

    notes = f"orig_sr={orig_sr}, rms_orig={rms_orig:.6f}, dur_orig={dur_orig:.3f}s"
    return y_resampled, sr_target, rms, duration_sec, notes


def _crop_to_loudest_window(
    y: np.ndarray,
    sr: int,
    window_sec: float,
) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    """
    Crop audio to the loudest window of size window_sec.

    Returns:
        segment : np.ndarray
        start   : int | None
        end     : int | None
    """
    if y.size == 0 or window_sec <= 0.0:
        return y, None, None

    win_samples = int(round(window_sec * sr))
    if win_samples <= 0 or win_samples >= len(y):
        # Window longer than signal: return full signal
        return y, 0, len(y)

    # Energy via moving sum of squared amplitude
    y2 = y.astype(np.float32) ** 2
    window = np.ones(win_samples, dtype=np.float32)
    energy = np.convolve(y2, window, mode="valid")

    best_idx = int(np.argmax(energy))
    start = best_idx
    end = start + win_samples
    segment = y[start:end]

    return segment, start, end


def _get_unit_cfg_word(surah: int, ayah: int, word: int):
    """
    quran_index.get_unit_config() signature has changed a few times across your edits.
    This wrapper tries the common variants without changing quran_index itself.
    """
    # Variant A: keyword style
    try:
        return quran_index.get_unit_config(
            unit_type="word",
            surah_id=surah,
            ayah_id=ayah,
            word_id=word,
        )
    except TypeError:
        pass

    # Variant B: positional style
    return quran_index.get_unit_config("word", surah, ayah, word)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_word_from_mic(
    surah: int,
    ayah: int,
    word: int,
    audio: np.ndarray,
    sr: int,
    data_root: Optional[str] = None,  # kept for backward compatibility, not used
) -> WordEvalResult:
    """
    End-to-end word-level evaluation from microphone audio.
    """
    # 1) Basic preprocessing (mono + resample + RMS/duration)
    y, sr_proc, rms, duration_sec, notes_debug = _prepare_mic_audio(audio, sr)

    # Silence / no attempt check
    if rms < SILENCE_RMS_THRESHOLD or y.size == 0:
        return WordEvalResult(
            surah=surah,
            ayah=ayah,
            word=word,
            rms=rms,
            duration_sec=duration_sec,
            distance=None,
            avg_cost=None,
            score=0.0,
            label="wrong",
            notes=f"very low energy / silence; {notes_debug}",
        )

    # 2) Load unit configuration from CSV (dtw_units.csv)
    cfg = _get_unit_cfg_word(surah, ayah, word)

    # Reference path resolution (your quran_index already owns the truth here)
    try:
        ref_path = quran_index.get_ref_audio_path(cfg)
    except Exception:
        # Fallback if you ever rename helper: use cfg.ref_audio_relpath if present
        ref_rel = getattr(cfg, "ref_audio_relpath", None)
        if ref_rel is None:
            raise
        ref_path = ref_rel  # may already be absolute in your project

    # Duration expectations from CSV
    expected_duration_sec = float(getattr(cfg, "expected_duration_sec", 0.0) or 0.0)
    if expected_duration_sec <= 0:
        expected_duration_sec = duration_sec

    min_duration_sec = float(getattr(cfg, "min_duration_sec", 0.0) or 0.0)
    if min_duration_sec <= 0:
        min_duration_sec = 0.5 * expected_duration_sec

    max_duration_sec = float(getattr(cfg, "max_duration_sec", 0.0) or 0.0)
    if max_duration_sec <= 0:
        max_duration_sec = 3.0 * expected_duration_sec

    duration_flag: Optional[str] = None
    if duration_sec < min_duration_sec:
        duration_flag = f"too_short({duration_sec:.2f}s)"
    elif duration_sec > max_duration_sec:
        duration_flag = f"too_long({duration_sec:.2f}s)"

    # 3) Crop to loudest region (word-sized)
    crop_window_sec = min(
        duration_sec,
        max(MIN_CROP_WINDOW_SEC, expected_duration_sec * CROP_DUR_FACTOR),
    )

    y_segment, seg_start, seg_end = _crop_to_loudest_window(
        y,
        sr_proc,
        window_sec=crop_window_sec,
    )
    eff_dur = float(len(y_segment) / sr_proc) if sr_proc > 0 else 0.0

    # 4) Compute reference Mel from reference WAV (CSV-driven)
    # IMPORTANT: match your features.py signature (sample_rate=..., not sr=...)
    mel_ref = features.extract_mel_from_audio(
        str(ref_path),
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        trim_top_db=config.TRIM_TOP_DB,
    )

    # 5) Compute user Mel from the cropped in-memory audio
    mel_user = features.extract_mel_from_array(
        y_segment,
        sr_proc,
        n_mels=mel_ref.shape[0],
    )

    # 6) DTW distance and score (compute ONCE)
    distance, avg_cost = dtw_similarity.dtw_distance(mel_ref, mel_user)
    score = dtw_similarity.score_from_distance(distance, level="word")

    # 7) Two-class label selection (collapse intermediate -> wrong)
    # Global defaults (can be overridden per-word via CSV if present) #chnage append(moosa) #threshold set up 
    GLOBAL_GOOD_THRESHOLD = 0.75
    GLOBAL_INTER_THRESHOLD = 0.60
    
    good_th = float(getattr(cfg, "good_threshold", None) or GLOBAL_GOOD_THRESHOLD)
    inter_th = float(getattr(cfg, "intermediate_threshold", None) or GLOBAL_INTER_THRESHOLD)
    
    tier = "below_intermediate"
    if score >= good_th:
        tier = "good"
    elif score >= inter_th:
        tier = "intermediate"

    # visible label: only good / wrong
    label = "good" if tier == "good" else "wrong"

    # Duration-based capping: if duration is outside expected bounds, do NOT allow good
    if duration_flag and label == "good":
        label = "wrong"
        tier = "duration_capped"

    # 8) Build notes/debug string
    notes = notes_debug
    if expected_duration_sec > 0:
        ratio = duration_sec / expected_duration_sec
        notes += (
            f"; ref_dur≈{expected_duration_sec:.2f}s"
            f"; eff_dur≈{eff_dur:.2f}s"
            f"; ratio={ratio:.2f}"
        )
    if duration_flag:
        notes += f"; duration_flag={duration_flag}"
    if seg_start is not None:
        notes += f"; cropped=[{seg_start}:{seg_end}]"

    # Keep tier for tuning/debugging without exposing a 3rd class
    notes += f"; tier={tier}"

    # 9) Optional ASR logging (non-blocking, best-effort only)
    if _ASR_AVAILABLE and y_segment.size > 0:
        try:
            asr_res = asr_gate.recognise_text(
                y_segment,
                sr_proc,
                language="ar",
            )

            # Be robust to field naming variations across your edits
            raw = getattr(asr_res, "raw_text", None)
            if raw is None:
                raw = getattr(asr_res, "raw", "")

            norm = getattr(asr_res, "normalized_text", None)
            if norm is None:
                norm = getattr(asr_res, "asr_norm", None)
            if norm is None:
                norm = ""

            sim = getattr(asr_res, "sim_max", None)
            if sim is None:
                sim = getattr(asr_res, "similarity", None)
            if sim is None:
                sim = 0.0

            notes += f"; asr_raw='{raw}'; asr_norm='{norm}'; asr_sim_max={float(sim):.2f}"
        except Exception as e:  # pragma: no cover
            notes += f"; asr_error={type(e).__name__}: {e}"

    return WordEvalResult(
        surah=surah,
        ayah=ayah,
        word=word,
        rms=rms,
        duration_sec=duration_sec,
        distance=float(distance) if distance is not None else None,
        avg_cost=float(avg_cost) if avg_cost is not None else None,
        score=float(score),
        label=label,
        notes=notes,
    )
