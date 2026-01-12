"""
Realtime ayah-level evaluation helpers for microphone audio.

This mirrors the word pipeline style:
  mic audio -> RMS/duration checks -> resample -> (optional trim) ->
  Mel(ref wav) + Mel(user array) -> DTW -> score -> label

Ayah configuration (durations, thresholds, ref audio relpath) comes from:
  dtw_units.csv via quran_index (unit_type="ayah", word_id=0)

Visible labels:
  - "good"
  - "needs_improvement"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import resample

from . import config, features, dtw_similarity, quran_index

# ASR is optional. Only for logging/debug, never blocks DTW.
try:
    from . import asr_gate  # type: ignore[import-not-found]
    _ASR_AVAILABLE = True
except Exception:  # pragma: no cover
    asr_gate = None  # type: ignore[assignment]
    _ASR_AVAILABLE = False


# Anything below this RMS is treated as "no attempt / silence"
SILENCE_RMS_THRESHOLD: float = 0.02


@dataclass
class AyahEvalResult:
    surah: int
    ayah: int

    rms: float
    duration_sec: float

    distance: Optional[float]
    avg_cost: Optional[float]
    score: float
    label: str

    notes: str = ""


def _prepare_mic_audio(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int, float, float, str]:
    """
    Convert raw mic signal into float32 mono at config.SAMPLE_RATE.
    Returns: (y_resampled, sr_target, rms, duration_sec, notes)
    """
    y = np.asarray(audio, dtype=np.float32)

    # mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    orig_sr = int(sr)
    if y.size == 0:
        return y, config.SAMPLE_RATE, 0.0, 0.0, f"orig_sr={orig_sr}, empty_signal"

    rms_orig = float(np.sqrt(np.mean(y ** 2)))
    dur_orig = float(len(y) / orig_sr)

    # resample
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

    if y_resampled.size == 0:
        rms = 0.0
        duration_sec = 0.0
    else:
        rms = float(np.sqrt(np.mean(y_resampled ** 2)))
        duration_sec = float(len(y_resampled) / sr_target)

    notes = f"orig_sr={orig_sr}, rms_orig={rms_orig:.6f}, dur_orig={dur_orig:.3f}s"
    return y_resampled, sr_target, rms, duration_sec, notes


def _trim_silence_best_effort(y: np.ndarray) -> np.ndarray:
    """
    Best-effort trim of leading/trailing silence for mic audio.
    If librosa isn't available, returns y unchanged.
    """
    try:
        import librosa
        yt, _ = librosa.effects.trim(y.astype(np.float32), top_db=config.TRIM_TOP_DB)
        return yt.astype(np.float32, copy=False)
    except Exception:
        return y


def evaluate_ayah_from_mic(
    surah: int,
    ayah: int,
    audio: np.ndarray,
    sr: int,
    data_root: Optional[str] = None,  # kept for backward compatibility, not used
) -> AyahEvalResult:
    """
    End-to-end ayah-level evaluation from microphone audio.
    Uses unit_type="ayah" and word_id=0 in dtw_units.csv.
    """
    # 1) preprocess
    y, sr_proc, rms, duration_sec, notes_debug = _prepare_mic_audio(audio, sr)

    # silence
    if rms < SILENCE_RMS_THRESHOLD or y.size == 0:
        return AyahEvalResult(
            surah=surah,
            ayah=ayah,
            rms=rms,
            duration_sec=duration_sec,
            distance=None,
            avg_cost=None,
            score=0.0,
            label="needs_improvement",
            notes=f"very low energy / silence; {notes_debug}",
        )

    # optional mic trim
    y = _trim_silence_best_effort(y)
    duration_sec = float(len(y) / sr_proc) if sr_proc > 0 else 0.0

    # 2) cfg from CSV (ayah uses word_id=0)
    cfg = quran_index.get_unit_config("ayah", surah, ayah, 0)
    ref_path = quran_index.get_ref_audio_path(cfg)

    expected_duration_sec = cfg.expected_duration_sec or duration_sec
    min_duration_sec = cfg.min_duration_sec or (0.7 * expected_duration_sec)
    max_duration_sec = cfg.max_duration_sec or (1.5 * expected_duration_sec)

    duration_flag: Optional[str] = None
    if duration_sec < min_duration_sec:
        duration_flag = f"too_short({duration_sec:.2f}s)"
    elif duration_sec > max_duration_sec:
        duration_flag = f"too_long({duration_sec:.2f}s)"

    # 3) mel ref + mel user
    mel_ref = features.extract_mel_from_audio(
        str(ref_path),
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        trim_top_db=config.TRIM_TOP_DB,
    )
    mel_user = features.extract_mel_from_array(
        y,
        sr_proc,
        n_mels=mel_ref.shape[0],
    )

    # 4) dtw + score
    distance, avg_cost = dtw_similarity.dtw_distance(mel_ref, mel_user)
    score = dtw_similarity.score_from_distance(distance, level="ayah")

    # 5) verdict based on cfg thresholds (or fallback)
    good_th = cfg.good_threshold if cfg.good_threshold is not None else 0.90
    label = "good" if score >= good_th else "needs_improvement"

    # duration-based capping (optional): if you want to be strict
    # (keep minimal: only cap "good" if clearly wrong duration)
    if duration_flag and label == "good":
        label = "needs_improvement"

    notes = notes_debug
    if expected_duration_sec > 0:
        ratio = duration_sec / expected_duration_sec
        notes += f"; ref_dur≈{expected_duration_sec:.2f}s; ratio={ratio:.2f}"
    if duration_flag:
        notes += f"; duration_flag={duration_flag}"

    # 6) optional ASR logging (never blocks)
    if _ASR_AVAILABLE and y.size > 0:
        try:
            asr_res = asr_gate.recognise_text(y, sr_proc, language="ar")
            notes += (
                f"; asr_raw='{asr_res.raw_text}'"
                f"; asr_norm='{asr_res.normalized_text}'"
                f"; asr_sim_max={asr_res.sim_max:.2f}"
            )
        except Exception as e:
            notes += f"; asr_error={type(e).__name__}: {e}"

    return AyahEvalResult(
        surah=surah,
        ayah=ayah,
        rms=rms,
        duration_sec=duration_sec,
        distance=float(distance),
        avg_cost=float(avg_cost),
        score=float(score),
        label=label,
        notes=notes,
    )
