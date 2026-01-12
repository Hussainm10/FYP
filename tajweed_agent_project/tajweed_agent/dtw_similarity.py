"""
Dynamic Time Warping (DTW) based similarity metrics.

This module implements functions to compute the DTW cost between two
Mel spectrograms and convert that cost into a normalised score.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from . import config

# Try to use librosa's DTW if available (faster / battle-tested).
try:
    import librosa  # type: ignore[import-not-found]

    _LIBROSA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    librosa = None  # type: ignore[assignment]
    _LIBROSA_AVAILABLE = False


# ---------------------------------------------------------------------
# Core DTW
# ---------------------------------------------------------------------


def dtw_distance(
    mel_ref: np.ndarray,
    mel_user: np.ndarray,
    *,
    metric: str = "cosine",
) -> Tuple[float, float]:
    """
    Compute Dynamic Time Warping distance and average cost.

    Parameters
    ----------
    mel_ref, mel_user : np.ndarray [shape=(n_mels, T)]
        Reference and user Mel spectrograms.
    metric : {"cosine", "euclidean"}
        Local frame-to-frame distance metric.

    Returns
    -------
    distance : float
        Total DTW cost along the best path.
    avg_cost : float
        Average cost per step along that path.
    """
    X = np.asarray(mel_ref, dtype=np.float32)
    Y = np.asarray(mel_user, dtype=np.float32)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("mel_ref and mel_user must be 2-D arrays (n_mels, T)")

    if _LIBROSA_AVAILABLE:
        # librosa expects features as (n_features, T)
        D, wp = librosa.sequence.dtw(X=X, Y=Y, metric=metric)
        distance = float(D[-1, -1])
        avg_cost = distance / len(wp)
        return distance, avg_cost

    # ---------------- Fallback: manual DTW -----------------
    T_ref = X.shape[1]
    T_user = Y.shape[1]

    # Pairwise frame distances
    if metric == "cosine":
        eps = 1e-10
        X_norm = X / (np.linalg.norm(X, axis=0, keepdims=True) + eps)
        Y_norm = Y / (np.linalg.norm(Y, axis=0, keepdims=True) + eps)
        dot_prod = np.clip(X_norm.T @ Y_norm, -1.0, 1.0)  # (T_ref, T_user)
        pairwise = 1.0 - dot_prod
    elif metric == "euclidean":
        X_exp = X.T[:, None, :]  # (T_ref, 1, n_mels)
        Y_exp = Y.T[None, :, :]  # (1, T_user, n_mels)
        pairwise = np.linalg.norm(X_exp - Y_exp, axis=2)  # (T_ref, T_user)
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'euclidean'.")

    # DP table
    D = np.full((T_ref + 1, T_user + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    path_len = np.zeros((T_ref + 1, T_user + 1), dtype=np.int32)

    for i in range(1, T_ref + 1):
        for j in range(1, T_user + 1):
            cost = pairwise[i - 1, j - 1]
            idx = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if idx == 0:  # up
                D[i, j] = cost + D[i - 1, j]
                path_len[i, j] = path_len[i - 1, j] + 1
            elif idx == 1:  # left
                D[i, j] = cost + D[i, j - 1]
                path_len[i, j] = path_len[i, j - 1] + 1
            else:  # diag
                D[i, j] = cost + D[i - 1, j - 1]
                path_len[i, j] = path_len[i - 1, j - 1] + 1

    distance = float(D[T_ref, T_user])
    length = int(path_len[T_ref, T_user])
    avg_cost = distance / length if length > 0 else 0.0
    return distance, avg_cost


# ---------------------------------------------------------------------
# Level-aware scoring
# ---------------------------------------------------------------------


def score_from_distance(distance: float, level: str = "word") -> float:
    """
    Convert a DTW distance into a similarity score in [0, 1].

    Different `level`s use different scales, since distances grow
    with sequence length.

    Parameters
    ----------
    distance : float
        DTW path cost.
    level : {"word", "ayah", "surah", "generic"}
        Evaluation granularity.

    Returns
    -------
    float
        Similarity score in [0, 1], where 1.0 is a perfect match.
    """
    if level == "word":
        scale = 20.0
    elif level == "ayah":
        scale = 75.0
    elif level == "surah":
        scale = 150.0
    else:
        scale = 25.0

    norm = distance / max(scale, 1e-8)
    score = math.exp(-norm)
    return max(0.0, min(1.0, score))


def label_from_score(score: float, level: str = "generic") -> str:
    """
    Map a numeric score into a qualitative category.

    The mapping can depend on the evaluation level:

    - Word-level: we require a *very* high score for "good" so that
      wrong words do not easily pass as correct.
    - Ayah/Surah: similar idea but a bit more permissive.
    - Generic/legacy: fall back to thresholds from ``config`` so the
      CLI and other callers behave as before.
    """
    default_wrong = config.WRONG_THRESHOLD          # e.g. 0.4
    default_inter = config.INTERMEDIATE_THRESHOLD   # e.g. 0.75

    if level == "word":
        # Word-level:
        #   score >= 0.90  → good
        #   0.60–0.90      → intermediate
        #   < 0.60         → wrong
        wrong_th = 0.60
        inter_th = 0.90
    elif level in ("ayah", "surah"):
        wrong_th = 0.40
        inter_th = 0.90
    else:
        wrong_th = default_wrong
        inter_th = default_inter

    if score >= inter_th:
        return "good"
    if score >= wrong_th:
        return "intermediate"
    return "wrong"
