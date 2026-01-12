"""
Command-line entry point for evaluating a recitation against a reference.

Usage examples
--------------

Word-level (Surah 114, Ayah 1, Word 1):

    python -m tajweed_agent.main \
        --surah 114 --ayah 1 --word 1 \
        --user_audio data/QuranAudio/wav/words_wav/sura_114/ayah_001/114_001_001.wav \
        --data_root data/QuranAudio

Ayah-level (Surah 114, Ayah 1):

    python -m tajweed_agent.main \
        --surah 114 --ayah 1 \
        --user_audio data/QuranAudio/wav/ayahs_wav/sura_114/114_001.wav \
        --data_root data/QuranAudio

Surah-level (Surah 114):

    python -m tajweed_agent.main \
        --surah 114 \
        --user_audio data/QuranAudio/wav/surahs_wav/sura_114/114.wav \
        --data_root data/QuranAudio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, data_loader, features, dtw_similarity


def _format_progress_bar(score: float, width: int = 30) -> str:
    filled_length = int(round(width * score))
    bar = "#" * filled_length + "-" * (width - filled_length)
    return f"[{bar}]"


def run(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a recitation against a reference using DTW"
    )
    parser.add_argument(
        "--surah", type=int, required=True, help="Surah number (1-114)"
    )
    parser.add_argument(
        "--ayah",
        type=int,
        default=None,
        help="Ayah number (1-based); omit for surah-level",
    )
    parser.add_argument(
        "--word",
        type=int,
        default=None,
        help="Word number (1-based); omit for ayah/surah level",
    )
    parser.add_argument(
        "--user_audio",
        type=str,
        required=True,
        help="Path to the user's audio recording (WAV)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override the QuranAudio dataset root directory",
    )
    parsed = parser.parse_args(args)

    # Decide comparison level based on which indices are provided
    if parsed.word is not None:
        level = "word"
    elif parsed.ayah is not None:
        level = "ayah"
    else:
        level = "surah"

    # Resolve data root (if provided)
    data_root: Path | None = None
    if parsed.data_root:
        data_root = Path(parsed.data_root)
        if not data_root.exists():
            print(f"Error: data_root '{data_root}' does not exist", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    # Reference Mel (from WAV)
    # ------------------------------------------------------------------
    try:
        mel_ref = data_loader.get_reference_mel(
            level,
            parsed.surah,
            ayah=parsed.ayah,
            word=parsed.word,
            data_root=data_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading reference: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # User Mel
    # ------------------------------------------------------------------
    print(f"Loading user audio from {parsed.user_audio} ...")
    mel_user = features.extract_mel_from_audio(
        parsed.user_audio,
        n_mels=mel_ref.shape[0],
    )
    print(f"User Mel shape    : {mel_user.shape}")
    print(f"Reference Mel shape: {mel_ref.shape}")

    # ------------------------------------------------------------------
    # DTW + scoring
    # ------------------------------------------------------------------
    print("Computing DTW similarity ...")
    distance, avg_cost = dtw_similarity.dtw_distance(mel_ref, mel_user)
    score = dtw_similarity.score_from_distance(distance, level=level)
    label = dtw_similarity.label_from_score(score, level=level)

    progress_bar = _format_progress_bar(score)
    percent = score * 100.0
    print("\n=== Recitation Evaluation ===")
    print(f"DTW distance    : {distance:.4f}")
    print(f"Average cost    : {avg_cost:.4f}")
    print(f"Similarity score: {percent:.2f}%")
    print(f"Verdict         : {config.LABELS[label]}")
    print(f"Progress bar    : {progress_bar}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
