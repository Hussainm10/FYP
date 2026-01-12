# api/app.py
#
# FastAPI wrapper around your existing tajweed_agent_project evaluation logic.
# - Stateless
# - Exposes only final verdict (good / needs_improvement)
# - Serves local reference audio files (no TTS generation)
#
# Run:
#   pip install fastapi uvicorn python-multipart
#   uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

from __future__ import annotations

import io
from enum import Enum
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.io import wavfile

# Your existing pipeline modules
from tajweed_agent import quran_index, config, realtime_word

# Minimal patch: use the dedicated ayah evaluator (same style as word pipeline)
from tajweed_agent.realtime_ayah import evaluate_ayah_from_mic


# ----------------------------
# API models (STRICT contract)
# ----------------------------

class Mode(str, Enum):
    word_by_word = "word_by_word"
    ayah_by_ayah = "ayah_by_ayah"


class UserResult(BaseModel):
    status: Literal["good", "needs_improvement"]


class EvalResponse(BaseModel):
    mode: Literal["word_by_word", "ayah_by_ayah"]
    content_id: str
    arabic_text: str
    transliteration: str
    translation: str
    reference_audio_url: str
    user_result: UserResult
    feedback_message: str


# ----------------------------
# App + static audio serving
# ----------------------------

app = FastAPI(title="Tajweed Agent API", version="1.0")

# quran_index ref paths are resolved as: config.DATA_ROOT / cfg.ref_audio_relpath
# In your project, config.DATA_ROOT should point to: data/QuranAudio
# And ref_audio_relpath typically begins with: wav/...
AUDIO_ROOT = Path(config.DATA_ROOT) / "wav"

if not AUDIO_ROOT.exists():
    # Don’t silently run with broken paths
    raise RuntimeError(f"Audio root not found: {AUDIO_ROOT}")

# Serve: /audio/<anything under data/QuranAudio/wav/>
app.mount("/audio", StaticFiles(directory=str(AUDIO_ROOT)), name="audio")


# ----------------------------
# Helpers (no ML redesign)
# ----------------------------

def _decode_wav_upload(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode a WAV upload into (mono_float32, sr).
    Only WAV is supported here (matches your pipeline).
    """
    try:
        sr, x = wavfile.read(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {type(e).__name__}: {e}")

    if x is None or len(x) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    x = np.asarray(x)

    # If stereo -> mono
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Scale ints to float
    if np.issubdtype(x.dtype, np.integer):
        max_val = float(np.iinfo(x.dtype).max)
        if max_val <= 0:
            raise HTTPException(status_code=400, detail="Invalid integer PCM format.")
        y = (x.astype(np.float32) / max_val).astype(np.float32, copy=False)
    else:
        y = x.astype(np.float32, copy=False)

    return y, int(sr)


def _make_reference_audio_url(request: Request, ref_abs_path: Path) -> str:
    """
    Convert an absolute ref audio path into a URL served by /audio.
    We return a path-style URL (frontend can prefix host as needed).
    """
    try:
        rel = ref_abs_path.resolve().relative_to(AUDIO_ROOT.resolve())
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Reference audio path is not under AUDIO_ROOT: {ref_abs_path}",
        )
    return f"/audio/{rel.as_posix()}"


def _feedback_for(status: Literal["good", "needs_improvement"]) -> str:
    if status == "good":
        return "Masha’Allah, good! You recited it well."
    return "Don’t worry, you can do it better."


def _status_from_label(label: str) -> Literal["good", "needs_improvement"]:
    # Internal labels may be: good / intermediate / wrong / needs_improvement
    return "good" if (label or "").strip().lower() == "good" else "needs_improvement"


# ----------------------------
# Main endpoint
# ----------------------------

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(
    request: Request,
    mode: Mode = Form(...),
    surah_number: int = Form(...),
    ayah_number: int = Form(...),
    word_number: Optional[int] = Form(None),
    audio_file: UploadFile = File(...),
):
    # Validate mode + ids
    if mode == Mode.word_by_word and (word_number is None):
        raise HTTPException(status_code=400, detail="word_number is required for mode=word_by_word")

    if audio_file is None:
        raise HTTPException(status_code=400, detail="audio_file is required")

    # Decode WAV
    file_bytes = await audio_file.read()
    y, sr = _decode_wav_upload(file_bytes)

    # Load unit config + reference path + texts
    if mode == Mode.word_by_word:
        cfg = quran_index.get_unit_config(
            unit_type="word",
            surah_id=int(surah_number),
            ayah_id=int(ayah_number),
            word_id=int(word_number),
        )

        internal_result = realtime_word.evaluate_word_from_mic(
            int(surah_number),
            int(ayah_number),
            int(word_number),
            y,
            sr,
        )
        internal_label = internal_result.label
        content_id = f"{surah_number}:{ayah_number}:{word_number}"

    else:
        # Minimal patch: use realtime_ayah (CSV ayah rows are word_id=0)
        cfg = quran_index.get_unit_config(
            unit_type="ayah",
            surah_id=int(surah_number),
            ayah_id=int(ayah_number),
            word_id=0,
        )

        ayah_res = evaluate_ayah_from_mic(
            int(surah_number),
            int(ayah_number),
            y,
            sr,
        )
        # ayah_res.label is: "good" or "needs_improvement"
        internal_label = ayah_res.label
        content_id = f"{surah_number}:{ayah_number}:all"

    ref_path = quran_index.get_ref_audio_path(cfg)
    reference_audio_url = _make_reference_audio_url(request, ref_path)

    # Build response (no diagnostics exposed)
    status = _status_from_label(internal_label)

    arabic_text = (cfg.text_ar or "").strip()
    transliteration = (cfg.text_translit or "").strip()
    translation = ""  # Not present in your CSV/UnitConfig; do not infer.

    return EvalResponse(
        mode=mode.value,
        content_id=content_id,
        arabic_text=arabic_text,
        transliteration=transliteration,
        translation=translation,
        reference_audio_url=reference_audio_url,
        user_result=UserResult(status=status),
        feedback_message=_feedback_for(status),
    )
