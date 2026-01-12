"""
Habib HF Quranic Whisper (Transformers) ASR backend.

Model:
  Habib-HF/tarbiyah-ai-whisper-medium-merged

Fixes:
- Do NOT pass forced_decoder_ids as a generate() kwarg (some transformers versions reject it)
- Always provide max_new_tokens (this model's generation_config.max_length may be None)
- Provide attention_mask when available
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "Habib-HF/tarbiyah-ai-whisper-medium-merged"
TARGET_SR = 16000

# Cache
_PROCESSOR: Optional[Any] = None
_MODEL: Optional[Any] = None
_DEVICE: Optional[str] = None


def _ensure_mono_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        # (n, ch) or (ch, n) -> take first channel robustly
        if y.shape[0] < y.shape[1]:
            y = y[0, :]
        else:
            y = y[:, 0]
    return y.astype(np.float32, copy=False)


def _resample_to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return y

    # Try librosa first, else scipy
    try:
        import librosa
        return librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32, copy=False)
    except Exception:
        try:
            from scipy.signal import resample_poly
            g = math.gcd(sr, TARGET_SR)
            up = TARGET_SR // g
            down = sr // g
            return resample_poly(y, up, down).astype(np.float32, copy=False)
        except Exception as e:
            raise RuntimeError(
                f"Could not resample audio from sr={sr} to {TARGET_SR}. "
                f"Install librosa or scipy. Original error: {e}"
            )


def _load_model() -> None:
    global _PROCESSOR, _MODEL, _DEVICE
    if _PROCESSOR is not None and _MODEL is not None:
        return

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
    _MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)

    _MODEL.to(_DEVICE)
    _MODEL.eval()


def _choose_max_new_tokens(num_samples_16k: int) -> int:
    """
    Heuristic that works for both word and ayah audio without exploding runtime on CPU.
    """
    dur = num_samples_16k / float(TARGET_SR)
    # Roughly 6–10 tokens/sec; clamp to sane bounds
    return int(max(64, min(256, math.ceil(dur * 10))))


def _safe_generate(
    input_features: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    language: str,
    task: str,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Version-robust generate() wrapper.
    - Tries `language=`/`task=` (newer whisper API)
    - Falls back to forced_decoder_ids via generation_config (NOT kwarg)
    - Forces max_length if config has None to avoid TypeError
    """
    gen_kwargs = {"max_new_tokens": int(max_new_tokens)}
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    # 1) Try new-style args (works on many recent Transformers builds)
    try:
        return _MODEL.generate(input_features, language=language, task=task, **gen_kwargs)
    except Exception:
        pass

    # 2) Fallback: forced decoder ids via generation_config (NOT as kwarg)
    forced_ids = None
    try:
        forced_ids = _PROCESSOR.get_decoder_prompt_ids(language=language, task=task)
    except Exception:
        forced_ids = None

    gen_cfg = None
    if getattr(_MODEL, "generation_config", None) is not None:
        gen_cfg = copy.deepcopy(_MODEL.generation_config)

    if gen_cfg is not None and forced_ids is not None:
        gen_cfg.forced_decoder_ids = forced_ids

    # Critical fix: some models ship generation_config.max_length = None
    # Transformers validate compares int >= None -> TypeError unless we set it.
    if gen_cfg is not None and getattr(gen_cfg, "max_length", None) is None:
        gen_cfg.max_length = int(max_new_tokens + 10)

    # Also make sure pad_token_id is set (helps silence some warnings/edge cases)
    try:
        if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
            gen_cfg.pad_token_id = _PROCESSOR.tokenizer.pad_token_id
    except Exception:
        pass

    if gen_cfg is not None:
        return _MODEL.generate(input_features, generation_config=gen_cfg, **gen_kwargs)

    # 3) Last resort: set on model.config (still avoid forced_decoder_ids kwarg)
    if forced_ids is not None:
        try:
            setattr(_MODEL.config, "forced_decoder_ids", forced_ids)
        except Exception:
            pass
    if getattr(_MODEL.config, "max_length", None) is None:
        try:
            setattr(_MODEL.config, "max_length", int(max_new_tokens + 10))
        except Exception:
            pass

    return _MODEL.generate(input_features, **gen_kwargs)


def transcribe_numpy(
    y: np.ndarray,
    sr: int,
    language: str = "ar",
    task: str = "transcribe",
    max_new_tokens: Optional[int] = None,
    **_: Any,
) -> str:
    """
    Transcribe numpy audio using Habib Quranic Whisper.

    Returns:
      raw transcription string
    """
    _load_model()

    y = _ensure_mono_float32(y)
    y16 = _resample_to_16k(y, sr)

    # Ask for attention_mask when available (removes warning + more reliable)
    try:
        inputs = _PROCESSOR(
            y16,
            sampling_rate=TARGET_SR,
            return_attention_mask=True,
            return_tensors="pt",
        )
    except TypeError:
        # Older processor might not support return_attention_mask
        inputs = _PROCESSOR(
            y16,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )

    input_features = inputs.input_features.to(_DEVICE)

    attn = getattr(inputs, "attention_mask", None)
    if attn is not None:
        attn = attn.to(_DEVICE)

    if max_new_tokens is None:
        max_new_tokens = _choose_max_new_tokens(len(y16))

    with torch.no_grad():
        predicted_ids = _safe_generate(
            input_features=input_features,
            attention_mask=attn,
            language=language,
            task=task,
            max_new_tokens=max_new_tokens,
        )

    text = _PROCESSOR.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return (text or "").strip()
