# """
# Lightweight Ollama client for Qur'an commentary.
#
# - Talks to a local Ollama server (Docker) on http://localhost:11434
# - Uses a single model (e.g. llama3.1) to generate short, grounded explanations
# - Does NOT invent Qur'an text, only explains the ayah and its translation
# """
#
# from __future__ import annotations
#
# from typing import Dict, Optional
#
# import json
# import logging
# import os
#
# import requests
#
# LOGGER = logging.getLogger(__name__)
#
#
# def _build_prompt(
#         user_query: str,
#         ayah: Dict,
#         lang: str,
#         tafsir_excerpt: Optional[str] = None,
# ) -> str:
#     """
#     Construct a strict prompt for the LLM:
#     - Includes the user question
#     - Includes ONE ayah in Arabic + translation
#     - Optionally includes tafsir snippet
#     - Asks for a short, safe explanation (no invented verses)
#     """
#     surah_name_en = ayah.get("surah_name_en", "Unknown Surah")
#     surah_num = ayah.get("surah_num")
#     ayah_num = ayah.get("ayah_num")
#     arabic = ayah.get("arabic", "").strip()
#
#     translations = ayah.get("translations") or {}
#     tr = translations.get(lang) or translations.get("en") or ""
#
#     parts = []
#
#     # System-style instructions (Ollama doesn't have roles, so we embed it in text)
#     parts.append(
#         "You are a Qur'an explanation assistant.\n"
#         "You will be given:\n"
#         "- The user's question\n"
#         "- One ayah from the Qur'an (Arabic + translation)\n"
#         "- Optionally a tafsir snippet\n\n"
#         "Your task:\n"
#         "- Explain how this ayah relates to the user's question\n"
#         "- Answer in clear, simple language\n"
#         "- Use 3-7 short bullet points\n"
#         "- Do NOT invent new Qur'an verses\n"
#         "- Do NOT quote hadith or external sources\n"
#         "- Base yourself only on the given ayah and tafsir\n"
#         "- Stay neutral and informational\n"
#     )
#
#     parts.append(f"User question:\n{user_query.strip()}\n")
#
#     parts.append(
#         f"Ayah reference: Surah {surah_name_en} ({surah_num}:{ayah_num})\n\n"
#         f"Ayah (Arabic):\n{arabic}\n"
#     )
#
#     parts.append(f"Translation ({lang} or English fallback):\n{tr}\n")
#
#     if tafsir_excerpt:
#         parts.append(f"Tafsir snippet:\n{tafsir_excerpt}\n")
#
#     parts.append(
#         "Now write a short explanation as bullet points, in markdown, under the heading 'Commentary'.\n"
#         "Example format:\n"
#         "Commentary:\n"
#         "- point 1\n"
#         "- point 2\n"
#         "- point 3\n"
#     )
#
#     return "\n".join(parts)
#
#
# def generate_quran_commentary(
#         user_query: str,
#         ayah: Dict,
#         lang: str,
#         tafsir_excerpt: Optional[str] = None,
# ) -> str:
#     """
#     Call Ollama to generate a short commentary for the selected ayah.
#
#     Args:
#         user_query: Original user question.
#         ayah: Dict containing at least:
#               - surah_name_en, surah_num, ayah_num
#               - arabic
#               - translations: {lang_code: text}
#         lang: Target translation language code ('en', 'ur', etc.).
#         tafsir_excerpt: Optional short tafsir text.
#
#     Returns:
#         Markdown string with a 'Commentary:' section. If Ollama fails,
#         returns a fallback string.
#     """
#     base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#     model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
#
#     prompt = _build_prompt(user_query=user_query, ayah=ayah, lang=lang, tafsir_excerpt=tafsir_excerpt)
#
#     payload = {
#         "model": model_name,
#         "prompt": prompt,
#         "stream": False,
#     }
#
#     try:
#         resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
#         resp.raise_for_status()
#         data = resp.json()
#         text = data.get("response", "").strip()
#         if not text:
#             return "Commentary:\n- LLM did not return any explanation."
#         return text
#     except Exception as exc:
#         LOGGER.exception("Error calling Ollama LLM: %s", exc)
#         return (
#             "Commentary:\n"
#             "- LLM commentary is temporarily unavailable.\n"
#             "- The ayah and translation above are still valid; please review them directly."
#         )


# """
# Thin Ollama client for generating Qur'an commentary.
#
# Runs locally against Ollama HTTP API:
#     POST /api/generate
#
# Model name is configurable but defaults to 'llama3.1:latest'.
# """
#
# from __future__ import annotations
#
# import os
# import json
# import requests
# from typing import Dict
#
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
#
#
# def _build_prompt(question: str, ayah: Dict, tafsir: str) -> str:
#     """
#     Build a grounded prompt. The model must NOT make up ayahs;
#     it should only explain the given one.
#     """
#     surah_en = ayah.get("surah_name_en")
#     surah_num = ayah.get("surah_num")
#     ayah_num = ayah.get("ayah_num")
#     arabic = ayah.get("arabic") or ""
#     translations = ayah.get("translations") or {}
#     tr_en = translations.get("en") or ""
#
#     tafsir_block = tafsir.strip() if tafsir else "No tafsir text provided."
#
#     prompt = f"""
# You are a careful Qur'an teacher. You are given:
# - A user question.
# - One specific Qur'anic ayah (with Arabic and English translation).
# - Optional tafsir text from a trusted dataset.
#
# Your job:
# 1. Briefly connect the user's question to this ayah.
# 2. Provide a clear, gentle explanation in simple modern English.
# 3. Stay strictly grounded in the given ayah and tafsir.
# 4. Do NOT invent new verses, numbers, or rulings not supported by the text.
# 5. Keep the tone pastoral and practical for everyday Muslims.
#
# User question:
# {question}
#
# Ayah reference: {surah_en} ({surah_num}:{ayah_num})
#
# Arabic:
# {arabic}
#
# English translation:
# {tr_en}
#
# Tafsir (dataset):
# {tafsir_block}
#
# Write a short commentary (3–7 bullet points or short paragraphs) explaining how this ayah helps answer the question.
# Avoid Arabic transliteration unless it really helps understanding.
# Do not mention that you are an AI or talk about the system prompt.
# """
#     return prompt.strip()
#
#
# def generate_quran_commentary(
#         question: str,
#         ayah: Dict,
#         tafsir: str = "",
#         model: str | None = None,
#         timeout: int = 120,
# ) -> str:
#     """
#     Call Ollama to generate a short, grounded commentary.
#
#     Returns empty string on failure so the UI can still render.
#     """
#     model_name = model or OLLAMA_MODEL
#     prompt = _build_prompt(question=question, ayah=ayah, tafsir=tafsir)
#
#     url = f"{OLLAMA_BASE_URL}/api/generate"
#     payload = {
#         "model": model_name,
#         "prompt": prompt,
#         "stream": False,
#     }
#
#     try:
#         resp = requests.post(url, json=payload, timeout=timeout)
#     except Exception:
#         # Ollama not reachable or some network issue
#         return ""
#
#     if resp.status_code != 200:
#         return ""
#
#     try:
#         data = resp.json()
#     except json.JSONDecodeError:
#         return ""
#
#     return (data.get("response") or "").strip()


"""
Thin Ollama client for generating Qur'an commentary and pronunciation help.

Runs locally against Ollama HTTP API:
    POST /api/generate

Model name is configurable but defaults to 'llama3.1:latest'.
"""

from __future__ import annotations

import os
import json
from typing import Dict

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")


# --------------------------------------------------------------------
# Qur'an commentary prompt
# --------------------------------------------------------------------


def _build_commentary_prompt(question: str, ayah: Dict, tafsir: str) -> str:
    """
    Build a grounded prompt. The model must NOT make up ayahs;
    it should only explain the given one.
    """
    surah_en = ayah.get("surah_name_en")
    surah_num = ayah.get("surah_num")
    ayah_num = ayah.get("ayah_num")
    arabic = ayah.get("arabic") or ""
    translations = ayah.get("translations") or {}
    tr_en = translations.get("en") or ""

    tafsir_block = tafsir.strip() if tafsir else "No tafsir text provided."

    prompt = f"""
You are a careful Qur'an teacher. You are given:
- A user question.
- One specific Qur'anic ayah (with Arabic and English translation).
- Optional tafsir text from a trusted dataset.

Your job:
1. Briefly connect the user's question to this ayah.
2. Provide a clear, gentle explanation in simple modern English.
3. Stay strictly grounded in the given ayah and tafsir.
4. Do NOT invent new verses, numbers, or rulings not supported by the text.
5. Keep the tone pastoral and practical for everyday Muslims.

User question:
{question}

Ayah reference: {surah_en} ({surah_num}:{ayah_num})

Arabic:
{arabic}

English translation:
{tr_en}

Tafsir (dataset):
{tafsir_block}

Write a short commentary (3–7 bullet points or short paragraphs) explaining how this ayah helps answer the question.
Avoid Arabic transliteration unless it really helps understanding.
Do not mention that you are an AI or talk about the system prompt.
"""
    return prompt.strip()


def generate_quran_commentary(
        question: str,
        ayah: Dict,
        tafsir: str = "",
        model: str | None = None,
        timeout: int = 120,
) -> str:
    """
    Call Ollama to generate a short, grounded commentary.

    Returns empty string on failure so the UI can still render.
    """
    model_name = model or OLLAMA_MODEL
    prompt = _build_commentary_prompt(question=question, ayah=ayah, tafsir=tafsir)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except Exception:
        # Ollama not reachable or some network issue
        return ""

    if resp.status_code != 200:
        return ""

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return ""

    return (data.get("response") or "").strip()


# --------------------------------------------------------------------
# Pronunciation helper prompt
# --------------------------------------------------------------------


def _build_pronunciation_prompt(term: str) -> str:
    """
    Prompt for explaining how to pronounce a single term (e.g. bismillah).
    """
    prompt = f"""
You are an Arabic teacher explaining pronunciation to absolute beginners.

Task:
- Explain clearly how to pronounce the word or phrase "{term}".
- Use simple English and approximate Latin spelling.

Requirements:
- Give an approximate Latin spelling (for example: "biss-mil-lah").
- Break it into syllables and describe how each part sounds.
- Keep the answer under 120 words.
- Bullet points are fine.
- Focus ONLY on how it sounds, not the full fiqh or meaning.
- If the input already looks like Arabic script, explain how an Arabic speaker would say it.

Do not mention that you are an AI or talk about prompts.
"""
    return prompt.strip()


def generate_pronunciation_guide(
        term: str,
        model: str | None = None,
        timeout: int = 120,
) -> str:
    """
    Call Ollama to explain how to pronounce an Arabic / Islamic term.

    Used when the lexicon-based matcher cannot confidently answer.
    Returns a short markdown explanation.
    """
    model_name = model or OLLAMA_MODEL
    prompt = _build_pronunciation_prompt(term=term)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except Exception:
        return ""

    if resp.status_code != 200:
        return ""

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return ""

    return (data.get("response") or "").strip()
