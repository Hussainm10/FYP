from __future__ import annotations

from typing import Any, Dict, Optional
import os

from openai import OpenAI

from core import settings


def _lang_name(lang: str) -> str:
    m = {
        "en": "English",
        "ur": "Urdu",
        "fa": "Persian (Farsi)",
        "ps": "Pashto",
        "ar": "Arabic",
    }
    return m.get((lang or "en").strip().lower(), "English")


def _safe_str(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else ""


def _build_commentary_prompt(
    *,
    question: str,
    ayah_ref: str,
    arabic: str,
    translation: str,
    tafsir: str,
    lang: str,
) -> str:
    # Force the model to respond in the chosen language
    out_lang = _lang_name(lang)

    return f"""
You are a Quran-based guidance assistant.

OUTPUT LANGUAGE (must follow strictly): {out_lang}

Safety + scope rules:
- Use ONLY the provided ayah Arabic text, the provided translation, and the provided tafsir snippet.
- Do NOT introduce other verses, hadith, or external sources.
- Do NOT issue rulings or fatwas.
- Be calm, compassionate, and practical for emotional support.
- If the user's question suggests self-harm or imminent danger, advise immediate local help and trusted people.

User question:
{question}

Referenced ayah:
{ayah_ref}

Arabic:
{arabic}

Translation:
{translation}

Tafsir snippet:
{tafsir if tafsir else "No tafsir provided."}

Task:
1) Explain (briefly) what the ayah is saying (grounded in translation/tafsir).
2) Connect it to the user's question and offer consolation and guidance.
3) Provide 3 short actionable reflections (bullets).
4) End with a gentle note: "For religious rulings, consult a qualified scholar."
""".strip()


def generate_quran_commentary(
    *,
    question: str,
    ayah: Optional[Dict[str, Any]] = None,
    ayah_ref: Optional[str] = None,
    arabic: Optional[str] = None,
    translation: Optional[str] = None,
    tafsir: Optional[str] = None,
    lang: str = "en",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "response": <string>,
        "model": <string or None>,
        "key_themes": [..],
        "cautions": [..]
      }

    NOTE: Accepts either:
      - ayah=<dict>  (your current FastAPI/Gradio path)
      OR
      - explicit ayah_ref/arabic/translation
    """

    # Resolve OpenAI key
    api_key = getattr(settings, "openai_api_key", None) or os.getenv("OPENAI_API_KEY", "")
    api_key = (api_key or "").strip()
    if not api_key:
        return {
            "response": "No additional commentary generated.",
            "model": None,
            "key_themes": [],
            "cautions": ["OPENAI_API_KEY not set."],
        }

    client = OpenAI(api_key=api_key)

    # Resolve fields
    if ayah is not None:
        ayah_ref = ayah_ref or f"{ayah.get('surah_num')}:{ayah.get('ayah_num')}"
        arabic = arabic or ayah.get("arabic") or ""
        translations = ayah.get("translations") or {}
        translation = translation or translations.get(lang) or translations.get("en") or ""
        tafsirs = ayah.get("tafsirs") or {}
        tafsir = tafsir or tafsirs.get(lang) or tafsirs.get("en") or ""
    else:
        ayah_ref = ayah_ref or ""
        arabic = arabic or ""
        translation = translation or ""
        tafsir = tafsir or ""

    prompt = _build_commentary_prompt(
        question=_safe_str(question),
        ayah_ref=_safe_str(ayah_ref),
        arabic=_safe_str(arabic),
        translation=_safe_str(translation),
        tafsir=_safe_str(tafsir),
        lang=lang,
    )

    use_model = model or "gpt-4o-mini"

    try:
        # Use Chat Completions for maximum compatibility with installed SDK versions
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": "You follow instructions precisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=450,
        )

        text = (resp.choices[0].message.content or "").strip()

        return {
            "response": text if text else "No additional commentary generated.",
            "model": use_model,
            "key_themes": [],
            "cautions": ["This is an informational response. For rulings consult a qualified scholar."],
        }

    except Exception as e:
        print(f"OpenAI commentary error: {e}")
        return {
            "response": "No additional commentary generated.",
            "model": None,
            "key_themes": [],
            "cautions": ["OpenAI request failed."],
        }


def generate_pronunciation_guide(term: str, lang: str = "en") -> str:
    # Keep as safe fallback (optional)
    return ""
