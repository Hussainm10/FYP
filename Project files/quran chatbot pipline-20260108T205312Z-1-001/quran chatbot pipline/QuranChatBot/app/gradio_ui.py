"""
Gradio UI for Qur'an semantic + structured search (OpenAI embeddings + Qdrant).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import gradio as gr

from core import settings
from router.router import DataRouter
from search.retrieval import QuranSearcher, SearchConfig

# IMPORTANT: direct import (no relative imports)
from llm.openai_client import (
    generate_quran_commentary,
    generate_pronunciation_guide,
)

# -------------------------------------------------------------------
# Global router + paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AYAH_JSON = PROJECT_ROOT / "metadata" / "ayahs_collection.json"

ROUTER = DataRouter(ayahs_json_path=str(AYAH_JSON))
DEFAULT_TOP_K = getattr(settings, "default_top_k", 7)

# -------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------


def _format_pronunciation(term: str, matches: List[Dict]) -> str:
    if not matches:
        return f"> Could not find a pronunciation for **{term}**."

    lines = [
        f"## Pronunciation guide for **{term}**",
        "",
        "Here are Qur'anic forms and their transliterations:",
        "",
    ]
    for m in matches:
        lines.append(
            f"- **{m.get('arabic','')}** → `{m.get('transliteration','')}` "
            f"(match ≈ {m.get('score',0)}%)"
        )

    return "\n".join(lines)


def _format_quran_answer(
    question: str,
    lang: str,
    best: Dict,
    alts: List[Dict],
    reason: str,
    show_tafsir: bool,
    commentary: str,
) -> str:
    surah_en = best.get("surah_name_en")
    surah_num = best.get("surah_num")
    ayah_num = best.get("ayah_num")

    arabic = best.get("arabic") or ""
    translations = best.get("translations") or {}
    tafsirs = best.get("tafsirs") or {}

    translation = translations.get(lang) or translations.get("en") or ""
    tafsir = tafsirs.get(lang) or tafsirs.get("en") or ""

    lines = []
    lines.append("## The Qur'an says\n")
    lines.append(f"**{surah_en} ({surah_num}:{ayah_num})**\n")

    lines.append("**Arabic:**\n")
    lines.append(arabic or "_Arabic text not available._")
    lines.append("")

    lines.append(f"**Translation ({lang}):**\n")
    lines.append(translation or "_Translation not available._")
    lines.append("")

    if show_tafsir:
        lines.append("**Tafsir (dataset):**\n")
        lines.append(tafsir or "_Tafsir not available._")
        lines.append("")

    lines.append("**Commentary:**\n")
    lines.append(commentary.strip() if commentary else "_No additional commentary generated._")
    lines.append("")

    lines.append("**Why this ayah was selected:**\n")
    lines.append(reason or "")
    lines.append("")

    if alts:
        lines.append("<details><summary>See alternative ayahs considered</summary>\n")
        for h in alts[:5]:
            lines.append(
                f"- **{h.get('surah_name_en')} "
                f"({h.get('surah_num')}:{h.get('ayah_num')})**"
            )
        lines.append("\n</details>")

    return "\n".join(lines)


# -------------------------------------------------------------------
# Core callback
# -------------------------------------------------------------------


def on_search(query: str, lang: str, show_tafsir: bool) -> str:
    query = (query or "").strip()
    if not query:
        return "> Please enter a question."

    cfg = SearchConfig(
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
        ayah_collection=settings.qdrant_collection_ayahs,
        word_collection=settings.qdrant_collection_words,
        top_k=DEFAULT_TOP_K,
    )

    searcher = QuranSearcher(cfg)

    # Pronunciation intent
    if "pronounce" in query.lower():
        term = query.replace("pronounce", "").strip()
        matches = searcher.pronounce(term)
        if matches:
            return _format_pronunciation(term, matches)
        return generate_pronunciation_guide(term) or "> No pronunciation found."

    routed = ROUTER.route(query, ayah_collection=cfg.ayah_collection)

    if routed.intent != "SEMANTIC_FALLBACK" and routed.results:
        best = routed.results[0]
        alts = routed.results[1:]
        reason = (
            f"Matched your structured request to "
            f"{best.get('surah_name_en')} ({best.get('surah_num')}:{best.get('ayah_num')})."
        )
    else:
        best, alts, reason = searcher.search_ayahs_best(
            query=query,
            lang=lang,
            top_k=cfg.top_k,
            with_alternatives=True,
        )

    if not best:
        return "> No relevant ayah found."

    # ---------------- FIXED OPENAI CALL ----------------

    ayah_ref = f"{best.get('surah_name_en')} ({best.get('surah_num')}:{best.get('ayah_num')})"
    translation = (best.get("translations") or {}).get(lang) or ""
    tafsir = (best.get("tafsirs") or {}).get(lang) or ""

    commentary_obj = generate_quran_commentary(
        question=query,
        ayah_ref=ayah_ref,
        arabic=best.get("arabic", ""),
        translation=translation,
        tafsir=tafsir,
        lang=lang,
    )

    commentary = commentary_obj.get("commentary")

    return _format_quran_answer(
        question=query,
        lang=lang,
        best=best,
        alts=alts,
        reason=reason,
        show_tafsir=show_tafsir,
        commentary=commentary,
    )


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------


def main() -> None:
    with gr.Blocks(title="Qur'an Semantic Search + Commentary") as demo:
        gr.Markdown(
            "# Qur'an Search (OpenAI + Qdrant)\n"
            "Ask a question in natural language."
        )

        query = gr.Textbox(label="Your question", lines=1)
        lang = gr.Dropdown(["en", "ur", "fa", "ps"], value="en", label="Translation language")
        taf = gr.Checkbox(value=True, label="Show tafsir (dataset)")

        go = gr.Button("Search")
        out = gr.Markdown()

        go.click(fn=on_search, inputs=[query, lang, taf], outputs=[out])

    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
