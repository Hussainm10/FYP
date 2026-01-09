"""
Tiny CLI demo to test retrieval on CPU.

Usage:
    python -m search.demo_query --q "mercy" --lang en --topk 3 --taf
"""

from __future__ import annotations

import argparse

from .retrieval import QuranSearcher, SearchConfig, render_hit


def main() -> None:
    """
    Parse args, run a search, and print formatted hits.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "--query", dest="query", required=True, help="Your question or keywords")
    parser.add_argument("--lang", default="en", help="Translation language to show (default: en)")
    parser.add_argument("--topk", type=int, default=3, help="Number of hits (default: 3)")
    parser.add_argument("--taf", action="store_true", help="Include tafsir snippet")
    args = parser.parse_args()

    # CPU runtime config
    cfg = SearchConfig(device="cpu")
    searcher = QuranSearcher(cfg)

    hits = searcher.search_ayahs(args.query, top_k=args.topk)
    if not hits:
        print("No results.")
        return

    for i, h in enumerate(hits, 1):
        print("=" * 80)
        print(f"[{i}]")
        print(render_hit(h, lang=args.lang, show_tafsir=args.taf))

        # optional: fetch word tokens for the first result to prove linkage
        if i == 1:
            words = searcher.fetch_words(int(h["surah_num"]), int(h["ayah_num"]))
            if words:
                sample = " | ".join([w.get("arabic", "") for w in words[:6]])
                print(f"[word-by-word] {sample}")


if __name__ == "__main__":
    main()
