import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LEXICON_IN = DATA / "lexicon.json"
AYAH_INDEX_IN = DATA / "ayah_index.json"

LEXICON_OUT = DATA / "lexicon_112_114.json"
AYAH_INDEX_OUT = DATA / "ayah_index_112_114.json"

SURAH_MIN = 112
SURAH_MAX = 114

lex = json.loads(LEXICON_IN.read_text(encoding="utf-8"))
idx = json.loads(AYAH_INDEX_IN.read_text(encoding="utf-8"))

# Filter lexicon rows for surahs 112..114
lex_small = [
    r for r in lex
    if SURAH_MIN <= int(r.get("surah", 0)) <= SURAH_MAX
]

# Filter ayah_index keys for surahs 112..114 (keys like "112:1")
idx_small = {}
for k, v in idx.items():
    try:
        s_str, a_str = str(k).split(":")
        s = int(s_str)
    except Exception:
        continue
    if SURAH_MIN <= s <= SURAH_MAX:
        idx_small[k] = v

LEXICON_OUT.write_text(json.dumps(lex_small, ensure_ascii=False, indent=2), encoding="utf-8")
AYAH_INDEX_OUT.write_text(json.dumps(idx_small, ensure_ascii=False, indent=2), encoding="utf-8")

print("✅ Wrote:", LEXICON_OUT, "rows:", len(lex_small))
print("✅ Wrote:", AYAH_INDEX_OUT, "keys:", len(idx_small))
print("✅ Surahs included:", SURAH_MIN, "to", SURAH_MAX)
