"""
Package init for tajweed_agent.

Expose core modules so they can be imported as:
    from tajweed_agent import config, data_loader, features, dtw_similarity, quran_index, realtime_word
"""

from . import config
from . import data_loader
from . import features
from . import dtw_similarity
from . import mel_cache
from . import realtime_word
from . import quran_index

__all__ = [
    "config",
    "data_loader",
    "features",
    "dtw_similarity",
    "mel_cache",
    "realtime_word",
    "quran_index",
]
