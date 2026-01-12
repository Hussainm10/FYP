import sounddevice as sd
import numpy as np

from tajweed_agent import realtime_word, quran_index

# Default settings
TARGET_SR = 22050          # must match config.SAMPLE_RATE
SURA_DEFAULT = 114
AYAH_DEFAULT = 1
WORD_DEFAULT = 1           # first word of the ayah


def record_from_mic(duration: float, sr: int = TARGET_SR):
    """
    Record from the default microphone for `duration` seconds.
    Returns (audio, sr) where audio is a 1-D numpy array.
    """
    print(f"\nRecording {duration:.3f} seconds at {sr} Hz...")
    frames = int(duration * sr)
    audio = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio[:, 0], sr


def _prompt_int(prompt: str, default: int) -> int:
    """
    Ask the user for an int with a default when they press Enter.
    """
    txt = input(prompt).strip()
    if not txt:
        return default
    try:
        return int(txt)
    except ValueError:
        print(f"Invalid input '{txt}', using default {default}.")
        return default


def main():
    print("=== Mic word test (dynamic Surah/Ayah/Word + CSV durations) ===")

    # ------- ask user which unit to test -------
    surah = _prompt_int(f"Enter Surah number [{SURA_DEFAULT}]: ", SURA_DEFAULT)
    ayah = _prompt_int(f"Enter Ayah number [{AYAH_DEFAULT}]: ", AYAH_DEFAULT)
    word = _prompt_int(f"Enter Word number [{WORD_DEFAULT}]: ", WORD_DEFAULT)

    # ------- look up this word in dtw_units.csv via quran_index -------
    try:
        cfg = quran_index.get_unit_config("word", surah, ayah, word)
    except KeyError as e:
        print(
            f"\n[ERROR] No DTW unit found for "
            f"Surah={surah}, Ayah={ayah}, Word={word}.\n{e}"
        )
        return

    expected = cfg.expected_duration_sec or 1.0
    max_dur = cfg.max_duration_sec or (expected * 3.0)

    # recording window: up to max_dur, but not shorter than 1 second
    record_sec = max(1.0, min(max_dur, expected * 3.0))

    print("\nUsing CSV durations for this word:")
    print(f"  expected_duration_sec = {expected:.3f}s")
    print(f"  max_duration_sec      = {max_dur:.3f}s")
    print(f"→ recording window      = {record_sec:.3f}s")

    # ------- record audio -------
    audio, sr = record_from_mic(record_sec, TARGET_SR)

    # ------- run the realtime matcher -------
    # NOTE: evaluate_word_from_mic(surah, ayah, word, audio, sr, data_root=None)
    result = realtime_word.evaluate_word_from_mic(
        surah,
        ayah,
        word,
        audio,
        sr,
    )

    # ------- pretty-print result -------
    print("\n=== Mic Word Evaluation ===")
    print(f"Surah: {result.surah}, Ayah: {result.ayah}, Word: {result.word}")
    print(f"RMS energy     : {result.rms:.6f}")
    print(f"Duration (sec) : {result.duration_sec:.3f}")
    print(
        f"DTW distance   : "
        f"{result.distance if result.distance is not None else 'N/A'}"
    )
    print(
        f"Average cost   : "
        f"{result.avg_cost if result.avg_cost is not None else 'N/A'}"
    )
    print(f"Similarity score: {result.score * 100:.2f}%")
    print(f"Verdict        : {result.label}")
    print(f"Notes          : {result.notes}")


if __name__ == "__main__":
    main()
