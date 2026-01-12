import sounddevice as sd
import numpy as np

from tajweed_agent import quran_index
from tajweed_agent.realtime_ayah import evaluate_ayah_from_mic

TARGET_SR = 22050
SURA_DEFAULT = 114
AYAH_DEFAULT = 1


def record_from_mic(duration: float, sr: int = TARGET_SR):
    print(f"\nRecording {duration:.3f} seconds at {sr} Hz...")
    frames = int(duration * sr)
    audio = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio[:, 0], sr


def _prompt_int(prompt: str, default: int) -> int:
    txt = input(prompt).strip()
    if not txt:
        return default
    try:
        return int(txt)
    except ValueError:
        print(f"Invalid input '{txt}', using default {default}.")
        return default


def main():
    print("=== Mic ayah test (dynamic Surah/Ayah + CSV durations) ===")

    surah = _prompt_int(f"Enter Surah number [{SURA_DEFAULT}]: ", SURA_DEFAULT)
    ayah = _prompt_int(f"Enter Ayah number [{AYAH_DEFAULT}]: ", AYAH_DEFAULT)

    # Ayah rows use word_id=0 in dtw_units.csv
    try:
        cfg = quran_index.get_unit_config("ayah", surah, ayah, 0)
    except KeyError as e:
        print(f"\n[ERROR] No DTW unit found for Surah={surah}, Ayah={ayah}.\n{e}")
        return

    expected = cfg.expected_duration_sec or 3.0
    max_dur = cfg.max_duration_sec or max(expected * 1.5, expected + 2.0)

    # record window: up to max_dur, but not shorter than 2 seconds
    record_sec = max(2.0, min(max_dur, expected * 1.5))

    print("\nUsing CSV durations for this ayah:")
    print(f"  expected_duration_sec = {expected:.3f}s")
    print(f"  max_duration_sec      = {max_dur:.3f}s")
    print(f"→ recording window      = {record_sec:.3f}s")

    audio, sr = record_from_mic(record_sec, TARGET_SR)

    result = evaluate_ayah_from_mic(
        surah,
        ayah,
        audio,
        sr,
    )

    print("\n=== Mic Ayah Evaluation ===")
    print(f"Surah: {result.surah}, Ayah: {result.ayah}")
    print(f"RMS energy     : {result.rms:.6f}")
    print(f"Duration (sec) : {result.duration_sec:.3f}")
    print(f"DTW distance   : {result.distance if result.distance is not None else 'N/A'}")
    print(f"Average cost   : {result.avg_cost if result.avg_cost is not None else 'N/A'}")
    print(f"Similarity score: {result.score * 100:.2f}%")
    print(f"Verdict        : {result.label}")
    print(f"Notes          : {result.notes}")


if __name__ == "__main__":
    main()
