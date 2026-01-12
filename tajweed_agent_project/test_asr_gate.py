import soundfile as sf
import numpy as np

from tajweed_agent import asr_gate, text_norm

def main():
    # Point to an existing clean word/ayah wav from your dataset
    audio_path = "data/QuranAudio/wav/ayahs_wav/sura_114/114_001.wav"

    audio, sr = sf.read(audio_path)
    print("Loaded:", audio_path, "sr=", sr, "shape=", audio.shape)

    asr_res = asr_gate.recognise_text(audio, sr, language="ar", model_name="small")
    print("ASR raw:", asr_res.raw_text)
    print("ASR norm_ar:", asr_res.norm_ar)

    expected_ar = "قل اعوذ برب الناس"  # approximate text
    exp_norm = text_norm.normalize_arabic(expected_ar)
    sim_ar = text_norm.char_similarity(asr_res.norm_ar, exp_norm)

    print("Expected norm:", exp_norm)
    print("Similarity (Arabic):", sim_ar)

if __name__ == "__main__":
    main()
