"""
Audio feature extraction utilities.

Functions in this module load raw audio files from disk, ensure
consistency of sample rate and channel configuration, optionally trim
silence, and convert the waveform into a Mel spectrogram in decibel
scale.  These spectrograms can then be compared against the
precomputed reference features using DTW.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from . import config


def _trim_silence(y: np.ndarray, top_db: int) -> np.ndarray:
    """Remove leading and trailing silence from an audio signal.

    A simplistic implementation inspired by ``librosa.effects.trim``.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D array).
    top_db : int
        Threshold in decibels relative to the peak; samples below
        ``peak - top_db`` dB are considered silent and removed.

    Returns
    -------
    np.ndarray
        The trimmed audio signal.
    """
    if y.size == 0:
        return y
    # Compute amplitude threshold
    peak = np.max(np.abs(y))
    if peak <= 0:
        return y
    # Convert top_db to a linear amplitude ratio: ratio = 10**(-top_db/20)
    threshold = peak * (10.0 ** (-top_db / 20.0))
    # Find indices where amplitude exceeds the threshold
    idx = np.where(np.abs(y) > threshold)[0]
    if idx.size == 0:
        return y
    start = idx[0]
    end = idx[-1] + 1
    return y[start:end]


def extract_mel_from_audio(
    audio_path: str | bytes,
    *,
    sample_rate: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    n_mels: int | None = config.N_MELS,
    trim_top_db: int = config.TRIM_TOP_DB,
) -> np.ndarray:
    """Load an audio file and compute its Mel spectrogram in dB scale.

    This implementation avoids external dependencies such as
    ``librosa`` by relying on ``scipy.signal.stft`` to compute the
    short-time Fourier transform and constructing the Mel filter bank
    manually.  See the ``config`` module for default parameter values.
    """
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import math

    # Read the audio file.  ``wavfile.read`` supports PCM WAV files.  It
    # returns the sample rate and the waveform as an array.  Only
    # 16-bit PCM WAV files are supported by this implementation.
    orig_sr, data = wavfile.read(audio_path)
    # Convert integer types to float32 in [-1, 1]
    if data.dtype.kind == "i":
        y = data.astype(np.float32) / max(1, np.iinfo(data.dtype).max)
    else:
        y = data.astype(np.float32)
    # If stereo, average channels to mono
    if y.ndim == 2:
        y = y.mean(axis=1)
    # Resample if needed
    if orig_sr != sample_rate:
        gcd = math.gcd(orig_sr, sample_rate)
        up = sample_rate // gcd
        down = orig_sr // gcd
        y = resample_poly(y, up, down)

    # Trim silence
    y_trimmed = _trim_silence(y, trim_top_db)

    # Determine number of mel bands
    num_mels = n_mels if n_mels is not None else 128

    # Compute STFT using scipy.  We choose a Hann window to match
    # librosa's default.  ``stft`` returns frequencies, times, and the
    # complex STFT matrix (f x t).  We need the squared magnitude.
    f, t, Zxx = stft(
        y_trimmed,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    # Magnitude squared (power) spectrogram.  ``Zxx`` has shape
    # (n_freq_bins, n_frames).
    power_spec = np.abs(Zxx) ** 2

    # Construct Mel filter bank matrix of shape (n_mels, n_fft//2 + 1)
    mel_fb = _mel_filter_bank(n_fft, sample_rate, num_mels)

    # Apply the Mel filter bank
    mel_spectrogram = mel_fb @ power_spec

    # Avoid taking log of zero by adding a small epsilon
    eps = 1e-10
    # Convert to decibel scale relative to maximum value
    mel_db = 10.0 * np.log10(np.maximum(eps, mel_spectrogram))
    mel_db -= mel_db.max()
    return mel_db.astype(np.float32)


def _mel_filter_bank(
    n_fft: int,
    sr: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Create a Mel filter bank similar to librosa's implementation.

    Parameters
    ----------
    n_fft : int
        Number of FFT components.
    sr : int
        Sampling rate of the audio.
    n_mels : int
        Number of Mel filters to create.
    f_min : float, optional
        Minimum frequency (Hz).  Default is 0.
    f_max : float or None, optional
        Maximum frequency (Hz).  If None, uses ``sr/2``.

    Returns
    -------
    np.ndarray
        A matrix of shape (n_mels, n_fft//2 + 1) where each row
        contains the triangular filter weights for a Mel band.
    """
    # Maximum frequency defaults to Nyquist
    if f_max is None:
        f_max = sr / 2.0

    # Convert frequencies to Mel scale
    def hz_to_mel(freq: float) -> float:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # Compute points evenly spaced in the Mel scale
    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(mels)
    # Convert Hz to FFT bin numbers
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    # Filter bank initialisation
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[i - 1], bins[i], bins[i + 1]
        if f_center == f_left:
            f_center += 1
        if f_right == f_center:
            f_right += 1
        # Rising slope
        for k in range(f_left, f_center):
            if f_center - f_left != 0:
                fb[i - 1, k] = (k - f_left) / (f_center - f_left)
        # Falling slope
        for k in range(f_center, f_right):
            if f_right - f_center != 0:
                fb[i - 1, k] = (f_right - k) / (f_right - f_center)
    return fb


def extract_mel_from_array(
    audio: np.ndarray,
    sr: int,
    *,
    sample_rate: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    n_mels: int | None = config.N_MELS,
    trim_top_db: int = config.TRIM_TOP_DB,
) -> np.ndarray:
    """
    Compute a Mel spectrogram (in dB) directly from an in-memory audio array.

    This mirrors `extract_mel_from_audio` but starts from a NumPy array
    instead of loading from disk.

    Parameters
    ----------
    audio : np.ndarray
        Audio samples, shape (T,) or (T, C).
    sr : int
        Original sample rate of `audio`.

    Returns
    -------
    np.ndarray
        Mel spectrogram in dB, shape (n_mels, T_frames).
    """
    from scipy.signal import resample_poly
    import math

    # Ensure float32 numpy array
    y = np.asarray(audio, dtype=np.float32)

    # If stereo or multi-channel, average to mono along last axis
    if y.ndim > 1:
        y = y.mean(axis=1)

    orig_sr = int(sr)

    # Resample to target sample_rate if needed (same approach as extract_mel_from_audio)
    if orig_sr != sample_rate:
        gcd = math.gcd(orig_sr, sample_rate)
        up = sample_rate // gcd
        down = orig_sr // gcd
        y = resample_poly(y, up, down)

    # Trim silence at beginning and end
    y_trimmed = _trim_silence(y, trim_top_db)

    # Number of Mel bands
    num_mels = n_mels if n_mels is not None else 128

    # STFT using the same parameters as extract_mel_from_audio
    f, t, Zxx = stft(
        y_trimmed,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )

    # Power spectrogram
    power_spec = np.abs(Zxx) ** 2  # shape: (freq_bins, time_frames)

    # Mel filter bank
    mel_fb = _mel_filter_bank(n_fft, sample_rate, num_mels)

    # Apply mel filter bank
    mel_spectrogram = mel_fb @ power_spec  # (n_mels, time_frames)

    # Avoid log(0)
    eps = 1e-10
    mel_db = 10.0 * np.log10(np.maximum(eps, mel_spectrogram))
    mel_db -= mel_db.max()

    return mel_db.astype(np.float32)
