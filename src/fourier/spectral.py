"""
Spectral analysis module for earnings-signals project.

Mathematical foundation:
Given a time series of log returns r[0], r[1], ..., r[N-1],
we compute the Discrete Fourier Transform (DFT):

    X[k] = sum_{n=0}^{N-1} r[n] * exp(-2*pi*i*k*n / N)

The normalized Power Spectral Density (PSD):

    P[k] = |X[k]|^2 / sum_j |X[j]|^2

Spectral Entropy (our main feature for H1):

    H_spec = -sum_k P[k] * log(P[k])

High H_spec → energy spread across many frequencies → disordered price
Low H_spec  → energy concentrated in few frequencies → dominant rhythm
"""

import numpy as np
from typing import Tuple


def compute_dft(returns: np.ndarray) -> np.ndarray:
    """
    Compute the Discrete Fourier Transform of a return series.
    Uses numpy FFT (Fast Fourier Transform algorithm).
    Returns complex-valued frequency coefficients X[k].
    """
    return np.fft.rfft(returns)


def compute_psd(X: np.ndarray) -> np.ndarray:
    """
    Compute normalized Power Spectral Density from DFT coefficients.
    P[k] = |X[k]|^2 / sum_j |X[j]|^2
    Sum of P[k] = 1 (normalized, treated as probability distribution).
    """
    power = np.abs(X) ** 2
    total = power.sum()
    if total == 0:
        raise ValueError("Zero power spectrum — constant return series.")
    return power / total


def compute_spectral_entropy(psd: np.ndarray) -> float:
    """
    Compute spectral entropy from normalized PSD.
    H_spec = -sum_k P[k] * log(P[k])

    Analogous to Shannon entropy but applied to frequency domain.
    Range: [0, log(N)] where N is number of frequency bins.
    High value → spread spectrum (disordered/noisy price)
    Low value  → peaked spectrum (dominant frequency component)
    """
   
    psd_safe = np.where(psd > 0, psd, 1e-12)
    return -np.sum(psd_safe * np.log(psd_safe))


def compute_spectral_features(returns: np.ndarray) -> dict:
    """
    Full pipeline: returns array → spectral features dict.
    Returns all intermediate values for transparency and plotting.
    """
    X   = compute_dft(returns)
    psd = compute_psd(X)
    H   = compute_spectral_entropy(psd)

    freqs = np.fft.rfftfreq(len(returns))
    dominant_freq = freqs[np.argmax(psd)]
    dominant_power = psd.max()

    return {
        "spectral_entropy":   H,
        "dominant_frequency": dominant_freq,
        "dominant_power":dominant_power,
        "psd": psd,
        "freqs": freqs,
        "n_obs":  len(returns),
    }