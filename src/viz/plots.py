"""
Visualization module for earnings-signals project.
All plots use matplotlib with consistent styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def set_style():
    """Apply consistent style to all plots."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         11,
    })


def plot_spectral_example(returns: np.ndarray, ticker: str,
                           event_date: str, features: dict):
    """
    Plot a single event's price series, PSD, and spectral entropy.
    Shows the Fourier pipeline end-to-end for one event.
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Log returns
    axes[0].plot(returns, color="#2c7bb6", linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title(f"Log returns — {ticker} pre {event_date}")
    axes[0].set_xlabel("Trading days before announcement")
    axes[0].set_ylabel("Log return $r_t$")

    # Panel 2: Power Spectral Density
    freqs = features["freqs"]
    psd   = features["psd"]
    axes[1].bar(freqs, psd, width=freqs[1]-freqs[0] if len(freqs)>1 else 0.05,
                color="#d7191c", alpha=0.7)
    axes[1].set_title(f"Power Spectral Density")
    axes[1].set_xlabel("Frequency (cycles/day)")
    axes[1].set_ylabel("Normalized power $P[k]$")

    # Panel 3: Spectral entropy annotation
    axes[2].bar(["Spectral\nEntropy"], [features["spectral_entropy"]],
                color="#1a9641", alpha=0.8, width=0.4)
    axes[2].axhline(np.log(16), color="red", linestyle="--",
                    label=f"Max possible: {np.log(16):.2f}")
    axes[2].set_ylim(0, 3.0)
    axes[2].set_ylabel("$H_{spec}$")
    axes[2].set_title("Spectral entropy")
    axes[2].legend(fontsize=9)
    axes[2].text(0, features["spectral_entropy"] + 0.05,
                 f"{features['spectral_entropy']:.3f}",
                 ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame):
    """
    Plot distributions of spectral entropy, LM polarity, and CAR.
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Spectral entropy
    axes[0].hist(df["spectral_entropy"], bins=20,
                 color="#2c7bb6", alpha=0.7, edgecolor="white")
    axes[0].axvline(df["spectral_entropy"].mean(), color="red",
                    linestyle="--", label=f"Mean: {df['spectral_entropy'].mean():.3f}")
    axes[0].axvline(np.log(16), color="orange", linestyle="--",
                    label=f"Max: {np.log(16):.2f}")
    axes[0].set_title("Distribution of spectral entropy")
    axes[0].set_xlabel("$H_{spec}$")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)

    # LM polarity
    axes[1].hist(df["lm_polarity"], bins=20,
                 color="#d7191c", alpha=0.7, edgecolor="white")
    axes[1].axvline(df["lm_polarity"].mean(), color="red",
                    linestyle="--", label=f"Mean: {df['lm_polarity'].mean():.3f}")
    axes[1].axvline(0, color="black", linestyle="-", linewidth=0.8)
    axes[1].set_title("Distribution of LM sentiment polarity")
    axes[1].set_xlabel("$S_{LM}$")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=9)

    # Cumulative abnormal return
    axes[2].hist(df["cumulative_abnormal_return"], bins=20,
                 color="#1a9641", alpha=0.7, edgecolor="white")
    axes[2].axvline(df["cumulative_abnormal_return"].mean(), color="red",
                    linestyle="--",
                    label=f"Mean: {df['cumulative_abnormal_return'].mean():.3f}")
    axes[2].axvline(0, color="black", linestyle="-", linewidth=0.8)
    axes[2].set_title("Distribution of cumulative abnormal return (CAR)")
    axes[2].set_xlabel("CAR (5-day)")
    axes[2].set_ylabel("Count")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_hypothesis_results(df: pd.DataFrame):
    """
    Plot H1, H2, H3 results side by side.
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # H1: Entropy groups vs CAR
    median_e = df["spectral_entropy"].median()
    high_e = df[df["spectral_entropy"] >= median_e]["cumulative_abnormal_return"]
    low_e  = df[df["spectral_entropy"] <  median_e]["cumulative_abnormal_return"]

    axes[0].boxplot([low_e, high_e], labels=["Low entropy", "High entropy"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#2c7bb6", alpha=0.6))
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("H₁: Spectral entropy vs CAR\n(p = 0.345, not significant)")
    axes[0].set_ylabel("Cumulative abnormal return")

    # H2: Sentiment groups vs CAR
    median_s = df["lm_polarity"].median()
    high_s = df[df["lm_polarity"] >= median_s]["cumulative_abnormal_return"]
    low_s  = df[df["lm_polarity"] <  median_s]["cumulative_abnormal_return"]

    axes[1].boxplot([low_s, high_s], labels=["Low sentiment", "High sentiment"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#d7191c", alpha=0.6))
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("H₂: LM sentiment vs CAR\n(ρ = -0.076, p = 0.406, not significant)")
    axes[1].set_ylabel("Cumulative abnormal return")

    # H3: Signal independence scatter
    axes[2].scatter(df["spectral_entropy"], df["lm_polarity"],
                    alpha=0.5, color="#984ea3", s=40)
    axes[2].set_xlabel("Spectral entropy $H_{spec}$")
    axes[2].set_ylabel("LM polarity $S_{LM}$")
    axes[2].set_title("H₃: Signal independence\n(r = 0.000, p = 0.997, independent)")

    # Add correlation line
    m, b = np.polyfit(df["spectral_entropy"], df["lm_polarity"], 1)
    x_line = np.linspace(df["spectral_entropy"].min(),
                          df["spectral_entropy"].max(), 100)
    axes[2].plot(x_line, m * x_line + b, color="red",
                 linewidth=1.5, linestyle="--", label=f"r = 0.000")
    axes[2].legend()

    plt.tight_layout()
    return fig