"""
Hypothesis testing module for earnings-signals project.

Three pre-registered hypotheses:

H1: Spectral entropy of pre-announcement price dynamics contains
    statistically significant information about post-announcement
    abnormal returns.
    Test: Split events into high/low entropy groups (median split).
          Compare mean cumulative abnormal return between groups.
          Method: independent two-sample t-test + Cohen's d effect size.

H2: LM sentiment polarity in earnings call transcripts is associated
    with post-announcement abnormal returns.
    Test: Spearman rank correlation between lm_polarity and CAR.
          Non-parametric — we don't assume normal distribution of sentiment.
          Method: scipy.stats.spearmanr + t-test on high/low sentiment groups.

H3: Spectral entropy and LM sentiment are statistically independent.
    Test: Pearson correlation between the two signals.
          If |r| is small and non-significant, signals are independent
          and carry complementary information.
    Method: scipy.stats.pearsonr

All tests use Bonferroni correction for multiple comparisons.
Alpha = 0.05. Corrected alpha = 0.05 / 3 = 0.0167.

Effect sizes reported for all significant results (Cohen's d).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


ALPHA = 0.05
N_TESTS = 3
ALPHA_CORRECTED = ALPHA / N_TESTS  # Bonferroni correction = 0.0167


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    d = (mean1 - mean2) / pooled_std

    Interpretation:
        |d| < 0.2  → negligible
        |d| < 0.5  → small
        |d| < 0.8  → medium
        |d| >= 0.8 → large
    """
    n1, n2   = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def test_h1(df: pd.DataFrame) -> dict:
    """
    H1: High spectral entropy events have different abnormal returns
        than low spectral entropy events.

    H0: mean CAR(high entropy) == mean CAR(low entropy)
    H1: mean CAR(high entropy) != mean CAR(low entropy)

    Method: median split on spectral_entropy → two groups.
            Independent two-sample t-test (two-tailed).
    """
    median_entropy = df["spectral_entropy"].median()
    high = df[df["spectral_entropy"] >= median_entropy]["cumulative_abnormal_return"].values
    low  = df[df["spectral_entropy"] <  median_entropy]["cumulative_abnormal_return"].values

    t_stat, p_value = stats.ttest_ind(high, low)
    d = cohens_d(high, low)

    return {
        "hypothesis":        "H1",
        "description":       "Spectral entropy predicts abnormal returns",
        "median_entropy":    median_entropy,
        "n_high":            len(high),
        "n_low":             len(low),
        "mean_CAR_high":     np.mean(high),
        "mean_CAR_low":      np.mean(low),
        "t_statistic":       t_stat,
        "p_value":           p_value,
        "alpha_corrected":   ALPHA_CORRECTED,
        "significant":       p_value < ALPHA_CORRECTED,
        "cohens_d":          d,
        "effect_size":       _interpret_d(d),
    }


def test_h2(df: pd.DataFrame) -> dict:
    """
    H2: LM sentiment polarity is correlated with abnormal returns.

    H0: Spearman rho == 0 (no monotonic relationship)
    H1: Spearman rho != 0

    Method: Spearman rank correlation (non-parametric).
            Also t-test on high/low sentiment groups for interpretability.
    """
    rho, p_spearman = stats.spearmanr(
        df["lm_polarity"],
        df["cumulative_abnormal_return"]
    )

    # Also median split for interpretability
    median_sentiment = df["lm_polarity"].median()
    high = df[df["lm_polarity"] >= median_sentiment]["cumulative_abnormal_return"].values
    low  = df[df["lm_polarity"] <  median_sentiment]["cumulative_abnormal_return"].values
    t_stat, p_ttest = stats.ttest_ind(high, low)
    d = cohens_d(high, low)

    return {
        "hypothesis":          "H2",
        "description":         "LM sentiment predicts abnormal returns",
        "spearman_rho":        rho,
        "p_value_spearman":    p_spearman,
        "p_value_ttest":       p_ttest,
        "alpha_corrected":     ALPHA_CORRECTED,
        "significant":         p_spearman < ALPHA_CORRECTED,
        "mean_CAR_high_sent":  np.mean(high),
        "mean_CAR_low_sent":   np.mean(low),
        "cohens_d":            d,
        "effect_size":         _interpret_d(d),
    }


def test_h3(df: pd.DataFrame) -> dict:
    """
    H3: Spectral entropy and LM sentiment are independent signals.

    H0: Pearson r == 0 (signals are uncorrelated)
    H1: Pearson r != 0 (signals carry overlapping information)

    If we FAIL to reject H0 → signals are independent → combining
    them adds predictive value beyond either signal alone.
    """
    r, p_value = stats.pearsonr(
        df["spectral_entropy"],
        df["lm_polarity"]
    )

    return {
        "hypothesis":      "H3",
        "description":     "Spectral entropy and LM sentiment are independent",
        "pearson_r":       r,
        "p_value":         p_value,
        "alpha_corrected": ALPHA_CORRECTED,
        "signals_independent": p_value >= ALPHA_CORRECTED,
        "interpretation":  _interpret_h3(r, p_value),
    }


def run_all_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run H1, H2, H3 and return a summary DataFrame.
    Prints a clean results table.
    """
    h1 = test_h1(df)
    h2 = test_h2(df)
    h3 = test_h3(df)

    print("\n" + "="*60)
    print("HYPOTHESIS TEST RESULTS")
    print(f"Bonferroni-corrected alpha: {ALPHA_CORRECTED:.4f}")
    print("="*60)

    print(f"\nH1 — Spectral entropy → abnormal returns")
    print(f"  t = {h1['t_statistic']:.3f}, p = {h1['p_value']:.4f}", end="  ")
    print("✓ SIGNIFICANT" if h1["significant"] else "✗ not significant")
    print(f"  Cohen's d = {h1['cohens_d']:.3f} ({h1['effect_size']})")
    print(f"  Mean CAR high entropy: {h1['mean_CAR_high']:.4f}")
    print(f"  Mean CAR low entropy:  {h1['mean_CAR_low']:.4f}")

    print(f"\nH2 — LM sentiment → abnormal returns")
    print(f"  ρ = {h2['spearman_rho']:.3f}, p = {h2['p_value_spearman']:.4f}", end="  ")
    print("✓ SIGNIFICANT" if h2["significant"] else "✗ not significant")
    print(f"  Cohen's d = {h2['cohens_d']:.3f} ({h2['effect_size']})")
    print(f"  Mean CAR high sentiment: {h2['mean_CAR_high_sent']:.4f}")
    print(f"  Mean CAR low sentiment:  {h2['mean_CAR_low_sent']:.4f}")

    print(f"\nH3 — Signal independence")
    print(f"  r = {h3['pearson_r']:.3f}, p = {h3['p_value']:.4f}", end="  ")
    print("✓ INDEPENDENT" if h3["signals_independent"] else "✗ correlated")
    print(f"  {h3['interpretation']}")
    print("="*60)

    return {"H1": h1, "H2": h2, "H3": h3}


def _interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def _interpret_h3(r: float, p: float) -> str:
    if p >= ALPHA_CORRECTED:
        return f"Signals are independent (r={r:.3f}, not significant). Combining them adds information."
    return f"Signals are correlated (r={r:.3f}, significant). They partially overlap."