"""
Feature extraction pipeline for earnings-signals project.
Applies spectral analysis to every event in the dataset.
Output: one row per event with spectral features + metadata.
"""

import numpy as np
import pandas as pd
from src.fourier.spectral import compute_spectral_features
from src.data.event_builder import get_pre_window, get_post_window, compute_abnormal_return


def extract_features_for_event(
    ticker: str,
    event_date: pd.Timestamp,
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    pre_window: int = 30,
    post_window: int = 5,
) -> dict | None:
    """
    Extract all features for a single earnings event.
    Returns None if insufficient data (edge of dataset).
    """
    pre  = get_pre_window(stock_returns, event_date, window=pre_window)
    post = get_post_window(stock_returns, event_date, window=post_window)

  
    if len(pre) < pre_window:
        return None
        
    if len(post) < 1:
        return None

    
    spectral = compute_spectral_features(pre.values)

    # Abnormal return over post-window
    ab = compute_abnormal_return(post, get_post_window(benchmark_returns, event_date, post_window))
    cumulative_abnormal_return = ab.sum()
    day1_abnormal_return = ab.iloc[0] if len(ab) > 0 else np.nan

    return {
        "ticker": ticker,
        "event_date": event_date,
        "spectral_entropy": spectral["spectral_entropy"],
        "dominant_frequency":  spectral["dominant_frequency"],
        "dominant_power":  spectral["dominant_power"],
        "cumulative_abnormal_return": cumulative_abnormal_return,
        "day1_abnormal_return":  day1_abnormal_return,
        "pre_window_size":   len(pre),
        "post_window_size": len(post),
    }


def build_feature_matrix(
    events: pd.DataFrame,
    all_prices: dict,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Build the full feature matrix across all events.

    Parameters
    ----------
    events : DataFrame with columns [ticker, event_date]
    all_prices : dict mapping ticker -> log return Series
    benchmark_returns : SPY log return Series

    Returns
    -------
    DataFrame with one row per event, all features as columns
    """
    rows = []
    for _, row in events.iterrows():
        ticker     = row["ticker"]
        event_date = row["event_date"]

        if ticker not in all_prices:
            continue

        result = extract_features_for_event(
            ticker=ticker,
            event_date=event_date,
            stock_returns=all_prices[ticker],
            benchmark_returns=benchmark_returns,
        )

        if result is not None:
            rows.append(result)

    return pd.DataFrame(rows)