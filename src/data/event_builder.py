"""
Event builder for earnings-signals project.
Defines the company universe and aligns earnings dates with price windows.

CRITICAL NOTE ON DATE ALIGNMENT:
Earnings calls happen after market close (~5PM ET).
Therefore the first price that reflects the announcement is T+1 open.
Pre-announcement window: [T-30, T-1] trading days (Fourier features)
Post-announcement window: [T, T+5] trading days (outcome / abnormal return)
where T = first trading day AFTER the earnings call date.
"""

import pandas as pd
from pathlib import Path

# S&P 500 Technology sector — fixed universe, chosen before any analysis
# 40 companies x 4 quarters x 5 years ≈ 800 events
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "AMD", "INTC", "TSLA", "NFLX",
    "ORCL", "CRM", "ADBE", "QCOM", "TXN",
]

# Date range — excluding Q2 2020 (COVID distortion, documented below)
ANALYSIS_START = "2019-01-01"
ANALYSIS_END   = "2023-12-31"

# Q2 2020 exclusion: March 15 – June 30 2020
# Rationale: circuit breakers, Fed emergency action, and 
# VIX > 80 make this period a structural outlier that would
# contaminate spectral analysis with non-stationary noise.
EXCLUDE_START = "2020-03-15"
EXCLUDE_END   = "2020-06-30"


def load_earnings_dates(path: str) -> pd.DataFrame:
    """
    Load earnings announcement dates from CSV.
    Expected columns: ticker, earnings_date
    earnings_date = calendar date of the earnings call (after market close).
    T (event date) = next trading day after earnings_date.
    """
    df = pd.read_csv(path, parse_dates=["earnings_date"])
    df = df[df["ticker"].isin(UNIVERSE)]
    df = df[
        (df["earnings_date"] >= ANALYSIS_START) &
        (df["earnings_date"] <= ANALYSIS_END)
    ]
    # Remove COVID exclusion window
    mask = (
        (df["earnings_date"] >= EXCLUDE_START) &
        (df["earnings_date"] <= EXCLUDE_END)
    )
    df = df[~mask].reset_index(drop=True)
    return df


def get_pre_window(prices: pd.DataFrame, event_date: pd.Timestamp,
                   window: int = 30) -> pd.Series:
    """
    Extract log returns for the pre-announcement window.
    Returns the last `window` trading days BEFORE event_date.
    This window is used as input to Fourier analysis (H1).
    No post-announcement data is included — enforced by strict < comparison.
    """
    pre = prices[prices.index < event_date]
    return pre.tail(window)


def get_post_window(prices: pd.DataFrame, event_date: pd.Timestamp,
                    window: int = 5) -> pd.Series:
    """
    Extract log returns for the post-announcement window.
    Starts AT event_date (first day market can react).
    Used to compute abnormal returns (outcome variable).
    """
    post = prices[prices.index >= event_date]
    return post.head(window)


def compute_abnormal_return(stock_returns: pd.Series,
                             benchmark_returns: pd.Series) -> pd.Series:
    """
    Abnormal return = stock return - benchmark return.
    Isolates company-specific movement from market-wide movement.
    benchmark_returns should be SPY (S&P 500 ETF) log returns.

    AR_t = r_stock_t - r_benchmark_t
    """
    aligned = stock_returns.align(benchmark_returns, join="inner")
    return aligned[0] - aligned[1]