"""
Price data loader for earnings-signals project.
Downloads and caches OHLCV data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single ticker.
    Returns a flat DataFrame with columns: Open, High, Low, Close, Volume.
    Caches result as CSV to avoid repeated API calls.
    """
    cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return df

    raw = yf.download(ticker, start=start, end=end, progress=False)

 
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cache_path)

    return raw


def get_log_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Compute daily log returns from Close prices.
    r_t = log(P_t / P_{t-1})
    """
    import numpy as np
    return np.log(prices["Close"] / prices["Close"].shift(1)).dropna()