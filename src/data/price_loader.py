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

def fetch_earnings_dates(ticker: str) -> pd.DataFrame:
    """
    Fetch real earnings dates from Yahoo Finance.
    Returns DataFrame with columns: ticker, earnings_date, eps_estimate, reported_eps, surprise_pct
    """
    import yfinance as yf

    t = yf.Ticker(ticker)
    dates = t.earnings_dates

    if dates is None or len(dates) == 0:
        return pd.DataFrame()

    dates = dates.reset_index()
    dates.columns = ["earnings_date", "eps_estimate", "reported_eps", "surprise_pct"]
    dates["ticker"] = ticker
    dates["earnings_date"] = dates["earnings_date"].dt.tz_localize(None).dt.normalize()

    return dates[["ticker", "earnings_date", "eps_estimate", "reported_eps", "surprise_pct"]]


def fetch_all_earnings_dates(universe: list) -> pd.DataFrame:
    """
    Fetch earnings dates for all tickers in universe.
    Saves result to data/processed/earnings_dates.csv
    """
    from pathlib import Path
    import time

    all_dates = []
    for ticker in universe:
        print(f"Fetching {ticker}...")
        try:
            df = fetch_earnings_dates(ticker)
            if len(df) > 0:
                all_dates.append(df)
        except Exception as e:
            print(f"  Warning: {ticker} failed — {e}")
        time.sleep(0.5)  # be polite to Yahoo Finance API

    result = pd.concat(all_dates, ignore_index=True)

    out_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "earnings_dates.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} earnings events to {out_path}")

    return result