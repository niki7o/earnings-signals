"""
Sentiment analysis module for earnings-signals project.

Two instruments applied to each transcript:
1. VADER — general purpose, baseline
2. Loughran-McDonald (LM) — finance-specific dictionary
   Words like "liability", "risk", "loss" are negative in finance
   but neutral in general English. LM corrects for this.

Sentiment score formula:
    S(d) = (N_pos - N_neg) / (N_pos + N_neg + N_uncertain + 1)

Range: [-1, 1]
    +1 = entirely positive language
    -1 = entirely negative language
     0 = neutral or no financial words found
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


# ── VADER ──────────────────────────────────────────────────────────────────

def get_vader_scores(text: str) -> dict:
    """
    Compute VADER sentiment scores for a transcript.
    Returns compound score in [-1, 1].
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_pos":      scores["pos"],
        "vader_neg":      scores["neg"],
        "vader_neu":      scores["neu"],
    }


# ── Loughran-McDonald ───────────────────────────────────────────────────────

def load_lm_dictionary() -> dict:
    """
    Load Loughran-McDonald financial sentiment word lists.
    Downloads from GitHub if not cached locally.
    Returns dict with keys: positive, negative, uncertain, litigious
    """
    import requests
    import os

    cache_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "lm_dictionary.csv"

    if not cache_path.exists():
        print("Downloading Loughran-McDonald dictionary...")
        url = "https://raw.githubusercontent.com/huangzichun/loughran-mcdonald/master/LoughranMcDonald_SentimentWordLists_2018.csv"
        response = requests.get(url)
        
        if response.status_code != 200:
            # Fallback: use hardcoded core word lists
            return _get_fallback_lm_words()
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(response.text)

    try:
        df = pd.read_csv(cache_path)
        return {
            "positive":   set(df[df["Positive"]  > 0]["Word"].str.lower()),
            "negative":   set(df[df["Negative"]  > 0]["Word"].str.lower()),
            "uncertain":  set(df[df["Uncertainty"]> 0]["Word"].str.lower()),
            "litigious":  set(df[df["Litigious"] > 0]["Word"].str.lower()),
        }
    except Exception:
        return _get_fallback_lm_words()


def _get_fallback_lm_words() -> dict:
    """Hardcoded core LM words as fallback."""
    return {
        "positive": {
            "strong", "strength", "growth", "opportunity", "innovative",
            "exceeded", "record", "outstanding", "profitable", "improved",
            "gain", "gains", "increase", "increases", "benefit", "benefits"
        },
        "negative": {
            "loss", "losses", "decline", "risk", "risks", "uncertainty",
            "weak", "weakness", "difficult", "challenges", "adverse",
            "negative", "impairment", "restructuring", "downturn"
        },
        "uncertain": {
            "may", "might", "could", "possibly", "uncertain", "unclear",
            "approximate", "contingent", "depend", "fluctuate"
        },
        "litigious": {
            "litigation", "lawsuit", "legal", "regulatory", "compliance"
        }
    }


def get_lm_scores(text: str, lm_dict: dict) -> dict:
    """
    Compute Loughran-McDonald sentiment scores.

    Formula:
        S = (N_pos - N_neg) / (N_pos + N_neg + N_uncertain + 1)

    The +1 in denominator prevents division by zero.
    """
    words = re.findall(r"\b[a-z]+\b", text.lower())
    total = len(words)

    if total == 0:
        return {
            "lm_score": 0.0,
            "lm_positive_count": 0,
            "lm_negative_count": 0,
            "lm_uncertain_count": 0,
            "lm_word_count": 0,
        }

    n_pos = sum(1 for w in words if w in lm_dict["positive"])
    n_neg = sum(1 for w in words if w in lm_dict["negative"])
    n_unc = sum(1 for w in words if w in lm_dict["uncertain"])

    score = (n_pos - n_neg) / (n_pos + n_neg + n_unc + 1)

    return {
        "lm_score":           score,
        "lm_positive_count":  n_pos,
        "lm_negative_count":  n_neg,
        "lm_uncertain_count": n_unc,
        "lm_word_count":      total,
    }


# ── Full pipeline ───────────────────────────────────────────────────────────

def compute_sentiment(text: str, lm_dict: dict) -> dict:
    """
    Run both VADER and LM on a single transcript.
    Returns combined dict of all sentiment features.
    """
    vader = get_vader_scores(text)
    lm    = get_lm_scores(text, lm_dict)
    return {**vader, **lm}


def build_sentiment_matrix(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment for all 123 matched events.
    Returns DataFrame with sentiment columns added.
    """
    print("Loading LM dictionary...")
    lm_dict = load_lm_dictionary()
    print(f"LM dictionary: {len(lm_dict['positive'])} positive, "
          f"{len(lm_dict['negative'])} negative words")

    results = []
    for i, row in matched_df.iterrows():
        scores = compute_sentiment(row["transcript"], lm_dict)
        scores["ticker"]        = row["ticker"]
        scores["earnings_date"] = row["earnings_date"]
        results.append(scores)

        if i % 20 == 0:
            print(f"  Processed {i}/{len(matched_df)}...")

    return pd.DataFrame(results)