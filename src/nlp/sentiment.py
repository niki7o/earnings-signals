"""
Sentiment analysis module for earnings-signals project.

Two instruments applied to each transcript:

1. VADER — general purpose, baseline.
   NOTE: VADER saturates on long texts (8000+ words). All 123 transcripts
   score compound ≈ 1.0 with std = 0.000009. Retained for documentation
   purposes but not used as a feature in hypothesis testing.

2. Loughran-McDonald (LM) via pysentiment2 — finance-specific dictionary.
   Uses stemming. Words like "liability", "risk", "loss" are negative in
   finance but neutral in general English. LM corrects for this.

LM Polarity formula (from pysentiment2):
    Polarity = (N_pos - N_neg) / (N_pos + N_neg)
    Range: [-1, 1]

LM Subjectivity formula:
    Subjectivity = (N_pos + N_neg) / N_total
    Range: [0, 1]
"""

import pandas as pd
import numpy as np
import pysentiment2 as ps
from pathlib import Path

# Initialize once — loading dictionary is expensive
_LM = None

def get_lm_analyzer():
    """Lazy-load the LM analyzer singleton."""
    global _LM
    if _LM is None:
        _LM = ps.LM()
    return _LM


def get_lm_scores(text: str) -> dict:
    """
    Compute Loughran-McDonald sentiment scores for a transcript.

    Parameters
    ----------
    text : raw transcript string

    Returns
    -------
    dict with keys:
        lm_polarity    : (N_pos - N_neg) / (N_pos + N_neg), range [-1, 1]
        lm_subjectivity: (N_pos + N_neg) / N_total, range [0, 1]
        lm_positive    : count of positive stems
        lm_negative    : count of negative stems
        lm_word_count  : total tokens after stemming
    """
    lm = get_lm_analyzer()
    tokens = lm.tokenize(text)
    score  = lm.get_score(tokens)

    return {
        "lm_polarity":     score["Polarity"],
        "lm_subjectivity": score["Subjectivity"],
        "lm_positive":     int(score["Positive"]),
        "lm_negative":     int(score["Negative"]),
        "lm_word_count":   len(tokens),
    }


def get_vader_scores(text: str) -> dict:
    """
    Compute VADER sentiment scores.
    Included for comparison — saturates on long texts.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)

    sia    = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_pos":      scores["pos"],
        "vader_neg":      scores["neg"],
    }


def compute_sentiment(text: str) -> dict:
    """Run both VADER and LM on a single transcript."""
    lm    = get_lm_scores(text)
    vader = get_vader_scores(text)
    return {**lm, **vader}


def build_sentiment_matrix(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment for all matched events.
    Returns DataFrame with sentiment columns.
    """
    print(f"Computing sentiment for {len(matched_df)} transcripts...")
    results = []

    for i, row in matched_df.iterrows():
        scores = compute_sentiment(row["transcript"])
        scores["ticker"]        = row["ticker"]
        scores["earnings_date"] = row["earnings_date"]
        results.append(scores)

        if i % 20 == 0:
            print(f"  {i}/{len(matched_df)}...")

    df = pd.DataFrame(results)
    print("Done.")
    return df