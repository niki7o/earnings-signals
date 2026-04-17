# earnings-signals

**Two Signals, One Event: Spectral and Semantic Analysis of Earnings Announcements**

*Data Science, SoftUni 2026 — Nikola Kolev*

---

## Research Question

Can pre-announcement price dynamics (measured via Fourier spectral entropy) 
or earnings call transcript sentiment (measured via Loughran-McDonald dictionary) 
predict post-announcement abnormal returns in S&P 500 technology stocks?

## Hypotheses

| | Hypothesis | Test | Result |
|---|---|---|---|
| H₁ | Spectral entropy predicts abnormal returns | t-test | Not significant (p=0.345) |
| H₂ | LM sentiment predicts abnormal returns | Spearman ρ | Not significant (p=0.406) |
| H₃ | The two signals are independent | Pearson r | Independent (r≈0, p=0.997) |

Results are consistent with the semi-strong form of the Efficient Market Hypothesis.

---

## Data Sources

| Source | Type | Provider | Events |
|---|---|---|---|
| Daily OHLCV prices | Numerical time series | Yahoo Finance via `yfinance` | 15 tickers, 2019–2023 |
| Earnings call transcripts | Unstructured text | Motley Fool via Kaggle | 18,755 transcripts |

**Matched dataset:** 123 earnings events with both price data and transcript.

---

## Project Structure

```
earnings-signals/
│
├── notebooks/
│   └── earnings_analysis.ipynb   ← main analysis notebook
│
├── src/
│   ├── data/
│   │   ├── price_loader.py       ← yfinance wrapper, log returns, earnings dates
│   │   └── event_builder.py      ← universe, window alignment, abnormal returns
│   ├── fourier/
│   │   ├── spectral.py           ← DFT, PSD, spectral entropy
│   │   └── features.py           ← feature extraction pipeline
│   ├── nlp/
│   │   ├── cleaning.py           ← transcript loading and preprocessing
│   │   └── sentiment.py          ← VADER and Loughran-McDonald sentiment
│   ├── hypothesis/
│   │   └── tests.py              ← H1, H2, H3 with Bonferroni correction
│   └── viz/
│       └── plots.py              ← all visualizations
│
├── data/
│   └── processed/
│       ├── earnings_dates.csv    ← 210 real earnings events
│       ├── transcripts.csv       ← 274 filtered transcripts
│       ├── matched_events.csv    ← 123 matched events
│       ├── sentiment_scores.csv  ← LM and VADER scores
│       └── analysis_dataset.csv  ← final combined dataset
│
├── requirements.txt
└── README.md
```
---

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
jupyter notebook notebooks/earnings_analysis.ipynb
```

### Data
All processed data is included in `data/processed/`. 
Raw price data is downloaded automatically via `yfinance` on first run.
Transcript data is included — no additional downloads required.

---

## Methods

- **Fourier Analysis:** DFT → normalized PSD → spectral entropy per event
- **Sentiment Analysis:** Loughran-McDonald financial dictionary via `pysentiment2`
- **Event Study:** Abnormal returns = stock returns − SPY benchmark returns
- **Hypothesis Testing:** t-test (H1), Spearman correlation (H2), Pearson r (H3)
- **Multiple Testing Correction:** Bonferroni (α = 0.05/3 = 0.0167)
- **Effect Sizes:** Cohen's d for all group comparisons

---

## References

1. Fama (1970) — Efficient Capital Markets
2. Loughran & McDonald (2011) — Financial sentiment dictionaries
3. Shannon (1948) — Information entropy
