# earnings-signals

**Two Signals, One Event: Spectral and Semantic Analysis of Earnings Announcements**

<<<<<<< HEAD
Author: Nikola Kolev
Course: Data Science, SoftUni 2026
Instructor: Yordan Darakchiev

---

## Research question

Earnings announcements are the most concentrated moments of information revelation in financial markets. At each event, two fundamentally different types of signals exist simultaneously:

1. A **frequency-domain signal** derived from pre-announcement price dynamics (Fourier spectral entropy of log returns).
2. A **semantic signal** derived from management speech during the call itself (Loughran-McDonald financial sentiment).

The project tests whether either signal predicts post-announcement abnormal returns, and whether the two signals are statistically independent.

## Hypotheses (pre-registered before any data was examined)

| ID | Hypothesis | Test | Bonferroni α |
|----|-----------|------|--------------|
| H₁ | Spectral entropy of pre-earnings price dynamics predicts post-announcement CAR | t-test (median split) + Mann-Whitney U robustness | 0.0167 |
| H₂ | LM sentiment in earnings call transcripts predicts post-announcement CAR | Spearman ρ | 0.0167 |
| H₃ | The two signals are statistically independent | Pearson r | 0.0167 |

All tests use a Bonferroni-corrected significance threshold $\alpha_{corrected} = 0.05/3 = 0.0167$ to control the family-wise error rate.

## Data sources

This project uses **two genuinely independent data sources**:

| Source | Type | Provider | Coverage |
|--------|------|----------|----------|
| Daily OHLCV prices | Numerical time series | Yahoo Finance via `yfinance` | 2018–2024 |
| Earnings call transcripts | Unstructured text | Motley Fool via Kaggle | 2019–2023 |

**Universe:** 15 S&P 500 technology companies (AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, INTC, TSLA, NFLX, ORCL, CRM, ADBE, QCOM, TXN), fixed *ex ante* before examining any results.

**Final matched sample:** 123 earnings events (2019-01-01 to 2023-12-31, Q2 2020 excluded as a COVID structural break).

## Project structure

```
earnings-signals/
├── data/
│   ├── raw/                     # gitignored: Motley Fool transcripts (download from Kaggle)
│   └── processed/               # committed: clean CSVs ready for analysis
│       ├── earnings_dates.csv
│       ├── transcripts.csv
│       ├── matched_events.csv
│       ├── sentiment_scores.csv
│       └── analysis_dataset.csv
├── src/                         # all analysis logic (notebook orchestrates only)
│   ├── data/
│   │   ├── price_loader.py      # yfinance wrapper, log returns
│   │   └── event_builder.py     # universe, window alignment, abnormal returns
│   ├── fourier/
│   │   ├── spectral.py          # DFT, PSD, spectral entropy
│   │   └── features.py          # per-event feature extraction pipeline
│   ├── nlp/
│   │   ├── cleaning.py          # transcript loading and filtering
│   │   └── sentiment.py         # LM (pysentiment2) and VADER comparison
│   ├── hypothesis/
│   │   └── tests.py             # H1/H2/H3 with Bonferroni correction, Cohen's d
│   └── viz/
│       └── plots.py             # all figure generation
├── notebooks/
│   └── earnings_analysis.ipynb  # the report — narrative + math + figures
├── requirements.txt
└── README.md
```

## How to reproduce the analysis

The processed CSVs are committed to the repo, so no Kaggle account or network access is required to verify the results.

```bash
# 1. Clone the repo
git clone https://github.com/niki7o/earnings-signals.git
cd earnings-signals

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebooks/earnings_analysis.ipynb

# 4. In Jupyter: Kernel → Restart & Run All
```

Every figure, table, and statistical result will regenerate from the committed processed CSVs. Total runtime is under 60 seconds on a modern laptop.

## How to regenerate the data from scratch (optional)

If you want to rebuild the processed CSVs from raw sources:

1. **Download the Motley Fool transcripts dataset** from Kaggle:
   https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts
   Place the CSV in `data/raw/`.

2. **Run the data pipeline scripts** (in order):
   ```bash
   python -m src.data.price_loader      # downloads OHLCV via yfinance
   python -m src.data.event_builder     # aligns events, computes abnormal returns
   python -m src.nlp.cleaning           # filters transcripts to universe
   python -m src.nlp.sentiment          # scores LM polarity per transcript
   python -m src.fourier.features       # extracts spectral features per event
   ```

This requires ~5 minutes and an internet connection. The Yahoo Finance download is rate-limited.

## Methods

| Stage | Method |
|-------|--------|
| Returns | Log returns: $r_t = \log(P_t / P_{t-1})$ |
| Spectral | Cooley-Tukey FFT, PSD normalization, Shannon entropy |
| Benchmark | SPY ETF as market factor; abnormal returns = stock return − benchmark return |
| Event window | Pre: $[T-30, T-1]$ trading days. Post: $[T, T+4]$ trading days. Strictly non-overlapping |
| Sentiment | Loughran-McDonald financial dictionary via `pysentiment2` (VADER tested but saturates) |
| Significance | Bonferroni-corrected $\alpha = 0.0167$, Cohen's $d$ effect sizes |
| Robustness | Shapiro-Wilk normality test, Mann-Whitney U non-parametric check, per-ticker analysis, TF-IDF discriminating-terms |

## Key results

| Hypothesis | Statistic | p-value | Conclusion |
|------------|-----------|---------|------------|
| H₁ (spectral → CAR) | t = 0.947, U = 2058 | 0.345 / 0.400 | Fail to reject — robust to non-normality |
| H₂ (sentiment → CAR) | Spearman ρ = −0.076 | 0.406 | Fail to reject |
| H₃ (signal independence) | Pearson r = 0.000 | 0.997 | **Signals are orthogonal** |

The null result on H₁ and H₂ is the **predicted outcome under the Efficient Market Hypothesis (Fama, 1970)** in its semi-strong form. The novel finding is H₃: the two signals measure entirely different aspects of the information environment with zero shared variance.

See Section 6 of the notebook for full discussion and limitations.

## References

1. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383–417.
2. Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
3. Tetlock, P. C. (2007). Giving Content to Investor Sentiment: The Role of Media in the Stock Market. *Journal of Finance*, 62(3), 1139–1168.
4. Boudoukh, J., Feldman, R., Kogan, S., & Richardson, M. (2019). Information, Trading, and Volatility: Evidence from Firm-Specific News. *Review of Financial Studies*, 32(3), 992–1033.
5. Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27, 379–423.
6. Cooley, J. W., & Tukey, J. W. (1965). An Algorithm for the Machine Calculation of Complex Fourier Series. *Mathematics of Computation*, 19(90), 297–301.
7. Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM 2014*.
8. Mandelbrot, B. (1963). The Variation of Certain Speculative Prices. *Journal of Business*, 36(4), 394–419.

## License

MIT (see LICENSE if present, otherwise default).
=======
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
>>>>>>> 898d34444bf0803c330db608a591164d627e0c0b
