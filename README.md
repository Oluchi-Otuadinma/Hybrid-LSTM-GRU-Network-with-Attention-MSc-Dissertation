# Hybrid LSTM–GRU Network with Attention (MSc Dissertation)

**Enhancing Stock Price Prediction with Hybrid LSTM-GRU Architectures and Attention Mechanisms on Grouped Time-Series Data**

MSc Dissertation — Department of Computing, Data Science and Artificial Intelligence, Goldsmiths, University of London (September 2025).
Author: Oluomachukwu Otuadinma · Supervisor: Dr V. L. Raju Chinthalapati

## Overview

This repository accompanies the dissertation, which advances grouped time-series forecasting by integrating LSTM and GRU architectures for stock price prediction across four NASDAQGS equities (**AMZN, GOOGL, BALL, QCOM**), using sliding 40-day windows to predict the 41st day. Three hybrid model families are evaluated:

| Family | Notebook | Architecture |
|---|---|---|
| **Parallel LSTM+GRU** (best performer) | `notebooks/GroupPredict_hybrid_LSTMGRU.ipynb` | Per-ticker GRU(160) ‖ LSTM(160) branches → Concatenate → per-ticker heads |
| Sequential LSTM→GRU | `notebooks/GroupPredict_LSTM_GRU.ipynb` | Per-ticker LSTM(160) → Dropout → GRU(160) → concatenate → heads |
| Sequential GRU→LSTM | `notebooks/GroupPredict_GRU_LSTM.ipynb` | Per-ticker GRU(160) → LSTM(160) (directionality ablation) |

Each notebook trains **4 block-architecture variants** (direct / downsizing / tuned-downsizing / stabilised-downsizing, `model3`–`model6`) with per-ticker leak-free MinMax scaling and a chronological 80/20 split, benchmarked against:

- `notebooks/GroupPredict_LSTM_baseline.ipynb` — standalone LSTM baseline (adapted from the Lawi et al. 2022 public Colab — see Attribution)
- `notebooks/Transformer.ipynb` — compact encoder-only Transformer with working `MultiHeadAttention` blocks (the only notebook where attention is actively used)

Headline result: the Parallel LSTM+GRU hybrid reduced MAPE by up to ~15% vs standalone models, while dedicated attention experiments did not improve predictive performance (see dissertation §Results).

## Repository layout

```
├── notebooks/                  # Self-contained experiment notebooks (Colab-era, Keras/TensorFlow)
│   ├── GroupPredict_hybrid_LSTMGRU.ipynb   # ⭐ parallel hybrid (headline model)
│   ├── GroupPredict_LSTM_GRU.ipynb         # sequential LSTM→GRU
│   ├── GroupPredict_GRU_LSTM.ipynb         # sequential GRU→LSTM
│   ├── GroupPredict_LSTM_baseline.ipynb    # standalone LSTM baseline
│   ├── Transformer.ipynb                   # Transformer baseline (working attention)
│   ├── consolidated-plots.ipynb            # regenerates the dissertation comparison figures
│   └── draft-legacy-all-in-one.ipynb       # legacy single-notebook prototype (2010–2022 data)
├── report/
│   └── Stock-Prediction-Dissertation.docx  # full dissertation (incl. Code appendix with Colab links)
├── requirements.txt
└── README.md
```

## Getting started

```bash
pip install -r requirements.txt
jupyter lab notebooks/GroupPredict_hybrid_LSTMGRU.ipynb
```

Notebooks download daily Close data via `yfinance` (2010-01-04 → 2025-09-09) — no API keys required. Metrics per ticker/model: MAE, MSE, RMSE, MAPE, R² (train & test), plus RMSPE/RMDPE accuracy bar-chart comparisons.

## Known caveats (honest notes)

- **Attention**: the custom `AttentionLayer` class is defined in the sequential-hybrid notebooks but is **not wired into the final models** — attention was evaluated during the project and ultimately not adopted (Transformer.ipynb contains the working attention implementation). This matches the dissertation's finding that attention *diminished* performance.
- **Window length**: the parallel-hybrid notebook uses `num_days_used = 60`, while the dissertation text and the other notebooks use 40 — flagged as a known discrepancy.
- Notebook outputs are stripped to keep the repo lightweight; re-run to regenerate plots and metrics.
- These notebooks were developed in Google Colab; expect `/content/...` paths and Colab badges.

## Attribution

The data pipeline and baseline architecture build on **Lawi et al. (2022)**, *Forecasting Stock Prices with Grouped Datasets* — public Colab/GitHub: [armin-lawi/ForcastingStockPrice-with-Grouped-Dataset](https://github.com/armin-lawi/ForcastingStockPrice-with-Grouped-Dataset). The LSTM baseline notebook in particular is adapted from their work.

## License

Released under the [MIT License](LICENSE). The dissertation document itself is © the author; it is included for reference. Portions of the baseline code are derived from Lawi et al. (2022) — see Attribution above.
