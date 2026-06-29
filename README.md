# PayShield

> Real-time payment transaction risk scoring with fraud analytics and operational impact simulation.

Generates 100,000 synthetic transactions with 90-day timestamps, merchant categories, payment methods, device fingerprints, and IP country risk. Trains a LightGBM model (IEEE-CIS Fraud Detection style) to score fraud risk, then surfaces transaction monitoring, rule engine analytics, operational cost projections, and trend analysis in a production-grade dashboard.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Holdout results (LightGBM):

| Metric | Value |
|---|---|
| ROC AUC | 0.849 |
| Accuracy | 0.791 |
| F1 Score | 0.688 |
| Positive Rate | 33.7% |

Trained on 3,000 samples, tested on 1,000.

## Features

| Component | What it does |
|---|---|
| **Transaction Monitor** | Real-time scrolling feed with fraud score, flags, and risk indicators |
| **Rule Engine** | Threshold configuration, rule triggers, precision/recall trade-off explorer |
| **Operational Impact** | Review queue sizing, ops cost modeling, fraud loss vs review cost optimisation |
| **Trend Analysis** | Fraud rate by merchant category, payment method, device type, time-of-day, and IP risk |

## Repo Structure

```
PayShield/
  train.py     LightGBM training (single model)
  app.py       Streamlit dashboard (1000+ lines)
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic payment transaction dataset: 100,000 records with amount, merchant category, payment method, device type, browser, IP country risk, transaction hour, day of week, and fraud label.

## License

MIT
