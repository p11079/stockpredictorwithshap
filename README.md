# Antigravity Stock Predictor with SHAP

Streamlit app for next-day stock direction prediction with explainable AI.

## What it does

- Downloads historical price data from Yahoo Finance.
- Builds lightweight technical and sentiment features.
- Trains a tree-based classifier on the fly.
- Predicts next-day direction as `UP` or `DOWN`.
- Explains each prediction with SHAP.
- Shows a simple risk label and action-oriented guidance.

## Core signals

- News sentiment
- Volume ratio
- Momentum over 5 and 10 trading days
- Market regime relative to the 200-day moving average
- Trend context from MA50 and MA200
- RSI and rolling volatility for risk framing

## Why this model

LightGBM is the primary model because it is:

- low maintenance
- fast to retrain
- strong on tabular financial features
- easy to explain with SHAP

If LightGBM is unavailable, the app falls back to scikit-learn `HistGradientBoostingClassifier`.

## Explainability

The app converts SHAP values into:

- top contributing drivers
- a human-readable reason summary
- a plain-language action recommendation
- a simple risk label

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- This project is educational and should not be treated as financial advice.
- Predictions are directional and probabilistic, not guaranteed outcomes.
- The data loader is hardened for Yahoo Finance multi-index responses.
- The training and SHAP explanation flow was stress-tested across repeated runs to reduce crash risk.
