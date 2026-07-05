import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from textblob import TextBlob

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


st.set_page_config(page_title="Antigravity Stock Predictor", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3d4a5e; }
    h1, h2, h3 { color: #00d4ff; }
    .prediction-up { color: #00ff88; font-size: 24px; font-weight: bold; }
    .prediction-down { color: #ff4b4b; font-size: 24px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_stock_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    df = df.reset_index()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def fetch_news_data(ticker: str, limit: int = 10) -> list[dict]:
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items = []
    for item in news[:limit]:
        title = item.get("title") or item.get("headline") or ""
        summary = item.get("summary") or ""
        content = f"{title}. {summary}".strip()
        if not content:
            continue
        published = item.get("providerPublishTime")
        date_text = pd.to_datetime(published, unit="s").date().isoformat() if published else ""
        items.append(
            {
                "date": date_text,
                "headline": title,
                "summary": summary,
                "sentiment": TextBlob(content).sentiment.polarity,
            }
        )
    return items


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(price_df: pd.DataFrame, news_list: list[dict] | None = None) -> pd.DataFrame:
    df = price_df.copy()
    df["Return_1D"] = df["Close"].pct_change()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"], 14)
    df["Volatility"] = df["Return_1D"].rolling(20).std()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Momentum_5D"] = df["Close"].pct_change(5)
    df["Momentum_10D"] = df["Close"].pct_change(10)
    df["Market_Regime"] = (df["Close"] > df["MA200"]).astype(int)

    news_list = news_list or []
    news_sentiment = float(np.mean([item["sentiment"] for item in news_list])) if news_list else 0.0
    df["News_Sentiment"] = news_sentiment
    return df.dropna().reset_index(drop=True)


def train_model(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols]
    y = df["Target"]
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    else:
        model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=42)

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def get_latest_data_for_prediction(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df[feature_cols].tail(1)


def get_shap_explanations(model, X_train: pd.DataFrame, latest_features: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    latest_shap = explainer.shap_values(latest_features)
    if isinstance(latest_shap, list):
        latest_shap = latest_shap[1]

    return explainer, shap_values, latest_shap


def summarize_reasons(latest_features: pd.DataFrame, latest_shap: np.ndarray, feature_cols: list[str], top_n: int = 3):
    row = latest_features.iloc[0]
    contribs = []
    for idx, feature in enumerate(feature_cols):
        contribs.append(
            {
                "feature": feature,
                "value": float(row[feature]),
                "shap": float(latest_shap[0][idx]),
                "direction": "bullish" if latest_shap[0][idx] > 0 else "bearish",
            }
        )
    contribs.sort(key=lambda item: abs(item["shap"]), reverse=True)
    return contribs[:top_n]


def recommendation_text(prediction: str, prob_up: float, reasons: list[dict]):
    strength = "strong" if abs(prob_up - 0.5) >= 0.15 else "moderate"
    if prediction == "UP":
        action = "Bullish signal. For lower risk, consider a staged entry and keep a stop-loss in place."
    else:
        action = "Bearish signal. Consider waiting for confirmation or trimming exposure if you already hold the stock."

    why = []
    for r in reasons:
        nice_name = r["feature"].replace("_", " ")
        why.append(f"{nice_name} is acting {r['direction']} for the model.")

    return strength, action, why


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "precision_up": precision_score(y_test, y_pred, zero_division=0),
        "recall_up": recall_score(y_test, y_pred, zero_division=0),
    }, confusion_matrix(y_test, y_pred)


def classify_risk(prob_up: float, volatility: float) -> str:
    confidence_gap = abs(prob_up - 0.5)
    if confidence_gap >= 0.2 and volatility < 0.025:
        return "Low"
    if confidence_gap >= 0.1 or volatility < 0.04:
        return "Medium"
    return "High"


def explain_risk(risk_label: str, prob_up: float) -> str:
    if risk_label == "Low":
        return "The signal is relatively cleaner, but still not guaranteed."
    if risk_label == "Medium":
        return "There is a usable edge, but price noise is still meaningful."
    return "The model is uncertain or the market is noisy, so treat this as a watchlist signal."


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Down", "Up"])
    ax.set_yticks([0, 1], ["Down", "Up"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


st.sidebar.title("🚀 Configuration")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").strip().upper()
run_btn = st.sidebar.button("Run Prediction", use_container_width=True)

st.title("🛡️ Antigravity Stock Predictor")
st.markdown("### Next-day directional prediction with SHAP explainability")

if run_btn:
    try:
        with st.status("Fetching data and training model...", expanded=True) as status:
            st.write("Downloading price history from yfinance...")
            price_df = fetch_stock_data(ticker)

            st.write("Building technical indicators...")
            df = build_features(price_df)

            feature_cols = [
                "MA50",
                "MA200",
                "RSI",
                "Volatility",
                "Volume_Ratio",
                "Momentum_5D",
                "Momentum_10D",
                "Market_Regime",
                "News_Sentiment",
            ]

            st.write("Training the model...")
            model, X_train, X_test, y_train, y_test = train_model(df, feature_cols)

            st.write("Scoring the latest candle...")
            latest_features = get_latest_data_for_prediction(df, feature_cols)
            prob_up = float(model.predict_proba(latest_features)[0][1])
            prediction = "UP" if prob_up >= 0.5 else "DOWN"

            explainer, shap_values, latest_shap = get_shap_explanations(model, X_train, latest_features)
            reasons = summarize_reasons(latest_features, latest_shap, feature_cols, top_n=3)
            strength, action_text, why_text = recommendation_text(prediction, prob_up, reasons)
            risk_label = classify_risk(prob_up, float(latest_features["Volatility"].iloc[0]))
            risk_text = explain_risk(risk_label, prob_up)

            metrics, cm = evaluate_model(model, X_test, y_test)

            status.update(label="Complete!", state="complete", expanded=False)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("## 🎯 Final Prediction")
            if prediction == "UP":
                st.markdown(
                    f"The stock is predicted to go <span class='prediction-up'>{prediction}</span> tomorrow.",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"The stock is predicted to go <span class='prediction-down'>{prediction}</span> tomorrow.",
                    unsafe_allow_html=True,
                )

            st.metric("Probability of UP", f"{prob_up * 100:.1f}%")
            st.metric("Signal Strength", strength.title())
            st.metric("Risk Level", risk_label)

            st.markdown("### Why the model is leaning this way")
            for item in why_text:
                st.write(f"- {item}")

            st.markdown("### What should the user do?")
            st.info(action_text)
            st.caption(risk_text)

            st.markdown("### Overall Feature Importance (SHAP)")
            fig_summary = plt.figure(figsize=(9, 4))
            shap.summary_plot(shap_values, X_train, show=False, plot_type="bar")
            st.pyplot(plt.gcf())
            plt.close("all")

        with col2:
            st.markdown("## 📈 Performance Metrics")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
            m_col2.metric("MCC Score", f"{metrics['mcc']:.2f}")
            m_col1.metric("Precision (Up)", f"{metrics['precision_up'] * 100:.1f}%")
            m_col2.metric("Recall (Up)", f"{metrics['recall_up'] * 100:.1f}%")
            st.pyplot(plot_confusion_matrix(cm))

            st.markdown("### Top SHAP drivers for this prediction")
            driver_df = pd.DataFrame(reasons)
            driver_df["impact"] = driver_df["shap"].apply(lambda x: "Positive" if x > 0 else "Negative")
            st.dataframe(driver_df[["feature", "value", "impact", "shap"]], use_container_width=True, hide_index=True)

            st.markdown("### Model Choice")
            if HAS_LIGHTGBM:
                st.success("Using LightGBM: low maintenance, fast to retrain, and strong on tabular stock features.")
            else:
                st.warning("LightGBM is unavailable, so the app is using HistGradientBoostingClassifier as a fallback.")

            st.markdown("### News Sentiment")
            if news_list:
                avg_sent = float(np.mean([n["sentiment"] for n in news_list]))
                st.write(f"Average news sentiment: `{avg_sent:.2f}`")
                for n in news_list[:3]:
                    st.write(f"- **[{n['date']}]** {n['headline']}")
            else:
                st.info("No recent news items were available from Yahoo Finance for this ticker.")

        st.markdown("---")
        st.markdown("## 🔍 Single-prediction SHAP view")
        force_fig = plt.figure(figsize=(10, 2.8))
        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            latest_shap[0],
            latest_features.iloc[0],
            matplotlib=True,
            show=False,
        )
        st.pyplot(plt.gcf())
        plt.close("all")

    except Exception as e:
        st.error(f"Execution Error: {e}")
        st.exception(e)
else:
    st.info("Enter a ticker and click Run Prediction in the sidebar to begin.")
    st.image(
        "https://img.freepik.com/free-vector/stock-market-exchange-graph-with-up-down-arrow_1017-38025.jpg",
        caption="Antigravity Stock Analysis",
    )
