import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from modules.data_fetcher import fetch_stock_data, fetch_news_data
from modules.feature_engineer import generate_features_and_labels
from modules.model_trainer import train_model, get_latest_data_for_prediction
from modules.explainer import generate_shap_plots
from modules.evaluator import evaluate_model
from utils.helpers import get_api_key

# --- Page Config ---
st.set_page_config(
    page_title="Antigravity Stock Predictor",
    page_icon="📈",
    layout="wide",
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d4a5e;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .prediction-up {
        color: #00ff88;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-down {
        color: #ff4b4b;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🚀 Configuration")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
use_news = st.sidebar.checkbox("Include News Sentiment", value=True)
run_btn = st.sidebar.button("Run Prediction", use_container_width=True)

st.title("🛡️ Antigravity Stock Predictor")
st.markdown("### Next-Day Directional Prediction with SHAP Explainability")

if run_btn:
    api_key = get_api_key()
    
    with st.status("Fetching Data and Training Model...", expanded=True) as status:
        try:
            # 1. Fetch Price Data
            st.write("Fetching OHLCV data from yfinance...")
            price_df = fetch_stock_data(ticker)
            
            # 2. Fetch News Data
            news_list = []
            if use_news:
                st.write("Fetching live news from MarketAux...")
                news_list = fetch_news_data(ticker, api_key, days=2) # Last 48h for better overlap
            
            # 3. Feature Engineering
            st.write("Engineering technical & sentiment features...")
            df = generate_features_and_labels(price_df, news_list)
            
            feature_cols = ['Lag_1', 'Lag_2', 'Lag_3', 'MA50', 'RSI', 'Volatility', 'Sentiment']
            
            # 4. Train Model
            st.write("Training LightGBM classifier...")
            model, X_train, y_train = train_model(df, feature_cols)
            
            # 5. Prediction for Tomorrow
            latest_features = get_latest_data_for_prediction(df, feature_cols)
            prob_up = model.predict(latest_features)[0]
            prediction = "UP" if prob_up > 0.5 else "DOWN"
            
            status.update(label="Complete!", state="complete", expanded=False)
            
            # --- Layout ---
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("## 🎯 Final Prediction")
                if prediction == "UP":
                    st.markdown(f"The stock is predicted to go <span class='prediction-up'>{prediction}</span> tomorrow.", unsafe_allow_html=True)
                else:
                    st.markdown(f"The stock is predicted to go <span class='prediction-down'>{prediction}</span> tomorrow.", unsafe_allow_html=True)
                
                st.metric("Probability of UP", f"{prob_up*100:.1f}%")
                
                # SHAP Summary
                st.markdown("### 📊 Overall Feature Importance (SHAP)")
                fig_summary, fig_force = generate_shap_plots(model, X_train)
                st.pyplot(fig_summary)
                
            with col2:
                # Metrics
                st.markdown("## 📈 Performance Metrics")
                metrics, fig_cm = evaluate_model(model, X_train, y_train)
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                m_col2.metric("MCC Score", f"{metrics['mcc']:.2f}")
                
                m_col1.metric("Precision (Up)", f"{metrics['precision_up']*100:.1f}%")
                m_col2.metric("Recall (Up)", f"{metrics['recall_up']*100:.1f}%")
                
                st.pyplot(fig_cm)

            # Force Plot Section
            st.markdown("---")
            st.markdown("## 🔍 Why this prediction? (Force Plot)")
            st.markdown("This chart shows how each individual feature contributed to *today's* specific prediction.")
            st.pyplot(fig_force)
            
            # News Section
            if use_news and news_list:
                st.markdown("---")
                st.markdown("## 📰 Relevant News Headlines")
                for n in news_list[:5]: # Top 5
                    st.write(f"- **[{n['date']}]** {n['headline']}")
            elif use_news:
                st.info("No relevant news found for this ticker in the last 24h.")
                
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Enter a ticker and click 'Run Prediction' in the sidebar to begin.")
    st.image("https://img.freepik.com/free-vector/stock-market-exchange-graph-with-up-down-arrow_1017-38025.jpg", caption="Antigravity Stock Analysis")
