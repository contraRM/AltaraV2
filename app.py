import streamlit as st
import yfinance as yf
import requests
from openai import OpenAI
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
import time
import re

# Setup
st.set_page_config(page_title="Altara", page_icon="📈", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
FINNHUB_KEY = st.secrets["FINNHUB_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
ASSISTANT_ID = st.secrets["ASSISTANT_ID"]

# Header
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
  <h1 style='color:#1E40AF; font-size: 3em;'>Altara</h1>
  <p style='color:#334155;'>AI-Powered Investment Insights</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Helper functions
def get_finnhub(endpoint, params=None):
    url = f"https://finnhub.io/api/v1/{endpoint}"
    params = params or {}
    params["token"] = FINNHUB_KEY
    res = requests.get(url, params=params)
    return res.json()

def get_analyst_rating(ticker):
    data = get_finnhub("stock/recommendation", {"symbol": ticker})
    if not data: return "No analyst ratings available."
    latest = data[0]
    return f"Buy: {latest['buy']}, Hold: {latest['hold']}, Sell: {latest['sell']} (Updated {latest['period']})"

def get_insider_activity(ticker):
    data = get_finnhub("stock/insider-transactions", {"symbol": ticker})
    txns = data.get("data", [])
    if not isinstance(txns, list) or not txns:
        return "No insider activity reported."
    buys = [d for d in txns if d.get("transactionType") == "P - Purchase"]
    sells = [d for d in txns if d.get("transactionType") == "S - Sale"]
    return f"Purchases: {len(buys)}, Sales: {len(sells)} (last 3 months)"

def get_sentiment(ticker):
    data = get_finnhub("news-sentiment", {"symbol": ticker})
    score = data.get("companyNewsScore")
    if score is None: return "No sentiment data available."
    return f"News Sentiment Score: {round(score, 2)} (scale: 0–1)"

def get_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    res = requests.get(url)
    articles = res.json().get("articles", [])
    return [a["title"] for a in articles[:5]]

def clean_response(text):
    text = re.sub(r"[\\*_`$]", "", text)
    return text.strip()

def ask_assistant(prompt):
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)
    while True:
        status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status
        if status == "completed":
            break
        elif status in ["failed", "cancelled", "expired"]:
            return "⚠️ Assistant failed."
        time.sleep(1)
    msg = client.beta.threads.messages.list(thread_id=thread.id).data[0]
    return msg.content[0].text.value

# Forecast chart
def forecast_chart(hist, days):
    model = sm.tsa.ExponentialSmoothing(hist["Close"], trend='add').fit()
    forecast = model.forecast(days)
    future_dates = [hist.index[-1] + timedelta(days=i+1) for i in range(days)]
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["Close"], label="History", linewidth=2)
    ax.plot(future_dates, forecast, label=f"{days}-Day Forecast", linestyle="--", color="orange")
    ax.set_title("🔮 Forecast")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Technical chart
def tech_chart(hist):
    hist["MA7"] = hist["Close"].rolling(7).mean()
    hist["MA30"] = hist["Close"].rolling(30).mean()
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["Close"], label="Close", linewidth=2)
    ax.plot(hist.index, hist["MA7"], label="7D MA", linestyle="--")
    ax.plot(hist.index, hist["MA30"], label="30D MA", linestyle=":")
    ax.set_title("📊 Technical Chart")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Input panel
st.markdown("### 📈 Analyze a Stock")
ticker = st.text_input("Enter Ticker (e.g., AAPL)").upper()
col_risk, col_days = st.columns(2)
risk = col_risk.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
forecast_days = col_days.slider("Forecast Horizon (Days)", 7, 30, 7)

if st.button("Analyze") and ticker:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2mo")
    if hist.empty:
        st.error("Invalid ticker or no data.")
    else:
        # Gather data
        headlines = get_news(ticker)
        rating = get_analyst_rating(ticker)
        insider = get_insider_activity(ticker)
        sentiment = get_sentiment(ticker)
        info = stock.info

        price = info.get("currentPrice", "N/A")
        volume = info.get("volume", "N/A")
        high = info.get("fiftyTwoWeekHigh", "N/A")
        low = info.get("fiftyTwoWeekLow", "N/A")
        market_cap = info.get("marketCap", "N/A")

        ma7 = hist["Close"].rolling(7).mean().dropna().iloc[-1]
        ma30 = hist["Close"].rolling(30).mean().dropna().iloc[-1]
        pct_change = round(((hist["Close"].iloc[-1] - hist["Close"].iloc[-7]) / hist["Close"].iloc[-7]) * 100, 2)

        prompt = f"""
Analyze the following stock:

Ticker: {ticker}
Current Price: ${price}
Volume: {volume}
Market Cap: {market_cap}
52W High/Low: ${high} / ${low}
7D MA: {ma7:.2f}
30D MA: {ma30:.2f}
7D Change: {pct_change}%
Risk Profile: {risk}
Analyst Rating: {rating}
Insider Activity: {insider}
Sentiment: {sentiment}
News:
- {'\\n- '.join(headlines)}
"""

        with st.spinner("Analyzing with Altara AI..."):
            response = ask_assistant(prompt)
            st.success("✅ Analysis Complete")

        # Display
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 💬 Altara Recommendation")
            st.markdown(clean_response(response), unsafe_allow_html=True)
            with st.expander("📰 News Headlines"):
                for h in headlines:
                    st.markdown(f"- {h}")

        with col2:
            tech_chart(hist)
            forecast_chart(hist, forecast_days)
