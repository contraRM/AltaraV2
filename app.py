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

# Theme toggle
mode = st.sidebar.radio("🎨 Theme", ["Dark Mode", "Light Mode"])
bg = "#0F172A" if mode == "Dark Mode" else "#FFFFFF"
card = "#1E293B" if mode == "Dark Mode" else "#F1F5F9"
text = "#F8FAFC" if mode == "Dark Mode" else "#1E293B"

st.markdown(f"""
<style>
html, body, [class*="css"] {{
    background-color: {bg};
    color: {text};
    font-family: 'Segoe UI', sans-serif;
}}
.card {{
    background-color: {card};
    padding: 1.5rem;
    border-radius: 1rem;
    margin-bottom: 1.5rem;
}}
.header-title {{
    text-align: center;
    color: #1E40AF;
    font-size: 3rem;
    margin-bottom: 0;
}}
.subtext {{
    text-align: center;
    font-size: 1.1rem;
    color: #94A3B8;
    margin-bottom: 2rem;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='header-title'>Altara</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>AI-Powered Investment Insights</p>", unsafe_allow_html=True)
st.markdown("---")

# Helpers
def get_finnhub(endpoint, params=None):
    url = f"https://finnhub.io/api/v1/{endpoint}"
    params = params or {}
    params["token"] = FINNHUB_KEY
    return requests.get(url, params=params).json()

def get_analyst_rating(ticker):
    data = get_finnhub("stock/recommendation", {"symbol": ticker})
    if not data: return "No analyst ratings available."
    latest = data[0]
    return f"Buy: {latest['buy']}, Hold: {latest['hold']}, Sell: {latest['sell']} (as of {latest['period']})"

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
    return f"News Sentiment Score: {round(score, 2)} (scale: 0–1)" if score else "No sentiment data available."

def get_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    articles = requests.get(url).json().get("articles", [])
    return [a["title"] for a in articles[:5]]

def ask_assistant(prompt):
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)
    while True:
        status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status
        if status == "completed": break
        elif status in ["failed", "cancelled", "expired"]:
            return "⚠️ Assistant failed."
        time.sleep(1)
    msg = client.beta.threads.messages.list(thread_id=thread.id).data[0]
    return msg.content[0].text.value.strip()

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

def forecast_chart(hist, days):
    model = sm.tsa.ExponentialSmoothing(hist["Close"], trend='add').fit()
    forecast = model.forecast(days)
    future = [hist.index[-1] + timedelta(days=i+1) for i in range(days)]
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["Close"], label="History", linewidth=2)
    ax.plot(future, forecast, label=f"{days}-Day Forecast", linestyle="--", color="orange")
    ax.set_title("🔮 Forecast")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Main app
st.markdown("### 📈 Analyze a Stock")
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL)").upper()
col1, col2 = st.columns(2)
risk_level = col1.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
forecast_days = col2.slider("Forecast Days", 7, 30, 7)

if st.button("Run Analysis") and ticker:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2mo")
    if hist.empty:
        st.error("Invalid ticker or no data.")
    else:
        info = stock.info
        price = info.get("currentPrice", "N/A")
        volume = info.get("volume", "N/A")
        market_cap = info.get("marketCap", "N/A")
        high = info.get("fiftyTwoWeekHigh", "N/A")
        low = info.get("fiftyTwoWeekLow", "N/A")
        ma7 = hist["Close"].rolling(7).mean().dropna().iloc[-1]
        ma30 = hist["Close"].rolling(30).mean().dropna().iloc[-1]
        pct = round(((hist["Close"].iloc[-1] - hist["Close"].iloc[-7]) / hist["Close"].iloc[-7]) * 100, 2)

        rating = get_analyst_rating(ticker)
        insider = get_insider_activity(ticker)
        sentiment = get_sentiment(ticker)
        news = get_news(ticker)

        prompt = f"""
Analyze this stock:

Ticker: {ticker}
Price: ${price}
Volume: {volume}
Market Cap: {market_cap}
52W High/Low: ${high} / ${low}
7D MA: {ma7:.2f}, 30D MA: {ma30:.2f}
7D Change: {pct}%
Risk: {risk_level}
Analyst Ratings: {rating}
Insider Activity: {insider}
Sentiment: {sentiment}
News:
- {'\\n- '.join(news)}
"""
        with st.spinner("🧠 Generating AI Insights..."):
            response = ask_assistant(prompt)

        a, b = st.columns([1, 2])
        with a:
            st.markdown(f"<div class='card'><h4>💬 Altara Recommendation</h4><p>{response}</p></div>", unsafe_allow_html=True)
            with st.expander("🗞️ Recent Headlines"):
                for h in news:
                    st.markdown(f"- {h}")
        with b:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            tech_chart(hist)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            forecast_chart(hist, forecast_days)
            st.markdown("</div>", unsafe_allow_html=True)
