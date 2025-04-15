# === (your existing imports and config stay the same) ===
import streamlit as st
import yfinance as yf
import requests
from openai import OpenAI
import matplotlib.pyplot as plt
import pandas as pd
import time
import re
from matplotlib.dates import DateFormatter

st.set_page_config(page_title="Altara", page_icon="📈", layout="wide")

# API keys
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
FINNHUB_KEY = st.secrets["FINNHUB_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
ASSISTANT_ID = st.secrets["ASSISTANT_ID"]

# --- Styling ---
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #F4F7FA;
    color: #0A1D37;
    font-family: 'Segoe UI', sans-serif;
}
.section {
    background-color: #294873;
    padding: 0.2rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
}
h3 {
    color: #1A3C63;
    margin-bottom: 1rem;
}
hr {
    border: none;
    border-top: 1px solid #D1D5DB;
    margin: 2rem 0;
}
.stButton>button {
    border: 1px solid #1A3C63;
    background-color: white;
    color: #1A3C63;
}
.stButton>button:hover {
    background-color: #E8EEF5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;color:#1A3C63;font-size:3em;'>Altara</h1>
<p style='text-align:center;color:#6B7280;font-size:1.2em;'>AI-Powered Market Intelligence</p>
<hr />
""", unsafe_allow_html=True)

# === Market Overview Functions ===
def get_index_summary():
    indexes = {'^DJI': 'Dow Jones', '^IXIC': 'Nasdaq', '^GSPC': 'S&P 500'}
    data = {}
    for symbol, name in indexes.items():
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if hist.empty:
            data[name] = "N/A"
        else:
            close = hist['Close'].iloc[-1]
            prev = hist['Open'].iloc[0]
            change = round((close - prev) / prev * 100, 2)
            data[name] = f"{close:.2f} ({change:+.2f}%)"
    return data

def get_top_movers():
    url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_KEY}"
    headlines = requests.get(url).json()
    tickers = list(set(re.findall(r'\b[A-Z]{1,5}\b', ' '.join([a['headline'] for a in headlines]))))
    movers = []
    for t in tickers[:10]:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="2d")
            if len(hist) == 2:
                pct = round((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100, 2)
                movers.append((t, pct))
        except:
            continue
    movers = sorted(movers, key=lambda x: x[1], reverse=True)
    return movers[:5], movers[-5:]

def get_sector_performance():
    url = f"https://finnhub.io/api/v1/stock/sector-performance?token={FINNHUB_KEY}"
    data = requests.get(url).json()
    return data if data else []

def generate_ai_recommendations():
    prompt = """
You are a financial AI assistant. List 3 U.S. stocks that look attractive right now based on a combination of sentiment, technicals, and growth potential. Present in this format:

1. **[TICKER] - Company Name**
- Reason 1
- Reason 2
- Reason 3

Only list 3 and keep it concise.
"""
    return ask_assistant(prompt)

# === Market Overview UI ===
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("### 🌐 Market Overview")

# Major Index Summary
index_data = get_index_summary()
cols = st.columns(3)
for i, (name, val) in enumerate(index_data.items()):
    cols[i].metric(label=name, value=val)

# Top Gainers / Losers
gainers, losers = get_top_movers()
st.markdown("#### 📈 Top Gainers")
for g in gainers:
    st.markdown(f"- **{g[0]}**: {g[1]:+.2f}%")

st.markdown("#### 📉 Top Losers")
for l in losers:
    st.markdown(f"- **{l[0]}**: {l[1]:+.2f}%")

# Sector Performance
st.markdown("#### 🧩 Sector Performance")
sector_data = get_sector_performance()
for sector in sector_data:
    st.markdown(f"- **{sector['sector']}**: {sector['change']}%")

# AI Stock Recommendations
st.markdown("#### 🤖 Top AI-Generated Stock Picks")
ai_response = generate_ai_recommendations()
st.markdown(ai_response)
st.markdown("</div>", unsafe_allow_html=True)

# === (Your existing Stock Analysis UI remains below this point) ===
# [Rest of your code unchanged: Stock input, analysis, chart, etc.]
