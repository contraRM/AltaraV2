import streamlit as st
import yfinance as yf
import requests
from openai import OpenAI
import matplotlib.pyplot as plt
import pandas as pd
import time
import re
from matplotlib.dates import DateFormatter

# Streamlit config
st.set_page_config(page_title="Altara", page_icon="📈", layout="wide")

# API keys
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
FINNHUB_KEY = st.secrets["FINNHUB_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
ASSISTANT_ID = st.secrets["ASSISTANT_ID"]

# --- Modern Premium CSS ---
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0D1117;
    color: #E5E7EB;
    font-family: 'Segoe UI', sans-serif;
}
.section {
    background-color: #1A1C20;
    padding: 1.5rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(255, 215, 0, 0.05);
}
h1, h2, h3, h4 {
    color: #FFD369;
}
.subtext {
    text-align: center;
    font-size: 1.1rem;
    color: #8B949E;
    margin-bottom: 2rem;
}
input, textarea, .stTextInput > div > div > input {
    background-color: #161B22;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>Altara</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>AI-Powered Investment Insights</p>", unsafe_allow_html=True)

# --- Helper Functions ---
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
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
    articles = requests.get(url).json().get("articles", [])
    relevant = [a["title"] for a in articles if ticker.upper() in a["title"].upper()]
    return relevant[:5] if relevant else [a["title"] for a in articles[:5]]

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
    return re.sub(r"[\\*_`]", "", msg.content[0].text.value).strip()

def tech_chart(hist):
    hist["MA7"] = hist["Close"].rolling(7).mean()
    hist["MA30"] = hist["Close"].rolling(30).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist.index, hist["Close"], label="Close", linewidth=2)
    ax.plot(hist.index, hist["MA7"], label="7D MA", linestyle="--")
    ax.plot(hist.index, hist["MA30"], label="30D MA", linestyle=":")
    ax.set_title("📊 Technical Chart", color="white")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    st.pyplot(fig)

def summary_panel(info):
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### 📋 Stock Summary")
    cols = st.columns(2)
    cols[0].markdown(f"- **Price:** ${info.get('currentPrice','N/A')}")
    cols[0].markdown(f"- **Volume:** {info.get('volume','N/A')}")
    cols[0].markdown(f"- **P/E Ratio:** {info.get('trailingPE','N/A')}")
    cols[0].markdown(f"- **Market Cap:** {info.get('marketCap','N/A')}")
    cols[1].markdown(f"- **52W High:** ${info.get('fiftyTwoWeekHigh','N/A')}")
    cols[1].markdown(f"- **52W Low:** ${info.get('fiftyTwoWeekLow','N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- UI Input ---
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("### 📈 Analyze a Stock")
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL)").upper()
st.markdown("</div>", unsafe_allow_html=True)

# --- On Click ---
if st.button("Run Analysis") and ticker:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2mo")
    if hist.empty:
        st.error("Invalid ticker or no data.")
    else:
        info = stock.info
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
Price: ${info.get("currentPrice","N/A")}
Volume: {info.get("volume","N/A")}
Market Cap: {info.get("marketCap","N/A")}
52W High/Low: ${info.get("fiftyTwoWeekHigh","N/A")} / ${info.get("fiftyTwoWeekLow","N/A")}
7D MA: {ma7:.2f}, 30D MA: {ma30:.2f}
7D Change: {pct}%
Analyst Ratings: {rating}
Insider Activity: {insider}
Sentiment: {sentiment}
News:
- {'\\n- '.join(news)}
"""

        with st.spinner("🧠 Generating AI Insights..."):
            response = ask_assistant(prompt)

        summary_panel(info)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("### 💬 Altara Recommendation")
        st.markdown(response)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📊 View Technical Chart"):
            tech_chart(hist)

        with st.expander("🗞️ View Recent Headlines"):
            for h in news:
                st.markdown(f"- {h}")
