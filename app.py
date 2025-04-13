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
    background-color: #0D1117;
    color: #E5E7EB;
    font-family: 'Segoe UI', sans-serif;
}
.section {
    background-color: #1A1C20;
    padding: 1.5rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
h3 {
    color: #FFD369;
    margin-bottom: 1rem;
}
hr {
    border-top: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#FFD369;'>Altara</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#8B949E;'>AI-Powered Investment Intelligence</p>", unsafe_allow_html=True)

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
    return f"{round(score, 2)}" if score else "N/A"

def get_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
    articles = requests.get(url).json().get("articles", [])
    relevant = [a["title"] for a in articles if ticker.upper() in a["title"].upper()]
    return relevant[:3] if relevant else [a["title"] for a in articles[:3]]

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
    fig, ax = plt.subplots(figsize=(6, 3.5))
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
    cols[0].markdown(f"**Price:** ${info.get('currentPrice','N/A')}")
    cols[0].markdown(f"**Volume:** {info.get('volume','N/A')}")
    cols[0].markdown(f"**P/E Ratio:** {info.get('trailingPE','N/A')}")
    cols[0].markdown(f"**Market Cap:** {info.get('marketCap','N/A')}")
    cols[1].markdown(f"**52W High:** ${info.get('fiftyTwoWeekHigh','N/A')}")
    cols[1].markdown(f"**52W Low:** ${info.get('fiftyTwoWeekLow','N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- UI Input ---
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("### 📈 Analyze a Stock")
ticker = st.text_input("Enter Stock Symbol (ex. AAPL, TSLA)").upper()
st.markdown("</div>", unsafe_allow_html=True)

# --- Main Action ---
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
You are a financial AI assistant generating a clear investment recommendation for a stock based on the following structured input:

Stock: {ticker}
Price: ${info.get("currentPrice","N/A")}
7D MA: {ma7:.2f}, 30D MA: {ma30:.2f}
7D % Change: {pct}
P/E Ratio: {info.get("trailingPE", "N/A")}
Market Cap: {info.get("marketCap", "N/A")}
52W High: {info.get("fiftyTwoWeekHigh", "N/A")}
52W Low: {info.get("fiftyTwoWeekLow", "N/A")}
Volume: {info.get("volume", "N/A")}

Analyst Ratings: {rating}
Insider Activity: {insider}
Sentiment Score: {sentiment}

Recent News:
- {news[0] if len(news) > 0 else ""}
- {news[1] if len(news) > 1 else ""}
- {news[2] if len(news) > 2 else ""}

Please provide a structured analysis with these 5 clearly labeled sections (use markdown formatting with bold headings):

1. **📊 Overall Sentiment**
2. **📉 Technical Analysis**
3. **🧠 Analyst + Insider Summary**
4. **📰 News Impact**
5. **✅ Final Recommendation**

Each section should be concise, visually readable, and formatted in bullet points where possible.
"""

        with st.spinner("Analyzing with Altara..."):
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
