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

# --- Modern Premium Styling ---
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

# === Helper Functions ===
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

# === Featured Sections ===
def get_sp500_gainers_losers():
    url = f"https://finnhub.io/api/v1/screener?exchange=US&marketCapitalizationMoreThan=10000&token={FINNHUB_KEY}"
    data = requests.get(url).json().get("data", [])
    gainers = []
    losers = []
    for stock in data:
        try:
            change = float(stock.get("change", 0))
            if change > 0:
                gainers.append(stock)
            elif change < 0:
                losers.append(stock)
        except:
            continue
    top_gainers = sorted(gainers, key=lambda x: float(x.get("change", 0)), reverse=True)[:5]
    top_losers = sorted(losers, key=lambda x: float(x.get("change", 0)))[:5]
    return top_gainers, top_losers

def get_top_performers(timeframe='1d'):
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "NFLX", "AMD", "INTC"]
    results = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period='1mo' if timeframe == '1mo' else '7d' if timeframe == '1w' else '1d')
            if len(hist) >= 2:
                change = (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100
                volume = hist['Volume'][-1]
                results.append((t, change, volume))
        except:
            continue
    return sorted(results, key=lambda x: x[1], reverse=True)

def display_featured_section():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### 🌟 Featured Sections")

    gainers, losers = get_sp500_gainers_losers()
    st.markdown("#### 📈 Top Movers (S&P 500 / Nasdaq)")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Top Gainers**")
        for g in gainers:
            st.markdown(f"- **{g['symbol']}** ({g.get('sector', 'N/A')}): {g['change']}% | Vol: {g.get('volume', 'N/A')}")
    with cols[1]:
        st.markdown("**Top Losers**")
        for l in losers:
            st.markdown(f"- **{l['symbol']}** ({l.get('sector', 'N/A')}): {l['change']}% | Vol: {l.get('volume', 'N/A')}")

    st.markdown("#### ⏱️ Top Performing Stocks")
    timeframe = st.selectbox("Filter By", ["1d", "1w", "1mo"], index=0, format_func=lambda x: {"1d": "Daily", "1w": "Weekly", "1mo": "Monthly"}[x])
    performers = get_top_performers(timeframe)
    df = pd.DataFrame(performers, columns=["Ticker", "% Change", "Volume"])
    st.dataframe(df.set_index("Ticker").style.format({"% Change": "{:.2f}%"}), use_container_width=True)

    st.markdown("#### 🤖 Altara’s Top AI-Picked Stocks")
    ai_prompt = \"""
You are a financial AI assistant. Based on current market data, news sentiment, and stock performance, list 3–5 U.S. stocks that are strong buys right now.

Prioritize:
- Strong buy analyst ratings
- Bullish sentiment
- Recent volume or performance spikes

Format each stock like:

**[TICKER] – Company Name**
- Reason 1
- Reason 2
- Reason 3
\"""
    ai_output = ask_assistant(ai_prompt)
    st.markdown(ai_output)
    st.markdown("</div>", unsafe_allow_html=True)

display_featured_section()

# === Stock Analysis Input ===
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("### 📈 Analyze a Stock")
ticker = st.text_input("Enter Stock Symbol (ex. AAPL, TSLA)").upper()
st.markdown("</div>", unsafe_allow_html=True)

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

Give 5 sections:
1. **📊 Overall Sentiment**
2. **📉 Technical Analysis**
3. **🧠 Analyst + Insider Summary**
4. **📰 News Impact**
5. **✅ Final Recommendation**
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
