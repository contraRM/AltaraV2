import streamlit as st
import yfinance as yf
import requests
from openai import OpenAI
import matplotlib.pyplot as plt
import time
import re

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
ASSISTANT_ID = st.secrets["ASSISTANT_ID"]

st.set_page_config(page_title="Altara", page_icon="📈", layout="wide")

st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color:#1E40AF; font-size: 3em; margin-bottom: 0;'>Altara</h1>
        <p style='font-size: 1.2em; color: #334155;'>AI-Powered Market Intelligence</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

def get_news(stock_name):
    url = f"https://newsapi.org/v2/everything?q={stock_name}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])[:3]
    return [a["title"] for a in articles]

def clean_response(text):
    text = re.sub(r"[\*_`$]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def ask_assistant(prompt):
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)
    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status in ["failed", "cancelled", "expired"]:
            st.error(f"⚠️ Assistant run failed with status: {run_status.status}")
            return "⚠️ Assistant failed to generate a response."
        time.sleep(1)
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages.data[0].content[0].text.value

def build_prompt(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    price = stock.info.get("currentPrice", "unknown")
    volume = stock.info.get("volume", "unknown")
    high = stock.info.get("fiftyTwoWeekHigh", "unknown")
    low = stock.info.get("fiftyTwoWeekLow", "unknown")
    cap = stock.info.get("marketCap", "unknown")
    ma7 = round(hist["Close"].rolling(7).mean().dropna().iloc[-1], 2) if len(hist) >= 7 else "N/A"
    ma30 = round(hist["Close"].rolling(30).mean().dropna().iloc[-1], 2) if len(hist) >= 30 else "N/A"
    pct_change_7d = (
        round(((hist["Close"].iloc[-1] - hist["Close"].iloc[-7]) / hist["Close"].iloc[-7]) * 100, 2)
        if len(hist) >= 7 else "N/A"
    )
    news = get_news(ticker)
    headlines = "- " + "\n- ".join([n[:100] for n in news[:3]])
    return f"""
You are a financial analyst generating a stock report.

Ticker: {ticker}
Current Price: ${price}
7-Day Moving Avg: {ma7}
30-Day Moving Avg: {ma30}
7-Day % Change: {pct_change_7d}%
52-Week High/Low: ${high} / ${low}
Volume: {volume}
Market Cap: {cap}

Recent Headlines:
{headlines}

Based on this data, provide:
- An overall sentiment (bullish, bearish, neutral)
- Technical interpretation (MA, trend)
- Brief interpretation of the news sentiment
- A Buy, Sell, or Hold recommendation with reasoning
"""

def plot_stock_chart(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    hist["MA7"] = hist["Close"].rolling(window=7).mean()
    hist["MA30"] = hist["Close"].rolling(window=30).mean()
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["Close"], label="Close Price", linewidth=2)
    ax.plot(hist.index, hist["MA7"], label="7-Day MA", linestyle="--")
    ax.plot(hist.index, hist["MA30"], label="30-Day MA", linestyle=":")
    ax.set_title(f"{ticker} - Price & Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Watchlist UI
st.markdown("### 🔍 Add a Stock to Watchlist")
new_ticker = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA)")
if st.button("➕ Add to Watchlist"):
    if new_ticker and new_ticker.upper() not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker.upper())

st.markdown("### 📋 Your Watchlist")
if st.session_state.watchlist:
    for ticker in st.session_state.watchlist:
        col1, col2, col3 = st.columns([2, 2, 1])
        try:
            stock = yf.Ticker(ticker)
            price = stock.info.get("currentPrice", "N/A")
            change = stock.history(period="2d")["Close"].pct_change().dropna()
            pct_change = round(change.iloc[-1] * 100, 2) if not change.empty else "N/A"
        except:
            price = "N/A"
            pct_change = "N/A"
        col1.markdown(f"**{ticker}**")
        col2.markdown(f"${price} ({pct_change}%)")
        if col3.button("❌ Remove", key=f"remove_{ticker}"):
            st.session_state.watchlist.remove(ticker)
            st.experimental_rerun()
else:
    st.markdown("Your watchlist is empty.")

# Stock Analysis Section
st.markdown("---")
st.markdown("### 📈 Analyze a Stock")
ticker = st.text_input("Enter a stock to analyze")
if st.button("Analyze"):
    if ticker:
        with st.spinner("Analyzing with Altara AI..."):
            prompt = build_prompt(ticker)
            result = ask_assistant(prompt)

        cleaned_result = clean_response(result)
        styled_result = cleaned_result
        styled_result = styled_result.replace("Overall Sentiment:", "**<span style='color:#1E40AF;'>Overall Sentiment:</span>**")
        styled_result = styled_result.replace("Technical Interpretation:", "**<span style='color:#1E40AF;'>Technical Interpretation:</span>**")
        styled_result = styled_result.replace("News Sentiment:", "**<span style='color:#1E40AF;'>News Sentiment:</span>**")
        styled_result = styled_result.replace("Recommendation:", "**<span style='color:#1E40AF;'>Recommendation:</span>**")

        st.success("✅ AI Analysis Complete")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🧠 Altara Recommendation")
            st.markdown(styled_result, unsafe_allow_html=True)
            with st.expander("📰 View Recent Headlines"):
                for headline in get_news(ticker):
                    st.markdown(f"- {headline}")

        with col2:
            st.markdown("### 📊 Stock Chart with Moving Averages")
            plot_stock_chart(ticker)
    else:
        st.warning("Please enter a valid stock symbol.")