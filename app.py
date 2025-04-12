import streamlit as st
import yfinance as yf
import requests
from openai import OpenAI
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
import time
import re
import statsmodels.api as sm

# === Configuration ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
ASSISTANT_ID = st.secrets["ASSISTANT_ID"]

st.set_page_config(page_title="Altara Beta", page_icon="📈", layout="wide")

st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color:#1E40AF; font-size: 3em; margin-bottom: 0;'>Altara</h1>
        <p style='font-size: 1.2em; color: #334155;'>AI-Powered Market Intelligence</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# === Utility Functions ===
def clean_response(text):
    text = re.sub(r"[\\*_`$]", "", text)
    return text.strip()

def get_news(stock_name):
    url = f"https://newsapi.org/v2/everything?q={stock_name}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])[:3]
    return [a["title"] for a in articles]

def ask_assistant(prompt):
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)
    while True:
        status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status
        if status == "completed":
            break
        elif status in ["failed", "cancelled", "expired"]:
            return "⚠️ Assistant failed to generate a response."
        time.sleep(1)
    msg = client.beta.threads.messages.list(thread_id=thread.id).data[0]
    return msg.content[0].text.value

# === Forecast Chart ===
def plot_forecast_chart(hist):
    hist = hist.reset_index()
    hist['Close'] = hist['Close'].fillna(method='ffill')
    model = sm.tsa.ExponentialSmoothing(hist['Close'], trend='add').fit()
    forecast = model.forecast(7)
    future_dates = [hist['Date'].iloc[-1] + timedelta(days=i+1) for i in range(7)]

    fig, ax = plt.subplots()
    ax.plot(hist['Date'], hist['Close'], label="Historical", linewidth=2)
    ax.plot(future_dates, forecast, label="7-Day Forecast", linestyle="--", color="orange")
    ax.set_title("🔮 Next 7-Day Forecast")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# === Stock Chart ===
def plot_stock_chart(hist):
    hist["MA7"] = hist["Close"].rolling(7).mean()
    hist["MA30"] = hist["Close"].rolling(30).mean()
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["Close"], label="Close Price", linewidth=2)
    ax.plot(hist.index, hist["MA7"], label="7-Day MA", linestyle="--")
    ax.plot(hist.index, hist["MA30"], label="30-Day MA", linestyle=":")
    ax.set_title("📊 Stock Price & Moving Averages")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# === Market Overview ===
def show_market_overview():
    st.markdown("## 🌍 Market Overview")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM"]
    data = []
    for s in symbols:
        stock = yf.Ticker(s)
        try:
            hist = stock.history(period="2d")
            price = hist["Close"].iloc[-1]
            change = ((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100
            data.append({"Symbol": s, "Price": round(price, 2), "Change (%)": round(change, 2)})
        except:
            continue
    df = pd.DataFrame(data).sort_values("Change (%)", ascending=False)
    st.dataframe(df, use_container_width=True)

# === Build Prompt ===
def ask_assistant_with_data(ticker, hist, news):
    info = yf.Ticker(ticker).info
    price = info.get("currentPrice", "N/A")
    volume = info.get("volume", "N/A")
    market_cap = info.get("marketCap", "N/A")
    high = info.get("fiftyTwoWeekHigh", "N/A")
    low = info.get("fiftyTwoWeekLow", "N/A")

    ma7 = round(hist["Close"].rolling(7).mean().dropna().iloc[-1], 2) if len(hist) >= 7 else "N/A"
    ma30 = round(hist["Close"].rolling(30).mean().dropna().iloc[-1], 2) if len(hist) >= 30 else "N/A"
    pct_change_7d = (
        round(((hist["Close"].iloc[-1] - hist["Close"].iloc[-7]) / hist["Close"].iloc[-7]) * 100, 2)
        if len(hist) >= 7 else "N/A"
    )
    headlines = "- " + "\\n- ".join([n[:100] for n in news])

    input_message = f"""
Ticker: {ticker}
Current Price: ${price}
Volume: {volume}
Market Cap: {market_cap}
7-Day MA: {ma7}
30-Day MA: {ma30}
7-Day % Change: {pct_change_7d}%
52-Week High/Low: ${high} / ${low}
Recent Headlines:
{headlines}
"""
    return ask_assistant(input_message)

# === UI ===
st.markdown("## 📈 Analyze a Stock")
ticker = st.text_input("Enter a stock symbol (e.g., AAPL)").upper()

if st.button("Analyze") and ticker:
    with st.spinner("Analyzing..."):
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            st.warning("No historical data available for this ticker.")
        else:
            news = get_news(ticker)
            response = ask_assistant_with_data(ticker, hist, news)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 💬 Altara Recommendation")
                response = clean_response(response)
                response = response.replace("Overall Sentiment:", "**<span style='color:#1E40AF;'>Overall Sentiment:</span>**")
                response = response.replace("Technical Analysis:", "**<span style='color:#1E40AF;'>Technical Analysis:</span>**")
                response = response.replace("News Sentiment Overview:", "**<span style='color:#1E40AF;'>News Sentiment Overview:</span>**")
                response = response.replace("Recommendation:", "**<span style='color:#1E40AF;'>Recommendation:</span>**")
                st.markdown(response, unsafe_allow_html=True)
                with st.expander("📰 News Headlines"):
                    for h in news:
                        st.markdown(f"- {h}")

            with col2:
                plot_stock_chart(hist)
                plot_forecast_chart(hist)

st.markdown("---")
show_market_overview()
