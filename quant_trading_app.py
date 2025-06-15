# Quantitative Trading Dashboard App
# ---------------------------------
# Save this file as `quant_trading_app.py`
# Requirements: pip install streamlit yfinance pandas numpy ta
# Run with: streamlit run quant_trading_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from ta.momentum import RSIIndicator

# Function to load historical data
@st.cache_data
def load_data(ticker: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

# Main application
def main():
    st.set_page_config(page_title="Quant Trading Dashboard", layout="wide")
    st.title("📈 Quantitative Trading Dashboard")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", "AAPL")
    start_date = st.sidebar.date_input(
        "Start Date", datetime.date.today() - datetime.timedelta(days=365)
    )
    end_date = st.sidebar.date_input(
        "End Date", datetime.date.today()
    )

    if st.sidebar.button("Refresh Data"):
        load_data.clear()

    # Load data
    data = load_data(ticker, start_date, end_date)
    if data.empty:
        st.error("No data found for the given ticker/timeframe.")
        return

    # Price chart
    st.subheader(f"🔍 {ticker} Close Price")
    st.line_chart(data['Close'], use_container_width=True)

    # Moving Average Crossover
    st.sidebar.subheader("Moving Averages")
    sma_short = st.sidebar.slider("Short Window", 5, 50, 10)
    sma_long = st.sidebar.slider("Long Window", 20, 200, 50)
    data['SMA_short'] = data['Close'].rolling(window=sma_short).mean()
    data['SMA_long'] = data['Close'].rolling(window=sma_long).mean()

    st.subheader(f"📊 Moving Average Crossover ({sma_short}, {sma_long})")
    st.line_chart(data[['Close','SMA_short','SMA_long']], use_container_width=True)

    # RSI
    st.sidebar.subheader("RSI Settings")
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    data['RSI'] = RSIIndicator(data['Close'], window=rsi_period).rsi()

    st.subheader(f"⚡ RSI (Period={rsi_period})")
    st.line_chart(data['RSI'], use_container_width=True)

    # Signal & Backtest
    data['Signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, -1)
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Close'].pct_change()
    data['Market_Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    # Cumulative returns
    data['Cumulative_Market'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

    st.subheader("🧮 Strategy vs. Market Performance")
    perf_df = data[['Cumulative_Market','Cumulative_Strategy']]
    perf_df.columns = ['Market','Strategy']
    st.line_chart(perf_df, use_container_width=True)

    # Performance metrics
    st.subheader("📈 Performance Metrics")
    total_ret = perf_df['Strategy'].iloc[-1] - 1
    annual_ret = data['Strategy_Returns'].mean() * 252
    sharpe = (data['Strategy_Returns'].mean() / data['Strategy_Returns'].std()) * np.sqrt(252)

    st.metric("Total Return", f"{total_ret:.2%}")
    st.metric("Annualized Return", f"{annual_ret:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # Data table
    with st.expander("Show raw data"):
        st.dataframe(data.tail(50))

if __name__ == "__main__":
    main()
