import yfinance as yf
import pandas_datareader.data as web
import datetime

# Define the timeframe for a comprehensive ten-year backtest (e.g., 2016-2026)[cite: 1]
start_date = '2016-01-01'
end_date = '2026-01-01'

# ==========================================
# 1. Yahoo Finance Data Extraction[cite: 1]
# ==========================================
# The primary programmatic tool for equity and commodity futures data is the Python yfinance library[cite: 1].

yf_tickers = [
    '^NSEI',      # Indian Equities: NIFTY 50 Index[cite: 1]
    '^GSPC',      # US Equities: S&P 500 Index[cite: 1]
    'GC=F',       # Precious Metals: COMEX Gold Futures[cite: 1]
    'CL=F',       # Energy Commodities: Crude Oil Futures (WTI/Brent)[cite: 1]
    'DBC',        # Broad Commodities: Bloomberg Commodity Index Tracker[cite: 1]
    'DX-Y.NYB',   # Macro Predictor 1: US Dollar Index[cite: 1]
    '^VIX'        # Macro Predictor 2: CBOE Volatility Index[cite: 1]
]

print("Fetching data from Yahoo Finance...")
# Interface directly with the Yahoo Finance API to download the asset universe[cite: 1]
yf_data = yf.download(yf_tickers, start=start_date, end=end_date)
print("Yahoo Finance Data Fetched Successfully.")
print(yf_data['Close'].head())


# ==========================================
# 2. FRED API Data Extraction[cite: 1]
# ==========================================
# For macroeconomic indicators and sovereign yields, the FRED API is queried using the pandas_datareader library[cite: 1].
# Note: In a production environment, you may need to configure a registered FRED API key[cite: 1].

fred_tickers = [
    'INDIRLTLT01STM',  # Indian Sovereign: India 10-Year Benchmark Yield (monthly)[cite: 1]
    'DGS10'            # US Sovereign: US 10-Year Treasury Yield[cite: 1]
]

print("\nFetching data from FRED...")
# Direct query to FRED using pandas_datareader[cite: 1]
fred_data = web.DataReader(fred_tickers, 'fred', start_date, end_date)
print("FRED Data Fetched Successfully.")
print(fred_data.head())
