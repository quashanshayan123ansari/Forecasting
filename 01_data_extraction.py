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

import pandas as pd
import numpy as np

# ==========================================
# Phase II: Data Engineering & Preprocessing[cite: 1]
# ==========================================
print("\n--- Starting Phase II: Preprocessing ---")

# 1. Outer Calendar Join[cite: 1]
# Combine the Yahoo Finance closing prices and FRED yields into a single DataFrame
# The outer join ensures we keep all dates from both calendars
portfolio_df = pd.concat([yf_data['Close'], fred_data], axis=1)

# 2. Last Known Value (LKV) Interpolation[cite: 1]
# Use forward-fill to carry the last observed price forward over holidays/outages[cite: 1]
portfolio_df = portfolio_df.ffill()

# Drop the initial rows that might still have NaNs before the first valid data point for all assets
portfolio_df = portfolio_df.dropna()

print("Calendar Alignment and LKV Interpolation Complete.")
print(portfolio_df.head())

# 3. Logarithmic Return Transformation[cite: 1]
# Transform prices/yields into continuously compounded logarithmic returns[cite: 1]
# This closer approximates normality and provides time-additive properties[cite: 1]
log_returns = np.log(portfolio_df / portfolio_df.shift(1))

# Drop the first row since the shift operation creates a NaN
log_returns = log_returns.dropna()

print("\nLogarithmic Returns Calculated Successfully:")
print(log_returns.head())

# ==========================================
# Phase II: Data Engineering & Preprocessing[cite: 1]
# ==========================================
print("\n--- Starting Phase II: Preprocessing ---")

# STRIP TIMEZONES so the Yahoo Finance and FRED calendars align perfectly
yf_data.index = yf_data.index.tz_localize(None)
fred_data.index = fred_data.index.tz_localize(None)

# 1. Outer Calendar Join[cite: 1]
portfolio_df = pd.concat([yf_data['Close'], fred_data], axis=1)

# 2. Last Known Value (LKV) Interpolation[cite: 1]
portfolio_df = portfolio_df.ffill()
portfolio_df = portfolio_df.dropna()

print("Calendar Alignment and LKV Interpolation Complete.")
import itertools
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch

# ==========================================
# Phase III: ARIMAX Mean Modeling[cite: 1]
# ==========================================
print("\n--- Starting Phase III: ARIMAX Modeling ---")

# Define our exogenous variables (Macro Predictors)[cite: 1]
# The document specifies using the log returns of DXY and VIX[cite: 1]
exog_vars = log_returns[['DX-Y.NYB', '^VIX']]

# Define a target asset to model (e.g., Indian Equities)[cite: 1]
target_asset = log_returns['^NSEI']

# Define the bounded parameter space for the grid search: p, q in [0, 4][cite: 1]
# d is set to 0 because we already differenced the data by calculating log returns[cite: 1]
p_values = range(0, 5)
d_values = [0]
q_values = range(0, 5)

# Generate all possible combinations of p, d, q
pdq_combinations = list(itertools.product(p_values, d_values, q_values))

best_bic = float("inf")
best_order = None
best_model = None

print("Executing programmatic grid search. This may take a moment...")

# Execute grid search minimizing the Bayesian Information Criterion (BIC)[cite: 1]
for order in pdq_combinations:
    try:
        # The statsmodels library provides functionality via the SARIMAX class[cite: 1]
        model = sm.tsa.SARIMAX(target_asset, exog=exog_vars, order=order, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        
        if results.bic < best_bic:
            best_bic = results.bic
            best_order = order
            best_model = results
    except Exception as e:
        continue

print(f"\nOptimal ARIMAX Order found: {best_order} with BIC: {best_bic:.2f}")

# Extract residuals[cite: 1]
residuals = best_model.resid

# Perform the ARCH-Lagrange Multiplier (ARCH-LM) test[cite: 1]
# This confirms the presence of conditional heteroscedasticity (volatility clustering)[cite: 1]
arch_test = het_arch(residuals)
print(f"ARCH-LM Test Statistic: {arch_test[0]:.4f}, p-value: {arch_test[1]:.4f}")

if arch_test[1] < 0.05:
    print("Conclusion: The null hypothesis of homoscedasticity is rejected. Residuals exhibit volatility clustering. Proceed to Phase IV (Variance Forecasting).")
else:
    print("Conclusion: No significant volatility clustering detected.")