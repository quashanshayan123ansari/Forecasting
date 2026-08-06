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