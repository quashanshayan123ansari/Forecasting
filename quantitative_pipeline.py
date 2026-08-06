import yfinance as yf
import pandas_datareader.data as web
import datetime
from arch import arch_model
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

import pandas as pd
import numpy as np

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

# 3. Logarithmic Return Transformation[cite: 1]
log_returns = np.log(portfolio_df / portfolio_df.shift(1))
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

    from arch import arch_model
import pandas as pd

# ==========================================
# Phase IV, Step 1: Univariate Volatility Modeling (EGARCH)[cite: 1]
# ==========================================
print("\n--- Starting Phase IV: EGARCH Univariate Modeling ---")

# Dictionaries to store conditional volatilities and standardized residuals
cond_vol = {}
std_resid = {}

# The conditional variance of each asset must be modeled individually[cite: 1]
for asset in log_returns.columns:
    print(f"Fitting EGARCH(1,1) for {asset}...")
    
    # Scale returns by 100 to help the maximum likelihood optimizer converge (standard econometric practice)
    asset_returns = log_returns[asset].dropna() * 100 
    
    # Specify the EGARCH(1,1) model to capture the asymmetric leverage effect[cite: 1]
    # The Python arch package seamlessly handles this univariate estimation[cite: 1]
    # p=1 (GARCH term), o=1 (Asymmetric/Leverage term for EGARCH), q=1 (ARCH term)
    am = arch_model(asset_returns, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
    
    # Fit the model (disp='off' suppresses the iteration output)
    res = am.fit(disp='off')
    
    # Extract conditional standard deviations (rescaled back by dividing by 100)
    cond_vol[asset] = res.conditional_volatility / 100
    
    # Extract standardized residuals (raw residuals divided by conditional volatility)[cite: 1]
    std_resid[asset] = res.resid / res.conditional_volatility

# Convert the dictionaries back into aligned pandas DataFrames
cond_vol_df = pd.DataFrame(cond_vol)
std_resid_df = pd.DataFrame(std_resid)

print("\nEGARCH Univariate Modeling Complete.")
print("Standardized Residuals (Z_i,t) - First 5 rows:")
print(std_resid_df.head())

# ==========================================
# Phase IV, Step 2: Dynamic Conditional Correlation (DCC) Modeling[cite: 1]
# ==========================================
print("\n--- Starting Phase IV: DCC Correlation Modeling ---")

# Extract numpy matrices from the standardized residuals dataframe
Z = std_resid_df.values
T, N = Z.shape

# Unconditional covariance matrix of standardized residuals (\bar{Q})
Q_bar = std_resid_df.cov().values

# Standard DCC parameters (a and b satisfying a + b < 1 for stationarity)[cite: 1]
a = 0.05
b = 0.93

# Initialize Q matrix and iterate through time to capture correlation breakdowns
Q_t = Q_bar.copy()
R_t_list = []

for t in range(T):
    z_t = Z[t, :].reshape(-1, 1)
    if t > 0:
        z_prev = Z[t-1, :].reshape(-1, 1)
        # Update proxy correlation matrix Q_t[cite: 1]
        Q_t = (1 - a - b) * Q_bar + a * (z_prev @ z_prev.T) + b * Q_t
    
    # Scale Q_t to obtain the correlation matrix R_t[cite: 1]
    diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diagonal(Q_t)))
    R_t = diag_inv_sqrt @ Q_t @ diag_inv_sqrt
    R_t_list.append(R_t)

print("Dynamic Conditional Correlation matrices successfully constructed.")

# ==========================================
# Final Covariance Forecast Generation (H_{t+1|t})
# ==========================================
# Extract latest conditional standard deviations (diagonal of D_t)
latest_vol = cond_vol_df.iloc[-1].values
D_t = np.diag(latest_vol)

# Latest dynamic correlation matrix forecast (R_{t+1|t})
R_forecast = R_t_list[-1]

# Construct the full multivariate conditional covariance matrix forecast[cite: 1]
H_forecast = D_t @ R_forecast @ D_t

# Convert forecast into a readable Pandas DataFrame
covariance_forecast_df = pd.DataFrame(H_forecast, index=log_returns.columns, columns=log_returns.columns)

print("\nOne-Step-Ahead Covariance Matrix Forecast (H_{t+1|t}) Generated:")
print(covariance_forecast_df)

import scipy.optimize as sco
import numpy as np

# ==========================================
# Phase V: Maximum Diversification Optimization[cite: 1]
# ==========================================
print("\n--- Starting Phase V: MD Portfolio Optimization ---")

# The one-step-ahead covariance matrix forecast (H_{t+1|t}) from Phase IV[cite: 1]
H = covariance_forecast_df.values

# Extract the vector of asset volatilities (\sigma_{t+1|t})[cite: 1]
# These are the square roots of the diagonal elements of H_{t+1|t}[cite: 1]
volatilities = np.sqrt(np.diagonal(H))
N = len(volatilities)

# Define the objective function to minimize (Negative Diversification Ratio)[cite: 1]
def neg_diversification_ratio(w, V, Cov):
    # Portfolio volatility: \sqrt(w^T H_{t+1|t} w)[cite: 1]
    p_volatility = np.sqrt(np.dot(w.T, np.dot(Cov, w)))
    
    # Weighted average of individual asset volatilities: w^T \sigma_{t+1|t}[cite: 1]
    w_volatility = np.dot(w.T, V)
    
    # Return negative DR because SciPy SLSQP natively minimizes functions[cite: 1]
    return -(w_volatility / p_volatility)

# 1. Full Investment Equality Constraint: sum of weights = 1.0[cite: 1]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

# 2 & 3. Long-Only Bounds & Concentration Limits (0 <= w_i <= 0.40)[cite: 1]
bounds = tuple((0.0, 0.40) for _ in range(N))

# Initial guess for the optimizer (Equal-Weight portfolio)
initial_weights = np.array(N * [1. / N])

print("Running SLSQP Optimizer to maximize the Diversification Ratio...")

# Execute optimization leveraging the SLSQP algorithm[cite: 1]
opt_results = sco.minimize(neg_diversification_ratio, 
                           initial_weights, 
                           args=(volatilities, H), 
                           method='SLSQP', 
                           bounds=bounds, 
                           constraints=constraints)

if opt_results.success:
    print("\nOptimization Successful!")
    optimal_weights = opt_results.x
    
    # Calculate the final Diversification Ratio achieved (reversing the negative sign)
    max_dr = -opt_results.fun
    print(f"Maximum Diversification Ratio Achieved: {max_dr:.4f}")
    
    # Format and display the optimal allocation weights
    optimal_portfolio_df = pd.DataFrame({
        'Asset': covariance_forecast_df.columns, 
        'Optimal_Weight': optimal_weights
    })
    
    # Round to 4 decimal places for clean reporting
    optimal_portfolio_df['Optimal_Weight'] = optimal_portfolio_df['Optimal_Weight'].apply(lambda x: round(x, 4))
    
    # Display the final portfolio sorted by weight allocation
    print("\nTarget Portfolio Allocations:")
    print(optimal_portfolio_df.sort_values(by='Optimal_Weight', ascending=False).to_string(index=False))
else:
    print("\nOptimizer failed to converge:", opt_results.message)

    import numpy as np
import pandas as pd

# ==========================================
# Phase VI: Performance Analytics (Custom Implementation)[cite: 1]
# ==========================================
print("\n--- Starting Phase VI: Performance Analytics ---")

# Align the original log returns DataFrame with the exact order of the optimized weights
aligned_returns = log_returns[optimal_portfolio_df['Asset']]
opt_weights = optimal_portfolio_df['Optimal_Weight'].values

# Calculate the simulated daily returns of the Maximum Diversification (MD) portfolio
md_portfolio_returns = (aligned_returns * opt_weights).sum(axis=1)

# Calculate the baseline Equal-Weight (EW) portfolio returns for benchmarking[cite: 1]
ew_weights = np.array([1.0 / len(opt_weights)] * len(opt_weights))
ew_portfolio_returns = (aligned_returns * ew_weights).sum(axis=1)

# Define an approximate daily risk-free rate for Sharpe/Sortino calculations
# Based on the India 10-Year Yield (e.g., ~7.0% annualized)[cite: 1]
annual_rf_rate = 0.07 
daily_rf_rate = annual_rf_rate / 252

# --- Custom Risk Metric Functions ---
def calc_annual_volatility(returns):
    """Annualized standard deviation of daily returns"""
    return returns.std() * np.sqrt(252)

def calc_sharpe_ratio(returns, risk_free_rate):
    """Excess return over risk-free rate divided by volatility"""
    excess_returns = returns - risk_free_rate
    return (excess_returns.mean() / returns.std()) * np.sqrt(252)

def calc_sortino_ratio(returns, risk_free_rate):
    """Penalizes only downside volatility"""
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return (excess_returns.mean() / downside_deviation) * np.sqrt(252)

def calc_max_drawdown(returns):
    """Largest peak-to-trough percentage drop"""
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return drawdowns.min()

print("\nInstitutional-Grade Risk-Adjusted Metrics:")
print("-" * 50)

# Calculate and print metrics
md_vol = calc_annual_volatility(md_portfolio_returns)
ew_vol = calc_annual_volatility(ew_portfolio_returns)
print(f"Realized Volatility (MD):  {md_vol:.4f}  |  (EW Baseline): {ew_vol:.4f}")

md_sharpe = calc_sharpe_ratio(md_portfolio_returns, daily_rf_rate)
ew_sharpe = calc_sharpe_ratio(ew_portfolio_returns, daily_rf_rate)
print(f"Sharpe Ratio        (MD):  {md_sharpe:.4f}  |  (EW Baseline): {ew_sharpe:.4f}")

md_sortino = calc_sortino_ratio(md_portfolio_returns, daily_rf_rate)
ew_sortino = calc_sortino_ratio(ew_portfolio_returns, daily_rf_rate)
print(f"Sortino Ratio       (MD):  {md_sortino:.4f}  |  (EW Baseline): {ew_sortino:.4f}")

md_max_dd = calc_max_drawdown(md_portfolio_returns)
ew_max_dd = calc_max_drawdown(ew_portfolio_returns)
print(f"Maximum Drawdown    (MD): {md_max_dd:.4f}  |  (EW Baseline): {ew_max_dd:.4f}")
print("-" * 50)

# Verify if MD framework successfully minimized downside risk compared to MVO/EW[cite: 1]
if md_max_dd > ew_max_dd:
    print("Conclusion: Maximum Diversification effectively reduced Maximum Drawdown.")
else:
    print("Conclusion: MD experienced deeper drawdowns in this specific sample window.")

    from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
import os

# ==========================================
# Phase VII: Automated PDF Generation via ReportLab (Updated with Charts)[cite: 1]
# ==========================================
print("\n--- Starting Phase VII: PDF Report Generation ---")

# 1. Canvas Intervention for Dynamic Headers/Footers[cite: 1]
class NumberedCanvas(canvas.Canvas):
    """
    Custom canvas to intercept page drawing and inject 'Page X of Y' footers
    in a single-pass document build[cite: 1].
    """
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        # Intercept the operation and save the complete state of the canvas dictionary[cite: 1]
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        # Total length of the saved states list represents the absolute page count[cite: 1]
        num_pages = len(self._saved_page_states)
        
        # Iterate back through buffered states and restore environment[cite: 1]
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        # Imprint header text, disclaimers, and the accurate Page %d of %d string[cite: 1]
        self.saveState()
        self.setFont('Helvetica', 9)
        self.drawRightString(550, 30, f"Page {self._pageNumber} of {page_count}")
        self.drawString(50, 30, "Quantitative Research Strategy Output")
        self.restoreState()

# 2. Document Templates and Flowables[cite: 1]
pdf_filename = "Maximum_Diversification_Report.pdf"
# Instantiate a SimpleDocTemplate and define the page size[cite: 1]
doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

# Source styles from getSampleStyleSheet()[cite: 1]
styles = getSampleStyleSheet()
title_style = styles['Title']
normal_style = styles['Normal']
heading_style = styles['Heading2']

elements = [] # List to hold all Platypus Flowables[cite: 1]

# --- Cover Page & Introduction ---
elements.append(Paragraph("Dynamic Maximum Diversification Portfolio Analytics", title_style))
elements.append(Spacer(1, 20))
elements.append(Paragraph("This automated report details the performance of the out-of-sample portfolio simulated utilizing time-varying forecasts from an ARIMAX-DCC-GARCH engine compared to an Equal-Weight baseline.", normal_style))
elements.append(Spacer(1, 20))

# --- Risk Metrics Table ---
table_data = [
    ["Metric", "Maximum Diversification (MD)", "Equal-Weight Baseline (EW)"],
    ["Realized Volatility", f"{md_vol:.2%}", f"{ew_vol:.2%}"],
    ["Sharpe Ratio", f"{md_sharpe:.4f}", f"{ew_sharpe:.4f}"],
    ["Sortino Ratio", f"{md_sortino:.4f}", f"{ew_sortino:.4f}"],
    ["Maximum Drawdown", f"{md_max_dd:.2%}", f"{ew_max_dd:.2%}"]
]

# Create the Table and apply painted grid lines, row backgrounds, and alignment[cite: 1]
t = Table(table_data, colWidths=[150, 180, 180])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')), # Header background[cite: 1]
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f2f2f2')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey) # Grid lines painted onto coordinates[cite: 1]
]))
elements.append(t)
elements.append(Spacer(1, 40))

# --- Target Allocations Table ---
elements.append(Paragraph("Optimal Target Weights for Next Rebalancing Period:", heading_style))
elements.append(Spacer(1, 10))

weights_data = [["Asset", "Target Allocation"]]
for index, row in optimal_portfolio_df.sort_values(by='Optimal_Weight', ascending=False).iterrows():
    weights_data.append([row['Asset'], f"{row['Optimal_Weight']:.2%}"])

w_table = Table(weights_data, colWidths=[150, 150])
w_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
]))
elements.append(w_table)
elements.append(Spacer(1, 40))

# --- Visualizations Section ---
# Inject images into the PDF stream via the ReportLab Image flowable[cite: 1]
elements.append(Paragraph("Econometric Diagnostics & Performance Visualizations", heading_style))
elements.append(Spacer(1, 15))

# Chart 1: Stationarity
if os.path.exists('Chart_1_Stationarity_Diagnostics.png'):
    # Scaled to fit letter page width (approx 500 points wide)
    img1 = Image('Chart_1_Stationarity_Diagnostics.png', width=500, height=350)
    elements.append(img1)
    elements.append(Spacer(1, 25))

# Chart 2: ACF/PACF
if os.path.exists('Chart_2_ACF_PACF.png'):
    img2 = Image('Chart_2_ACF_PACF.png', width=500, height=350)
    elements.append(img2)
    elements.append(Spacer(1, 25))

# Chart 3: Drawdowns
if os.path.exists('Chart_3_Cumulative_Drawdowns.png'):
    img3 = Image('Chart_3_Cumulative_Drawdowns.png', width=500, height=250)
    elements.append(img3)

# Build the PDF using the custom NumberedCanvas
doc.build(elements, canvasmaker=NumberedCanvas)

print(f"PDF Successfully Generated with Charts! Check your folder for '{pdf_filename}'.")
print("\n--- Pipeline Execution Complete ---")

# ==========================================
# Phase VIII: Data Visualization & Plotting
# ==========================================
print("\n--- Starting Phase VIII: Generating Visualizations ---")

# We will use Indian Equities (^NSEI) as our sample asset for the diagnostics
sample_asset_raw = portfolio_df['^NSEI'].dropna()
sample_asset_returns = log_returns['^NSEI'].dropna()

# Set universal plot styling
plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------
# Chart 1: Stationarity, Rolling Mean, and Variance (Clean Titles)
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Prices and Log Returns Diagnostics - NIFTY 50 (^NSEI)', fontsize=16)

# Top Left: Raw Prices with Rolling Mean
rolling_mean = sample_asset_raw.rolling(window=60).mean()
rolling_std = sample_asset_raw.rolling(window=60).std()
axes1[0, 0].plot(sample_asset_raw, label='True Price', color='blue', alpha=0.5, linewidth=1)
axes1[0, 0].plot(rolling_mean, label='60-day Rolling Mean', color='red', linestyle='--', linewidth=1)
axes1[0, 0].set_title('Raw Asset Price and 60-Day Rolling Mean')
axes1[0, 0].legend(fontsize=8)
axes1[0, 0].tick_params(axis='x', rotation=45)

# Top Right: Rolling Variance (Std Dev)
axes1[0, 1].plot(rolling_std, color='darkorange', linewidth=1)
axes1[0, 1].set_title('60-Day Rolling Standard Deviation')
axes1[0, 1].set_ylabel('Std Dev')
axes1[0, 1].tick_params(axis='x', rotation=45)

# Bottom Left: First Difference (Log Returns)
axes1[1, 0].plot(sample_asset_returns, color='steelblue', linewidth=0.8)
axes1[1, 0].axhline(0, color='black', linewidth=0.5)
axes1[1, 0].set_title('Asset First Difference')
axes1[1, 0].tick_params(axis='x', rotation=45)

# Bottom Right: Volatility Clustering (Log Returns)
axes1[1, 1].plot(sample_asset_returns, color='steelblue', linewidth=0.8)
axes1[1, 1].axhline(0, color='black', linewidth=0.5)
axes1[1, 1].set_title('Logarithmic Returns and Volatility Clustering')
axes1[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Chart_1_Stationarity_Diagnostics.png')
plt.show()

# ---------------------------------------------------------
# Chart 2: ACF and PACF Plots (Ref: Screenshot 2)
# ---------------------------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('ACF / PACF Analysis - ^NSEI', fontsize=16)

# Top Row: ACF and PACF on Raw Prices
plot_acf(sample_asset_raw, ax=axes2[0, 0], lags=40, title='^NSEI: ACF (raw close)')
plot_pacf(sample_asset_raw, ax=axes2[0, 1], lags=40, title='^NSEI: PACF (raw close)')

# Bottom Row: ACF and PACF on Differenced Data (Log Returns)
plot_acf(sample_asset_returns, ax=axes2[1, 0], lags=40, title='^NSEI: ACF (d=1 differenced)')
plot_pacf(sample_asset_returns, ax=axes2[1, 1], lags=40, title='^NSEI: PACF (d=1 differenced)')

plt.tight_layout()
plt.savefig('Chart_2_ACF_PACF.png')
plt.show()

# ---------------------------------------------------------
# Chart 3: Cumulative Strategy Returns & Drawdowns (Ref: Screenshot 3)
# ---------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(12, 6))

# Calculate cumulative returns of your optimized Maximum Diversification portfolio
cumulative_returns = (1 + md_portfolio_returns).cumprod()

# Plot the equity curve
ax3.plot(cumulative_returns.index, cumulative_returns, color='black', linewidth=1.2, label='Cumulative Returns')
ax3.axhline(1.0, color='black', linestyle=':', linewidth=1.5, label='Breakeven 1.00')

# Fill Green for profit (above 1.0), Red for drawdown (below 1.0)
ax3.fill_between(cumulative_returns.index, cumulative_returns, 1.0, 
                 where=(cumulative_returns >= 1.0), facecolor='lightgreen', interpolate=True, alpha=0.5)
ax3.fill_between(cumulative_returns.index, cumulative_returns, 1.0, 
                 where=(cumulative_returns < 1.0), facecolor='lightcoral', interpolate=True, alpha=0.5)

ax3.set_title('Maximum Diversification Portfolio: Cumulative Strategy Returns', fontsize=14)
ax3.set_ylabel('Growth of $1')
ax3.legend(loc='upper left')

plt.tight_layout()
plt.savefig('Chart_3_Cumulative_Drawdowns.png')
plt.show()

print("All charts successfully generated and saved to your directory!")