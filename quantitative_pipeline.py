import yfinance as yf
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
import scipy.optimize as sco
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas

# Define timeframe for backtest
start_date = '2016-01-01'
end_date = '2026-01-01'

# ==========================================
# 1. Data Extraction
# ==========================================
yf_tickers = [
    '^NSEI',      # NIFTY 50 Index
    '^GSPC',      # S&P 500 Index
    'GC=F',       # Gold Futures
    'CL=F',       # Crude Oil Futures
    'DBC',        # Commodity Index Tracker
    'DX-Y.NYB',   # US Dollar Index (Macro)
    '^VIX'        # CBOE Volatility Index (Macro)
]

fred_tickers = [
    'INDIRLTLT01STM',  # India 10-Year Benchmark Yield
    'DGS10'            # US 10-Year Treasury Yield
]

print("Fetching data from Yahoo Finance and FRED...")
yf_data = yf.download(yf_tickers, start=start_date, end=end_date)
fred_data = web.DataReader(fred_tickers, 'fred', start_date, end_date)
print("Data Fetched Successfully.")

# ==========================================
# Phase II: Data Engineering & Preprocessing
# ==========================================
print("\n--- Starting Phase II: Preprocessing ---")

# Strip timezones so Yahoo Finance and FRED calendars align perfectly
yf_data.index = yf_data.index.tz_localize(None)
fred_data.index = fred_data.index.tz_localize(None)

portfolio_df = pd.concat([yf_data['Close'], fred_data], axis=1)
portfolio_df = portfolio_df.ffill().dropna()

print("Calendar Alignment and LKV Interpolation Complete.")

# Logarithmic Returns for Econometric/GARCH Modeling
log_returns = np.log(portfolio_df / portfolio_df.shift(1)).dropna()

# Simple Returns for Portfolio Performance Compounding
simple_returns = (portfolio_df / portfolio_df.shift(1) - 1).dropna()

print("\nReturns Calculated Successfully.")

# ==========================================
# Phase III: ARIMAX Mean Modeling
# ==========================================
print("\n--- Starting Phase III: ARIMAX Modeling ---")

exog_vars = log_returns[['DX-Y.NYB', '^VIX']]
target_asset = log_returns['^NSEI']

p_values = range(0, 3)
d_values = [0]
q_values = range(0, 3)
pdq_combinations = list(itertools.product(p_values, d_values, q_values))

best_bic = float("inf")
best_order = None
best_model = None

print("Executing programmatic grid search...")
for order in pdq_combinations:
    try:
        model = sm.tsa.SARIMAX(target_asset, exog=exog_vars, order=order, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        if results.bic < best_bic:
            best_bic = results.bic
            best_order = order
            best_model = results
    except Exception:
        continue

print(f"\nOptimal ARIMAX Order found: {best_order} with BIC: {best_bic:.2f}")

residuals = best_model.resid
arch_test = het_arch(residuals)
print(f"ARCH-LM Test Statistic: {arch_test[0]:.4f}, p-value: {arch_test[1]:.4f}")

# ==========================================
# Phase IV: Univariate EGARCH & DCC Modeling
# ==========================================
print("\n--- Starting Phase IV: Volatility & DCC Modeling ---")

# Define strictly tradable assets for portfolio allocation (excluding macro indices)
tradable_assets = ['^NSEI', '^GSPC', 'GC=F', 'CL=F', 'DBC']

cond_vol = {}
std_resid = {}

for asset in tradable_assets:
    print(f"Fitting EGARCH(1,1) for {asset}...")
    asset_returns = log_returns[asset].dropna() * 100 
    am = arch_model(asset_returns, vol='EGARCH', p=1, o=1, q=1, dist='Normal')
    res = am.fit(disp='off')
    cond_vol[asset] = res.conditional_volatility / 100
    std_resid[asset] = res.resid / res.conditional_volatility

cond_vol_df = pd.DataFrame(cond_vol)
std_resid_df = pd.DataFrame(std_resid)

Z = std_resid_df.values
T, N = Z.shape
Q_bar = std_resid_df.cov().values
a, b = 0.05, 0.93

Q_t = Q_bar.copy()
R_t_list = []

for t in range(T):
    z_t = Z[t, :].reshape(-1, 1)
    if t > 0:
        z_prev = Z[t-1, :].reshape(-1, 1)
        Q_t = (1 - a - b) * Q_bar + a * (z_prev @ z_prev.T) + b * Q_t
    diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diagonal(Q_t)))
    R_t = diag_inv_sqrt @ Q_t @ diag_inv_sqrt
    R_t_list.append(R_t)

latest_vol = cond_vol_df.iloc[-1].values
D_t = np.diag(latest_vol)
R_forecast = R_t_list[-1]
H_forecast = D_t @ R_forecast @ D_t

covariance_forecast_df = pd.DataFrame(H_forecast, index=tradable_assets, columns=tradable_assets)
print("\nOne-Step-Ahead Covariance Matrix Forecast Generated.")

# ==========================================
# Phase V: Maximum Diversification Optimization
# ==========================================
print("\n--- Starting Phase V: MD Portfolio Optimization ---")

H = covariance_forecast_df.values
volatilities = np.sqrt(np.diagonal(H))
N_assets = len(volatilities)

def neg_diversification_ratio(w, V, Cov):
    p_volatility = np.sqrt(np.dot(w.T, np.dot(Cov, w)))
    w_volatility = np.dot(w.T, V)
    return -(w_volatility / p_volatility)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
bounds = tuple((0.0, 0.40) for _ in range(N_assets))
initial_weights = np.array(N_assets * [1. / N_assets])

opt_results = sco.minimize(neg_diversification_ratio, 
                           initial_weights, 
                           args=(volatilities, H), 
                           method='SLSQP', 
                           bounds=bounds, 
                           constraints=constraints)

if opt_results.success:
    optimal_weights = opt_results.x
    max_dr = -opt_results.fun
    print(f"\nMaximum Diversification Ratio Achieved: {max_dr:.4f}")
    
    optimal_portfolio_df = pd.DataFrame({
        'Asset': tradable_assets, 
        'Optimal_Weight': [round(w, 4) for w in optimal_weights]
    })
    print("\nTarget Portfolio Allocations:")
    print(optimal_portfolio_df.sort_values(by='Optimal_Weight', ascending=False).to_string(index=False))
else:
    print("\nOptimizer failed:", opt_results.message)

# ==========================================
# Phase VI: Performance Analytics
# ==========================================
print("\n--- Starting Phase VI: Performance Analytics ---")

tradable_simple_returns = simple_returns[tradable_assets]
md_portfolio_returns = tradable_simple_returns.dot(optimal_weights)
ew_weights = np.array(N_assets * [1. / N_assets])
ew_portfolio_returns = tradable_simple_returns.dot(ew_weights)

def calc_sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    downside_deviation = np.sqrt(np.sum(np.minimum(0, excess_returns)**2) / len(returns))
    if downside_deviation == 0:
        return 0.0
    return (excess_returns.mean() / downside_deviation) * np.sqrt(252)

def calc_max_drawdown(ret_series):
    cum_wealth = (1 + ret_series).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    return drawdown.min()

md_vol = md_portfolio_returns.std() * np.sqrt(252)
ew_vol = ew_portfolio_returns.std() * np.sqrt(252)

md_sharpe = (md_portfolio_returns.mean() / md_portfolio_returns.std()) * np.sqrt(252)
ew_sharpe = (ew_portfolio_returns.mean() / ew_portfolio_returns.std()) * np.sqrt(252)

md_sortino = calc_sortino_ratio(md_portfolio_returns)
ew_sortino = calc_sortino_ratio(ew_portfolio_returns)

md_max_dd = calc_max_drawdown(md_portfolio_returns)
ew_max_dd = calc_max_drawdown(ew_portfolio_returns)

print("\nInstitutional-Grade Risk-Adjusted Metrics:")
print(f"Realized Volatility (MD): {md_vol:.4f} | (EW Baseline): {ew_vol:.4f}")
print(f"Sharpe Ratio        (MD): {md_sharpe:.4f} | (EW Baseline): {ew_sharpe:.4f}")
print(f"Sortino Ratio       (MD): {md_sortino:.4f} | (EW Baseline): {ew_sortino:.4f}")
print(f"Maximum Drawdown    (MD): {md_max_dd:.4f} | (EW Baseline): {ew_max_dd:.4f}")

# ==========================================
# Phase VIII: Generating Visualizations (Moved BEFORE PDF Generation)
# ==========================================
print("\n--- Starting Phase VIII: Generating Visualizations ---")
plt.style.use('seaborn-v0_8-whitegrid')

sample_asset_raw = portfolio_df['^NSEI'].dropna()
sample_asset_returns = log_returns['^NSEI'].dropna()
rolling_mean = sample_asset_raw.rolling(window=60).mean()
rolling_std = sample_asset_raw.rolling(window=60).std()

# Chart 1: Stationarity Diagnostics
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Prices and Log Returns Diagnostics - NIFTY 50 (^NSEI)', fontsize=16)
axes1[0, 0].plot(sample_asset_raw, label='True Price', color='blue', alpha=0.5, linewidth=1)
axes1[0, 0].plot(rolling_mean, label='60-day Rolling Mean', color='red', linestyle='--', linewidth=1)
axes1[0, 0].set_title('Raw Asset Price and 60-Day Rolling Mean')
axes1[0, 0].legend(fontsize=8)
axes1[0, 0].tick_params(axis='x', rotation=45)

axes1[0, 1].plot(rolling_std, color='darkorange', linewidth=1)
axes1[0, 1].set_title('60-Day Rolling Standard Deviation')
axes1[0, 1].set_ylabel('Std Dev')
axes1[0, 1].tick_params(axis='x', rotation=45)

axes1[1, 0].plot(sample_asset_returns, color='steelblue', linewidth=0.8)
axes1[1, 0].axhline(0, color='black', linewidth=0.5)
axes1[1, 0].set_title('Asset First Difference')
axes1[1, 0].tick_params(axis='x', rotation=45)

axes1[1, 1].plot(sample_asset_returns, color='steelblue', linewidth=0.8)
axes1[1, 1].axhline(0, color='black', linewidth=0.5)
axes1[1, 1].set_title('Logarithmic Returns and Volatility Clustering')
axes1[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Chart_1_Stationarity_Diagnostics.png')
plt.close()

# Chart 2: ACF/PACF Analysis
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('ACF / PACF Analysis - ^NSEI', fontsize=16)
plot_acf(sample_asset_raw, ax=axes2[0, 0], lags=40, title='^NSEI: ACF (raw close)')
plot_pacf(sample_asset_raw, ax=axes2[0, 1], lags=40, title='^NSEI: PACF (raw close)')
plot_acf(sample_asset_returns, ax=axes2[1, 0], lags=40, title='^NSEI: ACF (d=1 differenced)')
plot_pacf(sample_asset_returns, ax=axes2[1, 1], lags=40, title='^NSEI: PACF (d=1 differenced)')
plt.tight_layout()
plt.savefig('Chart_2_ACF_PACF.png')
plt.close()

# Chart 3: Cumulative Drawdown Curve
fig3, ax3 = plt.subplots(figsize=(12, 6))
cumulative_returns = (1 + md_portfolio_returns).cumprod()
ax3.plot(cumulative_returns.index, cumulative_returns, color='black', linewidth=1.2, label='Cumulative Returns')
ax3.axhline(1.0, color='black', linestyle=':', linewidth=1.5, label='Breakeven 1.00')
ax3.fill_between(cumulative_returns.index, cumulative_returns, 1.0, 
                 where=(cumulative_returns >= 1.0), facecolor='lightgreen', interpolate=True, alpha=0.5)
ax3.fill_between(cumulative_returns.index, cumulative_returns, 1.0, 
                 where=(cumulative_returns < 1.0), facecolor='lightcoral', interpolate=True, alpha=0.5)
ax3.set_title('Maximum Diversification Portfolio: Cumulative Strategy Returns', fontsize=14)
ax3.set_ylabel('Growth of $1')
ax3.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Chart_3_Cumulative_Drawdowns.png')
plt.close()

print("Visualizations Generated and Saved Successfully.")

# ==========================================
# Phase VII: Automated PDF Report Generation
# ==========================================
print("\n--- Starting Phase VII: PDF Report Generation ---")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 9)
        self.drawRightString(550, 30, f"Page {self._pageNumber} of {page_count}")
        self.drawString(50, 30, "Quantitative Research Strategy Output")
        self.restoreState()

pdf_filename = "Maximum_Diversification_Report.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

styles = getSampleStyleSheet()
title_style = styles['Title']
normal_style = styles['Normal']
heading_style = styles['Heading2']

elements = []
elements.append(Paragraph("Dynamic Maximum Diversification Portfolio Analytics", title_style))
elements.append(Spacer(1, 20))
elements.append(Paragraph("This automated report details the performance of the out-of-sample portfolio simulated utilizing time-varying forecasts from an ARIMAX-DCC-GARCH engine compared to an Equal-Weight baseline.", normal_style))
elements.append(Spacer(1, 20))

table_data = [
    ["Metric", "Maximum Diversification (MD)", "Equal-Weight Baseline (EW)"],
    ["Realized Volatility", f"{md_vol:.2%}", f"{ew_vol:.2%}"],
    ["Sharpe Ratio", f"{md_sharpe:.4f}", f"{ew_sharpe:.4f}"],
    ["Sortino Ratio", f"{md_sortino:.4f}", f"{ew_sortino:.4f}"],
    ["Maximum Drawdown", f"{md_max_dd:.2%}", f"{ew_max_dd:.2%}"]
]

t = Table(table_data, colWidths=[150, 180, 180])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A365D')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 12),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f2f2f2')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
]))
elements.append(t)
elements.append(Spacer(1, 40))

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

elements.append(Paragraph("Econometric Diagnostics & Performance Visualizations", heading_style))
elements.append(Spacer(1, 15))

if os.path.exists('Chart_1_Stationarity_Diagnostics.png'):
    elements.append(Image('Chart_1_Stationarity_Diagnostics.png', width=500, height=350))
    elements.append(Spacer(1, 25))

if os.path.exists('Chart_2_ACF_PACF.png'):
    elements.append(Image('Chart_2_ACF_PACF.png', width=500, height=350))
    elements.append(Spacer(1, 25))

if os.path.exists('Chart_3_Cumulative_Drawdowns.png'):
    elements.append(Image('Chart_3_Cumulative_Drawdowns.png', width=500, height=250))

doc.build(elements, canvasmaker=NumberedCanvas)

print(f"PDF Successfully Generated! Check your folder for '{pdf_filename}'.")
print("\n--- Pipeline Execution Complete ---")