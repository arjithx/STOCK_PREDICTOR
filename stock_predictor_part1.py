# ============================================================
# STOCK PREDICTOR — PART 1
# Data Fetching + Technical Indicators + Linear Regression
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────

def fetch_stock_data(ticker: str, start_date: str = '2020-01-01', end_date: str = None) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    Returns a clean DataFrame with a DatetimeIndex.
    """
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    print(f"\nFetching '{ticker}' from {start_date} to {end_date} ...")

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    # Flatten MultiIndex columns produced by yfinance ≥ 0.2
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    print(f"Loaded {len(df):,} trading days of data.")
    return df


# ─────────────────────────────────────────────
# 2. TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append Moving Averages (7, 21, 50-day) and RSI-14 to the DataFrame.
    All calculations operate on a copy so the original is not mutated.
    """
    df = df.copy()

    # ── Moving Averages ──────────────────────
    df['MA_7']  = df['Close'].rolling(window=7,  min_periods=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21, min_periods=21).mean()
    df['MA_50'] = df['Close'].rolling(window=50, min_periods=50).mean()

    # ── RSI-14 ───────────────────────────────
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    rs    = gain / loss.replace(0, np.nan)          # avoid divide-by-zero
    df['RSI'] = 100 - (100 / (1 + rs))

    # ── Auxiliary features ───────────────────
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_MA']    = df['Volume'].rolling(window=20, min_periods=20).mean()

    # Drop rows where any indicator is NaN (first ~50 rows)
    df.dropna(inplace=True)

    print(f"Added technical indicators  |  {len(df):,} rows remain after NaN removal.")
    return df


# ─────────────────────────────────────────────
# 3. FEATURE PREPARATION
# ─────────────────────────────────────────────

FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MA_7', 'MA_21', 'RSI', 'Price_Change']


def prepare_features(df: pd.DataFrame):
    """
    Build X (feature matrix) and y (next-day Close) arrays.
    Returns X, y and the list of dates corresponding to each sample.
    """
    df = df.copy()

    # Target: next trading day's Close
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)          # removes the last row (no target yet)

    X     = df[FEATURE_COLS].values
    y     = df['Target'].values
    dates = df.index

    return X, y, dates


# ─────────────────────────────────────────────
# 4. LINEAR REGRESSION MODEL
# ─────────────────────────────────────────────

def train_linear_regression(df: pd.DataFrame, test_ratio: float = 0.20):
    """
    Train a Linear Regression model and return:
      (y_train, y_pred_train, y_test, y_pred_test, train_dates, test_dates, model, scaler)
    """
    print("\nTraining Linear Regression ...")

    X, y, dates = prepare_features(df)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test   = X[:split],     X[split:]
    y_train, y_test   = y[:split],     y[split:]
    train_dates        = dates[:split]
    test_dates         = dates[split:]

    scaler  = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_tr_sc, y_train)

    y_pred_train = model.predict(X_tr_sc)
    y_pred_test  = model.predict(X_te_sc)

    # ── Metrics ──────────────────────────────
    metrics = {
        'Train RMSE' : np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test  RMSE' : np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        'Test  MAE'  : mean_absolute_error(y_test,  y_pred_test),
        'Test  R²'   : r2_score(y_test, y_pred_test),
    }

    print("\nLinear Regression Results:")
    for k, v in metrics.items():
        unit = '' if 'R²' in k else '$'
        fmt  = '.4f' if 'R²' in k else '.2f'
        print(f"   {k}: {unit}{v:{fmt}}")

    return y_train, y_pred_train, y_test, y_pred_test, train_dates, test_dates, model, scaler, metrics


# ─────────────────────────────────────────────
# 5. VISUALISATION (Part 1)
# ─────────────────────────────────────────────

def visualize_linear_results(df: pd.DataFrame,
                              y_test, y_pred_test,
                              test_dates,
                              metrics: dict,
                              ticker: str):
    """
    4-panel chart:
      [0,0] Actual vs Predicted (test set)
      [0,1] Prediction error distribution
      [1,0] Price history with MA ribbons
      [1,1] RSI panel
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{ticker} — Linear Regression  |  '
                 f'RMSE ${metrics["Test  RMSE"]:.2f}  '
                 f'MAE ${metrics["Test  MAE"]:.2f}  '
                 f'R² {metrics["Test  R²"]:.4f}',
                 fontsize=13, fontweight='bold')

    # ── Panel A: Actual vs Predicted ─────────
    ax = axes[0, 0]
    ax.plot(test_dates, y_test,       label='Actual',    color='steelblue', lw=1.5)
    ax.plot(test_dates, y_pred_test,  label='Predicted', color='tomato',    lw=1.5, alpha=0.85)
    ax.fill_between(test_dates, y_test, y_pred_test, alpha=0.12, color='tomato')
    ax.set_title('Actual vs Predicted (test set)')
    ax.set_ylabel('Price ($)')
    ax.legend(); ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    # ── Panel B: Error distribution ──────────
    ax = axes[0, 1]
    errors = y_test - y_pred_test
    ax.hist(errors, bins=35, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0,          color='red',    lw=1.5, linestyle='--', label='Zero error')
    ax.axvline(errors.mean(), color='orange', lw=1.5, linestyle='--', label=f'Mean {errors.mean():.2f}')
    ax.set_title('Prediction Error Distribution')
    ax.set_xlabel('Error ($)'); ax.set_ylabel('Frequency')
    ax.legend(); ax.grid(alpha=0.25)

    # ── Panel C: Price + Moving Averages ─────
    ax = axes[1, 0]
    tail = df.iloc[-300:]
    ax.plot(tail.index, tail['Close'], label='Close',  color='steelblue', lw=1.2)
    ax.plot(tail.index, tail['MA_7'],  label='MA 7d',  color='orange',    lw=1,   linestyle='--')
    ax.plot(tail.index, tail['MA_21'], label='MA 21d', color='green',     lw=1,   linestyle='--')
    ax.plot(tail.index, tail['MA_50'], label='MA 50d', color='purple',    lw=1,   linestyle='--')
    ax.set_title('Price + Moving Averages (last 300 days)')
    ax.set_ylabel('Price ($)'); ax.legend(); ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # ── Panel D: RSI ─────────────────────────
    ax = axes[1, 1]
    ax.plot(tail.index, tail['RSI'], color='darkorange', lw=1.2)
    ax.axhline(70, color='red',   lw=1, linestyle='--', label='Overbought (70)')
    ax.axhline(30, color='green', lw=1, linestyle='--', label='Oversold (30)')
    ax.fill_between(tail.index, tail['RSI'], 70, where=(tail['RSI'] >= 70),
                    alpha=0.2, color='red')
    ax.fill_between(tail.index, tail['RSI'], 30, where=(tail['RSI'] <= 30),
                    alpha=0.2, color='green')
    ax.set_ylim(0, 100)
    ax.set_title('RSI-14')
    ax.set_ylabel('RSI'); ax.legend(); ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.tight_layout()
    plt.savefig('stock_part1_results.png', dpi=150, bbox_inches='tight')
    print("Chart saved → stock_part1_results.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TICKER     = "AAPL"
    START_DATE = "2020-01-01"

    # 1. Fetch
    df_raw = fetch_stock_data(TICKER, start_date=START_DATE)

    # 2. Enrich
    df = add_technical_indicators(df_raw)

    # 3. Train & evaluate
    y_train, y_pred_train, y_test, y_pred_test, \
        train_dates, test_dates, model, scaler, metrics = train_linear_regression(df)

    # 4. Visualise
    visualize_linear_results(df, y_test, y_pred_test, test_dates, metrics, TICKER)

    print("\n Part 1 complete ")