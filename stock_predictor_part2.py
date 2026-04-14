# ============================================================
# STOCK PREDICTOR — PART 2
# LSTM Deep Learning + Future Price Forecasting
# Depends on: stock_predictor_part1.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Re-use helpers from Part 1
from stock_predictor_part1 import (
    fetch_stock_data,
    add_technical_indicators,
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras (graceful import error)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for Part 2.\n"
        "Install it with:  pip install tensorflow\n"
        f"Original error: {e}"
    )


# ─────────────────────────────────────────────
# 1. SEQUENCE BUILDER
# ─────────────────────────────────────────────

def build_sequences(series: np.ndarray, lookback: int = 60):
    """
    Convert a 1-D price series into overlapping (X, y) sequences.

    Parameters
    ----------
    series   : 1-D numpy array of scaled prices
    lookback : how many past days to look at

    Returns
    -------
    X : (n_samples, lookback, 1)
    y : (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(series) - 1):
        X.append(series[i - lookback: i])
        y.append(series[i])
    return np.array(X)[..., np.newaxis], np.array(y)


# ─────────────────────────────────────────────
# 2. BUILD LSTM MODEL
# ─────────────────────────────────────────────

def build_lstm_model(lookback: int = 60) -> Sequential:
    """
    Stacked LSTM with dropout regularisation.
    Architecture: LSTM(128) → LSTM(64) → LSTM(32) → Dense(16) → Dense(1)
    """
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.25),
        LSTM(64, return_sequences=True),
        Dropout(0.20),
        LSTM(32, return_sequences=False),
        Dropout(0.15),
        Dense(16, activation='relu'),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae'],
    )
    return model


# ─────────────────────────────────────────────
# 3. TRAIN LSTM
# ─────────────────────────────────────────────

def train_lstm(df: pd.DataFrame,
               lookback: int = 60,
               test_ratio: float = 0.20,
               epochs: int = 80,
               batch_size: int = 32):
    """
    Full LSTM training pipeline.

    Returns
    -------
    y_test_real, y_pred_real : inverse-transformed price arrays
    test_dates               : DatetimeIndex aligned to y_test
    model, scaler            : for reuse in forecasting
    metrics                  : dict of evaluation scores
    history                  : Keras History object
    """
    print("\nTraining LSTM model …")

    close_vals = df['Close'].values.reshape(-1, 1)
    dates      = df.index

    # Scale to [0, 1]
    scaler      = MinMaxScaler()
    scaled      = scaler.fit_transform(close_vals).flatten()

    # Sequences
    X, y = build_sequences(scaled, lookback=lookback)

    # Train / test split (no shuffling — time-series!)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Dates aligned to test targets
    # Each target y[i] corresponds to dates[lookback + i]
    test_dates = dates[lookback + split: lookback + split + len(y_test)]

    # ── Build & train ─────────────────────────
    model = build_lstm_model(lookback=lookback)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=6, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.10,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Predict ───────────────────────────────
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # ── Metrics ───────────────────────────────
    metrics = {
        'Test  RMSE' : np.sqrt(mean_squared_error(y_test_real, y_pred_real)),
        'Test  MAE'  : mean_absolute_error(y_test_real, y_pred_real),
        'Test  R²'   : r2_score(y_test_real, y_pred_real),
    }

    print("\nLSTM Results:")
    for k, v in metrics.items():
        unit = '' if 'R²' in k else '$'
        fmt  = '.4f' if 'R²' in k else '.2f'
        print(f"   {k}: {unit}{v:{fmt}}")

    return y_test_real, y_pred_real, test_dates, model, scaler, metrics, history


# ─────────────────────────────────────────────
# 4. FUTURE FORECASTING
# ─────────────────────────────────────────────

def forecast_future(model, scaler, df: pd.DataFrame,
                    lookback: int = 60,
                    forecast_days: int = 30) -> tuple:
    """
    Roll the model forward for `forecast_days` into the future.

    Returns
    -------
    future_dates  : DatetimeIndex
    future_prices : numpy array of predicted prices
    """
    print(f"\nForecasting {forecast_days} trading days ahead …")

    last_close  = df['Close'].values[-lookback:].reshape(-1, 1)
    last_scaled = scaler.transform(last_close).flatten()

    predictions  = []
    current_seq  = last_scaled.copy()

    for _ in range(forecast_days):
        x_input  = current_seq[-lookback:].reshape(1, lookback, 1)
        next_val = model.predict(x_input, verbose=0)[0, 0]
        predictions.append(next_val)
        current_seq = np.append(current_seq, next_val)

    future_prices = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    # Generate business-day dates
    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1),
                                  periods=forecast_days)

    print(f"   Last known close  : ${df['Close'].iloc[-1]:.2f}")
    print(f"   Day-1 forecast    : ${future_prices[0]:.2f}")
    print(f"   Day-{forecast_days} forecast   : ${future_prices[-1]:.2f}")

    return future_dates, future_prices


# ─────────────────────────────────────────────
# 5. VISUALISATION (Part 2)
# ─────────────────────────────────────────────

def visualize_lstm_results(df: pd.DataFrame,
                            y_test, y_pred,
                            test_dates,
                            future_dates, future_prices,
                            metrics: dict,
                            history,
                            ticker: str):
    """
    4-panel chart:
      [0,0] Actual vs Predicted (test set)
      [0,1] Training & validation loss curves
      [1,0] Full history + test predictions in context
      [1,1] Historical tail + future forecast ribbon
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f'{ticker} — LSTM Deep Learning  |  '
        f'RMSE ${metrics["Test  RMSE"]:.2f}  '
        f'MAE ${metrics["Test  MAE"]:.2f}  '
        f'R² {metrics["Test  R²"]:.4f}',
        fontsize=13, fontweight='bold'
    )

    # ── Panel A: Actual vs Predicted ─────────
    ax = axes[0, 0]
    ax.plot(test_dates, y_test, label='Actual',    color='steelblue', lw=1.5)
    ax.plot(test_dates, y_pred, label='Predicted', color='tomato',    lw=1.5, alpha=0.85)
    ax.fill_between(test_dates, y_test, y_pred, alpha=0.12, color='tomato')
    ax.set_title('Actual vs Predicted (test set)')
    ax.set_ylabel('Price ($)'); ax.legend(); ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()

    # ── Panel B: Training curves ─────────────
    ax = axes[0, 1]
    ax.plot(history.history['loss'],     label='Train loss', color='steelblue', lw=1.2)
    ax.plot(history.history['val_loss'], label='Val loss',   color='tomato',    lw=1.2)
    ax.set_title('Training & Validation Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.legend(); ax.grid(alpha=0.25)

    # ── Panel C: Full series + overlaid test predictions ─
    ax = axes[1, 0]
    ax.plot(df.index, df['Close'],    label='Historical Close', color='steelblue', lw=1.0, alpha=0.7)
    ax.plot(test_dates, y_pred,       label='LSTM (test)',      color='tomato',    lw=1.4)
    ax.set_title('Full Price History with Test Predictions')
    ax.set_ylabel('Price ($)'); ax.legend(); ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ── Panel D: Future forecast ─────────────
    ax = axes[1, 1]
    tail_n    = 100
    hist_tail = df['Close'].iloc[-tail_n:]
    ax.plot(hist_tail.index, hist_tail.values,
            label='Historical (last 100d)', color='steelblue', lw=1.5)
    ax.plot(future_dates, future_prices,
            label=f'Forecast ({len(future_dates)}d)',
            color='tomato', lw=1.5, linestyle='--', marker='o', markersize=3)

    # Uncertainty ribbon: ±1 RMSE
    rmse = metrics['Test  RMSE']
    ax.fill_between(future_dates,
                    future_prices - rmse, future_prices + rmse,
                    alpha=0.15, color='tomato', label='±RMSE band')
    ax.axvline(df.index[-1], color='gray', lw=1, linestyle=':')
    ax.set_title('Future Price Forecast')
    ax.set_ylabel('Price ($)'); ax.legend(); ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.tight_layout()
    plt.savefig('stock_part2_results.png', dpi=150, bbox_inches='tight')
    print("Chart saved → stock_part2_results.png")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TICKER        = "AAPL"
    START_DATE    = "2020-01-01"
    LOOKBACK      = 60
    FORECAST_DAYS = 30

    # 1. Fetch & enrich (reuse Part 1 helpers)
    df_raw = fetch_stock_data(TICKER, start_date=START_DATE)
    df     = add_technical_indicators(df_raw)

    # 2. Train LSTM
    y_test, y_pred, test_dates, \
        model, scaler, metrics, history = train_lstm(df, lookback=LOOKBACK)

    # 3. Forecast future prices
    future_dates, future_prices = forecast_future(
        model, scaler, df,
        lookback=LOOKBACK,
        forecast_days=FORECAST_DAYS
    )

    # 4. Visualise
    visualize_lstm_results(
        df, y_test, y_pred, test_dates,
        future_dates, future_prices,
        metrics, history, TICKER
    )

    print("\nPart 2 complete.")
    print(f"   → stock_part1_results.png  (Linear Regression charts)")
    print(f"   → stock_part2_results.png  (LSTM + forecast charts)")