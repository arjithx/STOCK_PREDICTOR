# STOCK_PREDICTOR
# 📈 AI-Powered Stock Price Prediction & Forecasting System

###  Bridging Machine Learning & Deep Learning for Financial Intelligence

---

##  Overview

This project presents a comprehensive stock price prediction system using both **Machine Learning (ML)** and **Deep Learning (DL)** techniques.

It combines:
-  Technical Indicators (RSI, Moving Averages)
-  Linear Regression (Baseline Model)
- LSTM Neural Networks (Time-Series Forecasting)
-  Future Price Prediction (30 Days Ahead)

The goal is to analyze historical stock data and generate accurate predictions along with insightful visualizations.

---

##  Key Features

- Dual-model architecture (ML + DL)
- Advanced feature engineering
- Time-series aware data handling
- Future forecasting with uncertainty band
- Multi-panel visualization dashboards

---

##  Dataset

- Source: Yahoo Finance (`yfinance`)
- Stock: **AAPL (Apple Inc.)**
- Time Range: **2020 – Present**

---

##  Methodology

###  1. Data Collection
- Historical stock data fetched using `yfinance`
- Cleaned and structured into time-series format

---

###  2. Feature Engineering

The following indicators were added:
- Moving Averages (7, 21, 50 days)
- Relative Strength Index (RSI-14)
- Price Change (%)
- Volume Moving Average

---

###  3. Model 1 — Linear Regression

A supervised ML model trained on engineered features.

####  Results:
- RMSE: **4.35**
- MAE: **2.91**
- R² Score: **0.9741**

High accuracy and strong baseline performance

---

###  4. Model 2 — LSTM Deep Learning

A stacked LSTM network for sequential data modeling.

####  Architecture:
- LSTM (128 → 64 → 32)
- Dropout layers
- Dense layers

####  Results:
- RMSE: **10.21**
- MAE: **7.68**
- R² Score: **0.8620**

 Captures time dependencies but requires tuning

---

##  Future Forecasting

- Predicts next **30 trading days**
- Uses recursive forecasting
- Includes uncertainty band (±RMSE)

---

##  Visualizations

###  Linear Regression
- Actual vs Predicted Prices
- Error Distribution
- Moving Averages
- RSI Indicator

###  LSTM Model
- Prediction vs Actual
- Training & Validation Loss
- Full Historical Comparison
- Future Forecast

---

##  Model Comparison

| Model              | RMSE  | MAE  | R² Score |
|-------------------|------|------|----------|
| Linear Regression | 4.35 | 2.91 | 0.9741   |
| LSTM              | 10.21| 7.68 | 0.8620   |

Insight: Simpler models can outperform deep learning with strong features.

---

---

## 🛠️ Tech Stack

- Python
- NumPy & Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- yFinance API

---

## Key Learnings

- Feature engineering is critical
- Time-series data requires careful handling
- LSTM models need tuning
- Visualization improves understanding
- Simpler models can be powerful

---

##  Future Improvements

- Hyperparameter tuning
- Sentiment analysis integration
- Transformer-based models
- Streamlit deployment
- Multi-stock analysis

---

##  Conclusion

This project demonstrates a practical implementation of AI in financial forecasting by comparing Machine Learning and Deep Learning approaches on real-world stock data.

---

##  Author

**Arjith Velusamy**  
AI & Data Science Student  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and connect with me on LinkedIn!

