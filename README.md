# Gold Quantitative Trading Model (XAU/USD)

## Overview
This project presents a quantitative trading model for XAU/USD, integrating high-frequency market data with macroeconomic events to generate data-driven trading signals.

The model combines technical indicators and macroeconomic features to capture short-term price dynamics and improve intraday trading decision-making.

## Problem Statement
Intraday trading in gold (XAU/USD) is highly sensitive to market noise and macroeconomic announcements, making it difficult to identify high-probability trade setups and manage risk objectively.

## Methodology
- Integration of OHLCV market data (5-minute frequency)
- Incorporation of macroeconomic calendar events (FED, inflation, employment)
- Feature engineering:
  - Technical indicators: returns, RSI, ATR, volatility, moving averages
  - Macroeconomic features: event timing, impact level, event type
- Model development:
  - Logistic Regression
  - Random Forest
- Signal generation based on predicted probabilities
- Backtesting framework to evaluate strategy performance

## Technologies
- Python (Pandas, NumPy, Scikit-learn)
- Time series data analysis
- Machine learning for financial markets

## Objective
To build a quantitative framework capable of identifying trading opportunities in gold markets using a combination of technical and macroeconomic signals.
