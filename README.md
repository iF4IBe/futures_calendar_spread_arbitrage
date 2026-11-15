# Probabilistic Fusion of Statistical Arbitrage and Machine Learning for Rebar Futures Calendar Spread

This repository contains the source code, results, and the full paper for my research on calendar-spread arbitrage in Shanghai Futures Exchange (SHFE) Rebar (RB) futures.

Traditional calendar-spread arbitrage relies on the Z-score of cointegration residuals. While effective in stationary regimes, fixed-threshold rules generate false signals when the rolling equilibrium shifts. This project builds a hybrid framework that combines the statistical Z-score with Machine Learning (ML) directional probabilities. 

Instead of a black-box ensemble, we map both signals to a [0,1] probability scale (via Normal CDF for Z-score, and `predict_proba` for ML) and use a weighted linear fusion `P_combined = w * P_stat + (1-w) * P_ml`. The weight `w` and the trading threshold are optimized via rolling time-series cross-validation. 

Three ML models are tested: Logistic Regression (LR), XGBoost, and Random Forest (RF). All backtesting metrics are calculated on a net-of-cost basis using the exact SHFE 2025 fee and slippage model.
