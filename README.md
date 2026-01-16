# Advanced-ML-Project

# Learning-to-Rank for Cross-Sectional Systematic Strategies

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📌 Overview

This project explores the application of **Learning-to-Rank (LTR)** algorithms to construct market-neutral cross-sectional equity portfolios. Unlike traditional "Regress-then-Rank" approaches that minimize prediction error on returns, LTR models (like LambdaMART and ListNet) are optimized to directly order assets based on their expected relative performance.

The study compares heuristic baselines, linear models, and tree-based regressors against pairwise and listwise ranking algorithms on the **S&P 500** universe over a 25-year period.

## 🚀 Key Features

* **Robust Data Pipeline:** Sourcing from Yahoo Finance, with strictly point-in-time feature engineering to prevent look-ahead bias.
* **Feature Engineering:** Implementation of linear (MA, Z-Score), non-linear (Volatility, Skewness, Kurtosis), and microstructure features (Dollar Volume, Turnover).
* **Advanced Validation:** strictly time-ordered train/test split (80/20) with an **expanding-window cross-validation** scheme for hyperparameter tuning.
* **Diverse Model Zoo:**
    * **Heuristics:** Random, Momentum, MACD.
    * **Supervised Learning:** OLS, OLS+PCA, LightGBM Regressor.
    * **Learning-to-Rank:** LambdaMART (Pairwise), ListNet (Listwise/Neural Network).
* **Portfolio Construction:** Long/Short (Top 35 / Bottom 35), volatility-scaled weighting scheme.

## 📂 Repository Structure

```bash
├── notebook/              # Main Jupyter notebook contains all used methods analysis and visualization
├── notebook_listwise/     # Jupyter notebook used for analysis and visualization of listNet ranking methodology
└── requirements           # A folder data to dump yfinance downloaded data
