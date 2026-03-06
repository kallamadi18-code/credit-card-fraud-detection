# Credit Card Fraud Detection Project

## Overview
This project detects fraudulent credit card transactions using machine learning.

## Dataset
- 284,807 transactions
- 492 frauds (0.17% of data)
- 30 features (V1-V28, Time, Amount)

## Models Used
- Logistic Regression (baseline)
- XGBoost (best performer)

## Results (XGBoost)
- Frauds caught: 85 out of 98 (87% detection rate)
- False alarms: 262
- Missed frauds: 13
- Precision: 0.24 (24% of fraud alerts are correct)
- Recall: 0.87 (87% of actual frauds caught)

## Files in this project
- credit_card_fraud_detection.ipynb - Main code notebook
- xgb_fraud_model.pkl - Trained XGBoost model
- scaler.pkl - Scaler for preprocessing
- feature_names.pkl - Column names
- model_results.csv - Performance summary

## How to use
1. Load the model: joblib.load('xgb_fraud_model.pkl')
2. Load scaler: joblib.load('scaler.pkl')
3. Preprocess new data using the scaler
4. Make predictions using the model

## Example
```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('xgb_fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# New transaction data (must have same features)
new_transaction = [[...]]  # Your transaction features
new_transaction_scaled = scaler.transform(new_transaction)
prediction = model.predict(new_transaction_scaled)
probability = model.predict_proba(new_transaction_scaled)
```