import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

print("Model loaded")

# Example: take one sample from dataset
df = pd.read_csv("/Users/monishtammineni/Downloads/synthetic_customer_behavior_and_churn.csv")

# Same preprocessing
df = df.drop(['customer_id', 'signup_date'], axis=1)
df = df.dropna()
df = df.drop_duplicates()
df = pd.get_dummies(df)

# Take one row
sample = df.drop('churn', axis=1).iloc[0:1]

# Predict
prediction = model.predict(sample)

print("Prediction (0 = No churn, 1 = Churn):", prediction[0])