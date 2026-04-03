import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv("/Users/monishtammineni/Downloads/synthetic_customer_behavior_and_churn.csv")
print("Data loaded")

# Same preprocessing
df = df.drop(['customer_id', 'signup_date'], axis=1)
df = df.dropna()
df = df.drop_duplicates()
df = pd.get_dummies(df)

# Split data
X = df.drop('churn', axis=1)
y = df['churn']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Data split into train and test")

# Create models
model1 = LogisticRegression(max_iter=1000)
model2 = RandomForestClassifier()
model3 = xgb.XGBClassifier()

# Train models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Predictions
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

# Accuracy
acc1 = accuracy_score(y_test, pred1)
acc2 = accuracy_score(y_test, pred2)
acc3 = accuracy_score(y_test, pred3)

print("Logistic Accuracy:", acc1)
print("Random Forest Accuracy:", acc2)
print("XGBoost Accuracy:", acc3)

# Choose best model
best_model = model3  # usually XGBoost is best

# Save model
joblib.dump(best_model, "model.pkl")

print("Model saved as model.pkl")