import pandas as pd

# Load data
df = pd.read_csv("/Users/monishtammineni/Downloads/synthetic_customer_behavior_and_churn.csv")
print("Data loaded")

# Remove columns we don't need
df = df.drop(['customer_id', 'signup_date'], axis=1)
print("Removed unnecessary columns")

# Clean data
df = df.dropna()
df = df.drop_duplicates()
print("Cleaned data")

# Convert text to numbers
df = pd.get_dummies(df)
print("Converted text to numbers")

# Split into input and output
X = df.drop('churn', axis=1)
y = df['churn']

print("Preprocessing done")
print("X shape:", X.shape)
print("y shape:", y.shape)