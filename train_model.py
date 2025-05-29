import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df[df['TotalCharges'].str.strip() != '']
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get top 20 features
importances = model.feature_importances_
indices = importances.argsort()[-20:][::-1]  # Indices of top 20 features
top_features = X.columns[indices].tolist()

# Retrain model using only top 20 features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X[top_features], y)
model.feature_names_ = top_features  # Store the selected features

joblib.dump(model, "model.pkl")
print(f"Saved model with top 20 features: {top_features}")