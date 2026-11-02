
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# Create models folder
os.makedirs("models", exist_ok=True)

# Generate sample banking dataset
np.random.seed(42)

n = 2000

df = pd.DataFrame({
    "age": np.random.randint(21, 61, n),
    "annual_income": np.random.randint(150000, 1500000, n),
    "credit_score": np.random.randint(300, 851, n),
    "employment_type": np.random.choice(
        ["Salaried", "Self Employed", "Business"], n
    ),
    "existing_loan": np.random.randint(0, 500000, n),
    "loan_amount_requested": np.random.randint(50000, 1000000, n)
})

# Generate loan approval target
df["loan_approval"] = (
    (df["credit_score"] >= 600) &
    (df["annual_income"] >= 250000) &
    (df["loan_amount_requested"] <= df["annual_income"] * 0.8) &
    (df["existing_loan"] < df["annual_income"] * 1.5)
).astype(int)

# Generate recommended loan amount
df["recommended_loan_amount"] = (
    df["annual_income"] * 0.6
    + (df["credit_score"] - 300) * 500
    - df["existing_loan"] * 0.3
).clip(lower=0, upper=1500000)

# Encode categorical feature
df = pd.get_dummies(
    df,
    columns=["employment_type"],
    dtype=int
)

# Features
X = df.drop(columns=[
    "loan_approval",
    "recommended_loan_amount"
])

# Classification target
y_class = df["loan_approval"]

# Regression target
y_reg = df["recommended_loan_amount"]

# Split classification data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

# Decision Tree Classifier
classifier = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    random_state=42
)

classifier.fit(X_train, y_train)

# Classification evaluation
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Split regression data
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg,
    test_size=0.2,
    random_state=42
)

# Decision Tree Regressor
regressor = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=6,
    random_state=42
)

regressor.fit(X_train_r, y_train_r)

# Regression evaluation
reg_pred = regressor.predict(X_test_r)

mae = mean_absolute_error(y_test_r, reg_pred)
r2 = r2_score(y_test_r, reg_pred)

# Save models and feature names
joblib.dump(classifier, "models/loan_classifier.pkl")
joblib.dump(regressor, "models/loan_regressor.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

# Display results
print("\n================================")
print("BANKING LOAN PREDICTION SYSTEM")
print("================================")

print("Algorithm: Decision Tree")

print("\nClassification Results")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nRegression Results")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

print("\nModels saved successfully!")
print("================================")
