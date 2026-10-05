"""
Project 2: Titanic Survival - Preprocessing Pipeline and Cleaned Dataset
-------------------------------------------------------------------------
Steps:
  1. Load raw data and inspect quality (missing values, duplicates, outliers)
  2. Clean the data and engineer features  -> data/titanic_cleaned.csv
  3. Build a reusable scikit-learn preprocessing Pipeline (impute, scale, encode)
  4. Save model-ready data                  -> data/titanic_processed.csv
  5. Save the fitted pipeline               -> outputs/preprocessing_pipeline.joblib
  6. Sanity check: train a quick model on the processed data

Run:  python download_data.py   then   python titanic_preprocessing_pipeline.py
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ---------------- 1. LOAD AND INSPECT ----------------
df = pd.read_csv("data/titanic.csv")
print("Raw shape:", df.shape)
print("\nMissing values (raw):\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---------------- 2. CLEANING ----------------
clean = df.drop_duplicates().copy()

# Standardise column names
clean.columns = [c.strip().lower() for c in clean.columns]

# Feature engineering
clean["title"] = clean["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
clean["title"] = clean["title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir",
     "Jonkheer", "Dona"], "Rare")
clean["title"] = clean["title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
clean["family_size"] = clean["sibsp"] + clean["parch"] + 1
clean["is_alone"] = (clean["family_size"] == 1).astype(int)
clean["deck"] = clean["cabin"].str[0].fillna("Unknown")

# Missing values: Age filled with median age of the passenger's title group
clean["age"] = clean["age"].fillna(clean.groupby("title")["age"].transform("median"))
clean["age"] = clean["age"].fillna(clean["age"].median())
clean["embarked"] = clean["embarked"].fillna(clean["embarked"].mode()[0])
clean["fare"] = clean["fare"].fillna(clean["fare"].median())

# Outliers: cap Fare using the IQR rule (winsorising)
q1, q3 = clean["fare"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print(f"\nFare outliers capped above {upper:.2f}: {(clean['fare'] > upper).sum()} rows")
clean["fare"] = clean["fare"].clip(upper=upper)

# Before/after plot for the fare cleaning
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["Fare"], ax=ax[0]); ax[0].set_title("Fare - before capping")
sns.boxplot(y=clean["fare"], ax=ax[1]); ax[1].set_title("Fare - after capping")
plt.savefig("outputs/fare_before_after.png", dpi=120, bbox_inches="tight")
plt.close()

# Drop identifier / free-text columns
clean = clean.drop(columns=["passengerid", "name", "ticket", "cabin"])
clean.to_csv("data/titanic_cleaned.csv", index=False)
print("\nSaved data/titanic_cleaned.csv", clean.shape)
print("Missing values (cleaned):", int(clean.isnull().sum().sum()))

# ---------------- 3. PREPROCESSING PIPELINE ----------------
target = "survived"
X = clean.drop(columns=[target])
y = clean[target]

num_cols = ["age", "fare", "sibsp", "parch", "family_size"]
cat_cols = ["pclass", "sex", "embarked", "title", "deck", "is_alone"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Fit on TRAIN only to avoid data leakage
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
print("\nProcessed feature count:", len(feature_names))

# ---------------- 4. SAVE MODEL-READY DATA ----------------
processed = pd.DataFrame(preprocessor.transform(X), columns=feature_names)
processed[target] = y.values
processed.to_csv("data/titanic_processed.csv", index=False)
print("Saved data/titanic_processed.csv", processed.shape)

# ---------------- 5. SAVE PIPELINE ----------------
joblib.dump(preprocessor, "outputs/preprocessing_pipeline.joblib")
print("Saved outputs/preprocessing_pipeline.joblib")

# ---------------- 6. SANITY CHECK ----------------
model = LogisticRegression(max_iter=1000).fit(X_train_p, y_train)
acc = accuracy_score(y_test, model.predict(X_test_p))
print(f"\nSanity check - Logistic Regression accuracy: {acc:.4f}")
