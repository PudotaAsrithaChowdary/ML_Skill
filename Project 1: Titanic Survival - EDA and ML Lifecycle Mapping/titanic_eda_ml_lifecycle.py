"""
Project 1: Titanic Survival - EDA and ML Lifecycle Mapping
-----------------------------------------------------------
Maps every stage of the ML lifecycle onto the Titanic dataset:
  1. Problem definition
  2. Data collection
  3. Data understanding (EDA)
  4. Data preparation
  5. Modelling
  6. Evaluation
  7. Deployment / next steps

Dataset: data/titanic.csv  (Kaggle "Titanic - Machine Learning from Disaster", train.csv
renamed to titanic.csv). Columns: PassengerId, Survived, Pclass, Name, Sex, Age,
SibSp, Parch, Ticket, Fare, Cabin, Embarked

Run:  python titanic_eda_ml_lifecycle.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_PATH = "data/titanic.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# STAGE 1: PROBLEM DEFINITION
# ---------------------------------------------------------------
print("STAGE 1: Problem definition")
print("Goal: predict whether a passenger survived (1) or not (0).")
print("Type: supervised, binary classification. Metric: accuracy, F1.\n")

# ---------------------------------------------------------------
# STAGE 2: DATA COLLECTION
# ---------------------------------------------------------------
print("STAGE 2: Data collection")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {DATA_PATH}\n")

# ---------------------------------------------------------------
# STAGE 3: DATA UNDERSTANDING (EDA)
# ---------------------------------------------------------------
print("STAGE 3: Data understanding (EDA)")
print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe(include="all").T, "\n")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

print("Survival rate overall: {:.2%}".format(df["Survived"].mean()))
print("Survival rate by Sex:\n", df.groupby("Sex")["Survived"].mean(), "\n")
print("Survival rate by Pclass:\n", df.groupby("Pclass")["Survived"].mean(), "\n")

# Plot 1: survival count
plt.figure(figsize=(5, 4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.savefig(f"{OUT_DIR}/survival_count.png", dpi=120, bbox_inches="tight")
plt.close()

# Plot 2: survival by sex and class
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x="Sex", y="Survived", data=df, ax=ax[0])
ax[0].set_title("Survival Rate by Sex")
sns.barplot(x="Pclass", y="Survived", data=df, ax=ax[1])
ax[1].set_title("Survival Rate by Passenger Class")
plt.savefig(f"{OUT_DIR}/survival_by_sex_class.png", dpi=120, bbox_inches="tight")
plt.close()

# Plot 3: age distribution by survival
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Age", hue="Survived", kde=True, bins=30)
plt.title("Age Distribution by Survival")
plt.savefig(f"{OUT_DIR}/age_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# Plot 4: correlation heatmap (numeric columns)
plt.figure(figsize=(7, 5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# STAGE 4: DATA PREPARATION
# ---------------------------------------------------------------
print("\nSTAGE 4: Data preparation")
data = df.copy()

# Handle missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Feature engineering
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
data["Title"] = data["Title"].replace(
    ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir",
     "Jonkheer", "Dona"], "Rare")
data["Title"] = data["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Drop columns that are not useful for modelling
data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Encode categorical variables
data = pd.get_dummies(data, columns=["Sex", "Embarked", "Title"], drop_first=True)
print("Prepared dataset shape:", data.shape)

X = data.drop("Survived", axis=1)
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# ---------------------------------------------------------------
# STAGE 5: MODELLING
# ---------------------------------------------------------------
print("STAGE 5: Modelling")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = model
    print(f"Trained: {name}")

# ---------------------------------------------------------------
# STAGE 6: EVALUATION
# ---------------------------------------------------------------
print("\nSTAGE 6: Evaluation")
for name, model in results.items():
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"{OUT_DIR}/confusion_{fname}.png", dpi=120, bbox_inches="tight")
    plt.close()

# Feature importance from Random Forest
rf = results["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(7, 5))
imp.head(10).plot(kind="barh").invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# STAGE 7: DEPLOYMENT / NEXT STEPS
# ---------------------------------------------------------------
print("\nSTAGE 7: Deployment / next steps")
print("- Save the best model with joblib and serve it via Flask/FastAPI/Streamlit")
print("- Monitor model drift and retrain on new data")
print("- Try hyperparameter tuning (GridSearchCV) and more models (XGBoost)")
print(f"\nAll plots saved in '{OUT_DIR}/'")
