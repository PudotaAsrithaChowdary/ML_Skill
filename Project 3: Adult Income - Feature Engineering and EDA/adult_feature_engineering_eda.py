"""
Project 3: Adult Income - Feature Engineering and EDA
------------------------------------------------------
Goal: understand what drives income (>50K vs <=50K) and build better features.

Steps:
  1. Load and clean (strip spaces, '?' -> NaN, binary target)
  2. EDA: distributions, income vs key features, correlation
  3. Feature engineering: age groups, capital net, hours category, education
     level, marital simplification, region grouping
  4. Encoding and feature ranking (mutual information)
  5. Save engineered dataset -> data/adult_engineered.csv

Run:  python download_data.py   then   python adult_feature_engineering_eda.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
os.makedirs("outputs", exist_ok=True)

# ---------------- 1. LOAD AND CLEAN ----------------
df = pd.read_csv("data/adult.csv")
print("Shape:", df.shape)

for col in df.select_dtypes(include=["object", "category"]).columns:
    df[col] = df[col].astype(str).str.strip()
df = df.replace("?", np.nan).replace("nan", np.nan)
df["income"] = df["income"].str.replace(".", "", regex=False)
df["income_binary"] = (df["income"] == ">50K").astype(int)

print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

for col in ["workclass", "occupation", "native_country"]:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nIncome class balance:\n", df["income"].value_counts(normalize=True).round(3))

# ---------------- 2. EDA ----------------
print("\n", df.describe().T)

# Income distribution
plt.figure(figsize=(5, 4))
sns.countplot(x="income", data=df)
plt.title("Income Distribution")
plt.savefig("outputs/income_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# Numeric distributions
df[["age", "education_num", "hours_per_week", "capital_gain"]].hist(
    bins=30, figsize=(10, 7))
plt.suptitle("Numeric Feature Distributions")
plt.savefig("outputs/numeric_distributions.png", dpi=120, bbox_inches="tight")
plt.close()

# Income vs categorical features
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, col in zip(axes.ravel(), ["sex", "race", "marital_status", "workclass"]):
    rate = df.groupby(col)["income_binary"].mean().sort_values(ascending=False)
    sns.barplot(x=rate.values, y=rate.index, ax=ax)
    ax.set_title(f"Share earning >50K by {col}")
    ax.set_xlabel("Proportion >50K")
plt.tight_layout()
plt.savefig("outputs/income_by_category.png", dpi=120, bbox_inches="tight")
plt.close()

# Age and hours vs income
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x="income", y="age", data=df, ax=ax[0])
sns.boxplot(x="income", y="hours_per_week", data=df, ax=ax[1])
plt.savefig("outputs/age_hours_vs_income.png", dpi=120, bbox_inches="tight")
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------- 3. FEATURE ENGINEERING ----------------
fe = df.copy()

fe["age_group"] = pd.cut(fe["age"], bins=[0, 25, 35, 45, 55, 65, 100],
                         labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"])
fe["capital_net"] = fe["capital_gain"] - fe["capital_loss"]
fe["has_capital_gain"] = (fe["capital_gain"] > 0).astype(int)
fe["has_capital_loss"] = (fe["capital_loss"] > 0).astype(int)
fe["hours_category"] = pd.cut(fe["hours_per_week"], bins=[0, 34, 40, 60, 100],
                              labels=["part_time", "full_time", "overtime", "extreme"])
fe["education_level"] = pd.cut(fe["education_num"], bins=[0, 8, 9, 12, 13, 16],
                               labels=["school", "hs_grad", "some_college",
                                       "bachelors", "postgrad"])
fe["is_married"] = fe["marital_status"].isin(
    ["Married-civ-spouse", "Married-AF-spouse"]).astype(int)
fe["is_us"] = (fe["native_country"] == "United-States").astype(int)
fe["log_capital_gain"] = np.log1p(fe["capital_gain"])
fe["log_fnlwgt"] = np.log1p(fe["fnlwgt"])

# Engineered features vs income
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.barplot(x="age_group", y="income_binary", data=fe, ax=ax[0])
ax[0].set_title("Share >50K by Age Group")
sns.barplot(x="education_level", y="income_binary", data=fe, ax=ax[1])
ax[1].set_title("Share >50K by Education Level")
plt.savefig("outputs/engineered_features_vs_income.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------- 4. ENCODING AND FEATURE RANKING ----------------
drop_cols = ["income", "education", "fnlwgt", "capital_gain"]
model_df = fe.drop(columns=drop_cols)
y = model_df.pop("income_binary")
X = pd.get_dummies(model_df, drop_first=True).astype(float)
print("\nEncoded feature matrix:", X.shape)

mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print("\nTop 15 features by mutual information:\n", mi_series.head(15).round(4))

plt.figure(figsize=(8, 6))
mi_series.head(15).plot(kind="barh").invert_yaxis()
plt.title("Top 15 Features (Mutual Information with Income)")
plt.savefig("outputs/feature_ranking.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------- 5. SAVE ----------------
out = X.copy()
out["income_binary"] = y.values
out.to_csv("data/adult_engineered.csv", index=False)
fe.to_csv("data/adult_cleaned_with_features.csv", index=False)
print("\nSaved data/adult_engineered.csv and data/adult_cleaned_with_features.csv")
