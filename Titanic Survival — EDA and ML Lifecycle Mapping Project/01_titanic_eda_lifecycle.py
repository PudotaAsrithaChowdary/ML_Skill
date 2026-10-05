# 1. Titanic Survival — EDA and ML Lifecycle Mapping
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/titanic_train.csv")

print("\n=== DATASET SHAPE ===")
print(df.shape)
print("\n=== DATA TYPES ===")
print(df.dtypes)
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())
print("\n=== SUMMARY ===")
print(df.describe(include="all"))

# EDA
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=df, x="Survived", ax=ax[0])
ax[0].set_title("Survival Distribution")
sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax[1])
ax[1].set_title("Survival by Passenger Class")
plt.tight_layout()
plt.show()

print("\n=== KEY EDA ===")
print(df.groupby("Sex")["Survived"].mean())
print(df.groupby("Pclass")["Survived"].mean())

print("""
ML LIFECYCLE:
1. Problem definition -> predict passenger survival.
2. Data collection -> Titanic train.csv.
3. EDA -> shape, types, missing values, distributions.
4. Preprocessing -> impute missing values, encode categories.
5. Feature engineering -> FamilySize, IsAlone, Title.
6. Train/test split.
7. Model training -> classification algorithm.
8. Evaluation -> accuracy, precision, recall, F1, confusion matrix.
9. Interpretation -> identify important predictors.
10. Deployment/monitoring -> accept new passenger records and monitor performance.
""")
