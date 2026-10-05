# 3. Adult Income — Feature Engineering and EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cols = [
    "age","workclass","fnlwgt","education","education_num","marital_status",
    "occupation","relationship","race","sex","capital_gain","capital_loss",
    "hours_per_week","native_country","income"
]

df = pd.read_csv("datasets/adult.csv", names=cols, skipinitialspace=True, na_values="?")
df["income"] = df["income"].str.replace(".", "", regex=False).str.strip()

# Feature engineering
df["capital_net"] = df["capital_gain"] - df["capital_loss"]
df["is_married"] = df["marital_status"].str.contains("Married", na=False).astype(int)
df["age_group"] = pd.cut(
    df["age"], bins=[0,25,35,45,55,65,100],
    labels=["<=25","26-35","36-45","46-55","56-65","66+"]
)
df["hours_group"] = pd.cut(
    df["hours_per_week"], bins=[0,20,40,60,100],
    labels=["Part-time","Standard","Overtime","Extreme"]
)

print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nIncome distribution:\n", df["income"].value_counts())

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(data=df, x="age", hue="income", kde=True, ax=ax[0], element="step")
ax[0].set_title("Age by Income")
sns.countplot(data=df, x="education", hue="income", ax=ax[1])
ax[1].tick_params(axis="x", rotation=70)
ax[1].set_title("Education by Income")
plt.tight_layout()
plt.show()

print("\nAverage hours by income:")
print(df.groupby("income")["hours_per_week"].mean())
print("\nIncome rate by age group:")
print(pd.crosstab(df["age_group"], df["income"], normalize="index"))
