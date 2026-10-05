# 2. Titanic Survival — Preprocessing Pipeline and Cleaned Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("datasets/titanic_train.csv")

# Feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Keep useful columns
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "Embarked", "FamilySize", "IsAlone"]]
y = df["Survived"]

num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
cat_cols = ["Sex", "Embarked"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
clean_train = pd.DataFrame(X_train_clean, columns=feature_names)
clean_train["Survived"] = y_train.reset_index(drop=True)

clean_test = pd.DataFrame(X_test_clean, columns=feature_names)
clean_test["Survived"] = y_test.reset_index(drop=True)

clean_train.to_csv("datasets/titanic_clean_train.csv", index=False)
clean_test.to_csv("datasets/titanic_clean_test.csv", index=False)

print("Saved:")
print("datasets/titanic_clean_train.csv")
print("datasets/titanic_clean_test.csv")
print(clean_train.head())
