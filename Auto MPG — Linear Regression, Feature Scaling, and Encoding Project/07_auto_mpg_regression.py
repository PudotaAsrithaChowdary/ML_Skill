# 7. Auto MPG — Linear Regression, Feature Scaling, and Encoding
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

cols = ["mpg","cylinders","displacement","horsepower","weight",
        "acceleration","model_year","origin","car_name"]

df = pd.read_csv(
    "datasets/auto-mpg.data",
    sep=r"\s+",
    names=cols,
    na_values="?",
    engine="python"
)

# Remove car_name because it is an identifier/text description.
df = df.drop(columns=["car_name"])
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

X = df.drop(columns=["mpg"])
y = df["mpg"]

num_cols = ["cylinders","displacement","horsepower","weight",
            "acceleration","model_year"]
cat_cols = ["origin"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R2  :", r2_score(y_test, pred))
