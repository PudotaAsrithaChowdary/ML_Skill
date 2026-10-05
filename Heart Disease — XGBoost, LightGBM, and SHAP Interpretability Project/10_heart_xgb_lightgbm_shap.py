# 10. Heart Disease — XGBoost, LightGBM, and SHAP Interpretability
# Install first:
# pip install pandas scikit-learn xgboost lightgbm shap matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

df = pd.read_csv("datasets/heart.csv")
if "target" not in df.columns:
    df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
                  "exang","oldpeak","slope","ca","thal","target"]

df = df.replace("?", pd.NA).dropna()
X = df.drop(columns=["target"]).apply(pd.to_numeric)
y = (pd.to_numeric(df["target"]) > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        num_leaves=31, random_state=42, verbose=-1
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

# SHAP explanation for XGBoost
xgb = models["XGBoost"]
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.show()
