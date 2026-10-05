# 9. Heart Disease — Random Forest Ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

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

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", figsize=(9,6), title="Random Forest Feature Importance")
plt.tight_layout()
plt.show()
