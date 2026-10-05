# 11. Wine Quality — Decision Tree and Random Forest Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Use UCI red wine file. Change to winequality-white.csv if desired.
df = pd.read_csv("datasets/winequality-red.csv", sep=";")

# Convert ordered quality score to binary classification:
# 0 = quality < 7, 1 = quality >= 7
df["good_wine"] = (df["quality"] >= 7).astype(int)

X = df.drop(columns=["quality","good_wine"])
y = df["good_wine"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
forest = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

for name, model in [("Decision Tree", tree), ("Random Forest", forest)]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

importance = pd.Series(
    forest.feature_importances_, index=X.columns
).sort_values()
importance.plot(kind="barh", figsize=(9,6), title="Wine Quality Feature Importance")
plt.tight_layout()
plt.show()
