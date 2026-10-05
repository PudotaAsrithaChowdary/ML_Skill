# 8. Heart Disease — Decision Tree Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]

df = pd.read_csv("datasets/heart.csv")

# If your downloaded file uses UCI's processed naming, rename the target.
if "target" not in df.columns:
    df.columns = cols

df = df.replace("?", pd.NA).dropna()

X = df.drop(columns=["target"]).apply(pd.to_numeric)
y = pd.to_numeric(df["target"])

# Convert UCI goal 1-4 to presence=1 for binary classification if necessary.
y = (y > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

plt.figure(figsize=(18, 9))
plot_tree(model, feature_names=X.columns, class_names=["No Disease","Disease"],
          filled=True, rounded=True)
plt.tight_layout()
plt.show()
