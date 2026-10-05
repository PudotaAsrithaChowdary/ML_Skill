# 6. Diabetes Severity — Multinomial Logistic Regression
# Dataset: BRFSS 2015 diabetes_012_health_indicators
# Target: 0 = no diabetes, 1 = prediabetes, 2 = diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("datasets/diabetes_012_health_indicators_BRFSS2015.csv")

target = "Diabetes_012"
X = df.drop(columns=[target])
y = df[target]

# BRFSS file is already mostly numeric. Median imputation makes the script robust.
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000, multi_class="multinomial", solver="lbfgs"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClasses:", pipe.named_steps["model"].classes_)
