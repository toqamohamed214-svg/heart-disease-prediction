"""Heart disease prediction on the UCI Heart Disease (Cleveland) dataset.

Target: 1 = heart disease present, 0 = absent. We compare two models.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = pd.read_csv("data/processed.cleveland.data", header=None, names=cols, na_values="?")
df["target"] = (df["num"] > 0).astype(int)   # 0 = healthy, 1..4 = disease
X, y = df.drop(columns=["num", "target"]), df["target"]
print(f"{len(df)} patients, {int(y.sum())} with heart disease, "
      f"{int(df.isna().sum().sum())} missing values (filled with the median)\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"),
                                         StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=200, random_state=42)),
}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"== {name}: 5-fold CV {cv:.3f}, test accuracy {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=["no disease", "disease"], digits=2))

rf = models["Random Forest"]
imp = pd.Series(rf[-1].feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top features:", ", ".join(imp.head(5).index))
