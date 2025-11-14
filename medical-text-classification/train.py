import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

train  = pd.read_csv("data/train.csv").dropna(subset=["text", "label"])
test  = pd.read_csv("data/test.csv").dropna(subset=["text", "label"])

X_train, y_train_str = train["text"], train["label"]
X_test,  y_test_str  = test["text"],  test["label"]

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_str)
y_test  = le.transform(y_test_str)

print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 1. Define TF-IDF + candidate models
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

pipelines = {
    "lr": Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),

    "rf": Pipeline([
        ("tfidf", tfidf),
        ("clf", RandomForestClassifier(random_state=42))
    ]),

    "xgb": Pipeline([
        ("tfidf", tfidf),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="auto",
            random_state=42,
            use_label_encoder=False
        ))
    ]),
}


# 2. Define hyperparameters for each model
param_grids = {
    "lr": {
        "clf__C": [0.1, 1.0, 5.0],
    },
    "rf": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 20],
    },
    "xgb": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [4, 6],
        "clf__learning_rate": [0.05, 0.1],
    }
}

# 3. Train models using GridSearchCV
best_models = []
results_table = []

for name, pipe in pipelines.items():
    print(f"\n===== Training model: {name} =====")

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        scoring="f1_macro",   # macro-F1 is best for imbalanced multi-class tasks
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    # Fit using encoded integer labels y_train
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV macro-F1:", grid.best_score_)

    best_models.append((name, grid.best_estimator_, grid.best_score_))

    results_table.append({
        "model": name,
        "cv_macro_f1": grid.best_score_
    })

# 3.Choose best model
best_models_sorted = sorted(best_models, key=lambda x: x[2], reverse=True)
best_name, best_model, best_cv = best_models_sorted[0]
print(f"\n>>> Best model: {best_name} (CV macro-F1 = {best_cv:.4f})")

out_path =  "models/best_model.pkl"
joblib.dump(best_model, out_path)
print("Saved best model to:", out_path)

joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")