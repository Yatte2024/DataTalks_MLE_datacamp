import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

df = pd.read_csv('./raw_data/course_lead_scoring.csv')
target = 'converted'


# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=np.number).columns
num_cols = num_cols.drop(target)

# Handle missing values
df[cat_cols] = df[cat_cols].fillna('NA')
df[num_cols] = df[num_cols].fillna(0.0)

# Split Data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train[target]
y_test = df_test[target]
y_val = df_val[target]

del df_train[target]
del df_val[target]
del df_test[target]


# Q1
numeric_vars = [
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
    "annual_income"
]

auc_results = {}

for col in numeric_vars:
    vals = df_train[col].astype(float).values
    auc_val = roc_auc_score(y_train, vals)
    auc_results[col] = auc_val

best_feature = max(auc_results, key=auc_results.get)
print("Highest AUC feature:", best_feature)

# Q2
dv = DictVectorizer(sparse=False)
X_train_enc = dv.fit_transform(df_train.to_dict(orient="records"))
X_val_enc = dv.transform(df_val.to_dict(orient="records"))

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
model.fit(X_train_enc, y_train)

y_val_pred = model.predict_proba(X_val_enc)[:, 1]
auc_val = roc_auc_score(y_val, y_val_pred)

print(f"Q2: Validation AUC = {auc_val:.3f}")

# 3
thresholds = np.arange(0.0, 1.001, 0.01)
precisions, recalls = [], []

for t in thresholds:
    y_pred = (y_val_pred >= t).astype(int)
    p = precision_score(y_val, y_pred, zero_division=0)
    r = recall_score(y_val, y_pred, zero_division=0)
    precisions.append(p)
    recalls.append(r)

precisions, recalls = np.array(precisions), np.array(recalls)
t_intersect = thresholds[np.argmin(np.abs(precisions - recalls))]

print(f"Q3: Precision ≈ Recall at threshold ≈ {t_intersect:.3f}")

# 4
f1s = []
for p, r in zip(precisions, recalls):
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f1s.append(f1)

f1s = np.array(f1s)
t_best_f1 = thresholds[np.argmax(f1s)]

print(f"Q4: Max F1 at threshold ≈ {t_best_f1:.3f}")

# 5
df_full_train = df_train
df_full_train["target"] = y_train

kf = KFold(n_splits=5, shuffle=True, random_state=1)
aucs = []

for train_idx, val_idx in kf.split(df_full_train):
    df_tr = df_full_train.iloc[train_idx]
    df_va = df_full_train.iloc[val_idx]

    y_tr = df_tr["target"].values
    y_va = df_va["target"].values

    X_tr = df_tr.drop(columns=["target"])
    X_va = df_va.drop(columns=["target"])

    dv = DictVectorizer(sparse=False)
    X_tr_enc = dv.fit_transform(X_tr.to_dict(orient="records"))
    X_va_enc = dv.transform(X_va.to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    model.fit(X_tr_enc, y_tr)

    y_va_pred = model.predict_proba(X_va_enc)[:, 1]
    aucs.append(roc_auc_score(y_va, y_va_pred))

aucs = np.array(aucs)
print(f"\nQ5: AUCs per fold = {np.round(aucs, 3)}")
print(f"Std of AUCs = {np.std(aucs):.6f}")

#6
Cs = [1e-6, 1e-3, 1.0]
results = {}

for C in Cs:
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    aucs = []
    for train_idx, val_idx in kf.split(df_full_train):
        df_tr = df_full_train.iloc[train_idx]
        df_va = df_full_train.iloc[val_idx]
        y_tr = df_tr["target"].values
        y_va = df_va["target"].values
        X_tr = df_tr.drop(columns=["target"])
        X_va = df_va.drop(columns=["target"])
        dv = DictVectorizer(sparse=False)
        X_tr_enc = dv.fit_transform(X_tr.to_dict(orient="records"))
        X_va_enc = dv.transform(X_va.to_dict(orient="records"))
        model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
        model.fit(X_tr_enc, y_tr)
        y_va_pred = model.predict_proba(X_va_enc)[:, 1]
        aucs.append(roc_auc_score(y_va, y_va_pred))
    results[C] = {"mean": round(np.mean(aucs), 3), "std": round(np.std(aucs), 3)}

print("\nQ6: Hyperparameter tuning results:")
for C, res in results.items():
    print(f"C={C}: mean={res['mean']}, std={res['std']}")

best_C = sorted(results.items(), key=lambda x: (-x[1]["mean"], x[1]["std"], x[0]))[0][0]
print(f"✅ Best C = {best_C}")
