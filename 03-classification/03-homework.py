# ==========================================
# Lead Scoring Homework
# ==========================================

# Q0. Setup & Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv('./raw_data/course_lead_scoring.csv')


# Target variable
target = 'converted'

# Data Preparation

# Separate categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=np.number).columns
num_cols = num_cols.drop(target)

# Handle missing values
df[cat_cols] = df[cat_cols].fillna('NA')
df[num_cols] = df[num_cols].fillna(0.0)

# Q1
mode_industry = df['industry'].mode()[0]
print("Q1. Mode of 'industry':", mode_industry)

# Q2
corr_matrix = df[num_cols].corr()

pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count')
]

pair_corr = {pair: corr_matrix.loc[pair[0], pair[1]] for pair in pairs}
max_corr_pair = max(pair_corr, key=pair_corr.get)

print("Q2. Highest correlation pair:", max_corr_pair)

# Split Data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)


y_train = df_train[target].values
y_val = df_val[target].values

del df_train[target]
del df_val[target]
del df_test[target]

# Q3
def mutual_info_scores(X, y):
    return pd.Series(mutual_info_classif(X, y, discrete_features=True), index=X.columns)

# One-hot encode categorical vars for MI computation
X_train_cat = pd.get_dummies(df_train[cat_cols])
mi = mutual_info_scores(X_train_cat, y_train)
mi_sorted = mi.sort_values(ascending=False)
mi_rounded = mi_sorted.round(2)
print("Q3. Mutual Information Scores:\n", mi_rounded)

# Identify highest MI among given options
options = ['industry', 'location', 'lead_source', 'employment_status']
cat_with_max_mi = max(options, key=lambda x: mi_rounded.filter(like=x).max())
print("Q3. Highest MI categorical variable:", cat_with_max_mi)

# Q4
from sklearn.preprocessing import OneHotEncoder

# One-hot encode categorical variables
df_train_full = df_train.copy()
df_val_full = df_val.copy()

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(df_train_full[cat_cols])

X_train = np.hstack([
    encoder.transform(df_train_full[cat_cols]),
    df_train_full[num_cols].values
])
X_val = np.hstack([
    encoder.transform(df_val_full[cat_cols]),
    df_val_full[num_cols].values
])

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Q4. Validation Accuracy:", round(acc, 2))

# Q5
base_acc = acc
feature_diffs = {}

for feature in list(cat_cols) + list(num_cols):
    temp_df_train = df_train_full.drop(columns=[feature])
    temp_df_val = df_val_full.drop(columns=[feature])

    # Encode again
    temp_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    temp_encoder.fit(temp_df_train[cat_cols.intersection(temp_df_train.columns)])

    X_train_temp = np.hstack([
        temp_encoder.transform(temp_df_train[cat_cols.intersection(temp_df_train.columns)]),
        temp_df_train[num_cols.intersection(temp_df_train.columns)].values
    ])
    X_val_temp = np.hstack([
        temp_encoder.transform(temp_df_val[cat_cols.intersection(temp_df_val.columns)]),
        temp_df_val[num_cols.intersection(temp_df_val.columns)].values
    ])

    model_temp = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_temp.fit(X_train_temp, y_train)
    acc_temp = accuracy_score(y_val, model_temp.predict(X_val_temp))
    feature_diffs[feature] = abs(base_acc - acc_temp)

min_feature = min(feature_diffs, key=feature_diffs.get)
print("Q5. Least useful feature (smallest difference):", min_feature)

# Q6
C_values = [0.01, 0.1, 1, 10, 100]
acc_results = {}

for C in C_values:
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc_C = accuracy_score(y_val, model.predict(X_val))
    acc_results[C] = round(acc_C, 3)

best_C = max(acc_results, key=acc_results.get)
print("Q6. Validation accuracy by C:", acc_results)
print("Q6. Best C:", best_C)
