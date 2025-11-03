import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# Load dataset
df = pd.read_csv("./raw_data/car_fuel_efficiency.csv")

# Fill missing values
df = df.fillna(0)

# Target variable
y = df['fuel_efficiency_mpg']
X = df.drop('fuel_efficiency_mpg', axis=1)

# Split: 60% train, 20% val, 20% test
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=1) # 0.25 * 0.8 = 0.2

# Vectorize
dv = DictVectorizer(sparse=True)
X_train_dv = dv.fit_transform(X_train.to_dict(orient='records'))
X_val_dv = dv.transform(X_val.to_dict(orient='records'))


# Q1
dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train_dv, y_train)

feature_importance = dt.feature_importances_
feature_names = dv.get_feature_names_out()
top_feature = feature_names[np.argmax(feature_importance)]
top_feature


# Q2
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train_dv, y_train)

y_pred = rf.predict(X_val_dv)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
rmse

# Q3
scores = []
for n in range(10, 210, 10):
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train_dv, y_train)
    y_pred = rf.predict(X_val_dv)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    scores.append((n, round(rmse, 3)))

scores_df = pd.DataFrame(scores, columns=["n_estimators", "RMSE"])
scores_df

# Q4
results = []

for depth in [10, 15, 20, 25]:
    rmses = []
    for n in range(10, 210, 10):
        rf = RandomForestRegressor(max_depth=depth, n_estimators=n, random_state=1, n_jobs=-1)
        rf.fit(X_train_dv, y_train)
        y_pred = rf.predict(X_val_dv)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)
    results.append((depth, np.mean(rmses)))

results_df = pd.DataFrame(results, columns=["max_depth", "mean_RMSE"])
results_df

# Q5
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train_dv, y_train)

importances = rf.feature_importances_
feature_names = dv.get_feature_names_out()
sorted_idx = np.argsort(importances)[::-1]
[(feature_names[i], importances[i]) for i in sorted_idx[:5]]

# Q6
dtrain = xgb.DMatrix(X_train_dv, label=y_train)
dval = xgb.DMatrix(X_val_dv, label=y_val)

watchlist = [(dtrain, 'train'), (dval, 'val')]

params_03 = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 0,
}

params_01 = params_03.copy()
params_01['eta'] = 0.1

# Train both
model_03 = xgb.train(params_03, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)
model_01 = xgb.train(params_01, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)

# Evaluate RMSE
y_pred_03 = model_03.predict(dval)
y_pred_01 = model_01.predict(dval)

rmse_03 = np.sqrt(mean_squared_error(y_val, y_pred))
rmse_01 = np.sqrt(mean_squared_error(y_val, y_pred))

rmse_03, rmse_01
