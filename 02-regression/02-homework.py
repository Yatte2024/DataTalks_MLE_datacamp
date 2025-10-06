import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('./raw_data/car_fuel_efficiency.csv', usecols=['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg'])

# Q1
print(df.isna().sum())

# Q2
print(df.horsepower.median())


# prepare data
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df_shuffled)
n_train = int(n * 0.6)
n_val = int(n * 0.2)
n_test = n - n_train - n_val

df_train = df_shuffled.iloc[:n_train]
df_val = df_shuffled.iloc[n_train:n_train + n_val]
df_test = df_shuffled.iloc[n_train + n_val:]

# Q3
def prepare_X(df, fillna_value):
    df_copy = df.copy()
    df_copy = df_copy.fillna(fillna_value)
    X = df_copy.drop(columns=['fuel_efficiency_mpg']).values
    return X

y_train = df_train['fuel_efficiency_mpg'].values
y_val = df_val['fuel_efficiency_mpg'].values
y_test = df_test['fuel_efficiency_mpg'].values

X_train_0 = prepare_X(df_train, 0)
X_val_0 = prepare_X(df_val, 0)

model_0 = LinearRegression()
model_0.fit(X_train_0, y_train)
pred_0 = model_0.predict(X_val_0)
rmse_0 = round(np.sqrt(mean_squared_error(y_val, pred_0)), 2)

# Fill with mean (horsepower)
mean_hp = df_train['horsepower'].mean()
X_train_mean = prepare_X(df_train, {'horsepower': mean_hp})
X_val_mean = prepare_X(df_val, {'horsepower': mean_hp})

model_mean = LinearRegression()
model_mean.fit(X_train_mean, y_train)
pred_mean = model_mean.predict(X_val_mean)
rmse_mean = round(np.sqrt(mean_squared_error(y_val, pred_mean)), 2)

print("\nQ3 - RMSE with fill 0:", rmse_0)
print("Q3 - RMSE with fill mean:", rmse_mean)

# Q4
r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
rmse_ridge = {}

for r in r_values:
    model = Ridge(alpha=r)
    model.fit(X_train_0, y_train)
    pred = model.predict(X_val_0)
    rmse = round(np.sqrt(mean_squared_error(y_val, pred)), 2)
    rmse_ridge[r] = rmse

print("\nQ4 - RMSE by r value:")
for r, score in rmse_ridge.items():
    print(f"r={r}: RMSE={score}")

# Q5
rmse_seeds = []
for seed in range(10):
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_train = df_shuffled.iloc[:n_train]
    df_val = df_shuffled.iloc[n_train:n_train + n_val]

    X_train = prepare_X(df_train, 0)
    y_train = df_train['fuel_efficiency_mpg'].values
    X_val = prepare_X(df_val, 0)
    y_val = df_val['fuel_efficiency_mpg'].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    rmse_seeds.append(rmse)

std_rmse = round(np.std(rmse_seeds), 3)
print("\nQ5 - Std of RMSEs across seeds:", std_rmse)

# Q6
df_shuffled = df.sample(frac=1, random_state=9).reset_index(drop=True)
df_train_full = df_shuffled.iloc[:n_train + n_val]
df_test = df_shuffled.iloc[n_train + n_val:]

X_train_final = prepare_X(df_train_full, 0)
y_train_final = df_train_full['fuel_efficiency_mpg'].values
X_test = prepare_X(df_test, 0)
y_test = df_test['fuel_efficiency_mpg'].values

final_model = Ridge(alpha=0.001)
final_model.fit(X_train_final, y_train_final)
pred_test = final_model.predict(X_test)
rmse_final = round(np.sqrt(mean_squared_error(y_test, pred_test)), 3)
print("\nQ6 - Final test RMSE:", rmse_final)
