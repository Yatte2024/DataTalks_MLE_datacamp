import pandas as pd
import numpy as np
# 1
pd.__version__

#2
df = pd.read_csv('raw_data/car_fuel_efficiency.csv')
len(df)

#3
len(df.fuel_type.unique())

#4
sum(df.isna().sum()>0)

#5
df[df.origin=='Asia'].fuel_efficiency_mpg.max()

#6
median_1 = df.horsepower.median()
mode_ = df.horsepower.mode()[0]
df.horsepower = df.horsepower.fillna(mode_)
median_2 = df.horsepower.median()

print(median_1)
print(median_2)

#7

## (1-4)
X = df.loc[df.origin=='Asia', ['vehicle_weight', 'model_year']][:7].to_numpy()

## (5)
XT = X.T
XTX = np.dot(XT, X)

## (6)
XTX_inv = np.linalg.inv(XTX)

## (7)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

## (8)
w = np.dot(np.dot(XTX_inv, XT), y)

## (9)
w_sum = np.sum(w)
