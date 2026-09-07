import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

df = pd.read_csv("D:\\MLOPS_day1\\data\\data.csv")

X,y = df[["TV","radio","newspaper"]], df["sales"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=56)

model = LinearRegression()
model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)
r2= r2_score(ytest, ypred)
rmse = root_mean_squared_error(ytest, ypred)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

joblib.dump(model, "models/linear_model.pkl")