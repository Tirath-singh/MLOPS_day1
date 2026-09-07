import pandas as pd
import numpy as np
import joblib

new_data = pd.DataFrame([[150, 30, 20]], columns=["TV", "radio", "newspaper"])

model = joblib.load("models/linear_model.pkl")

predictions = model.predict(new_data)


print(f"Predicted sale is {predictions[0]}")