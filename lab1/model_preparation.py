import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

train = pd.read_csv("data/train/train_scaled.csv")
X_train = train[["day"]]
y_train = train["temperature"]

model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training completed")