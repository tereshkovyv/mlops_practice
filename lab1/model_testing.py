import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error

test = pd.read_csv("data/test/test_scaled.csv")
X_test = test[["day"]]
y_test = test["temperature"]

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Model test MSE is: {mse:.4f}")