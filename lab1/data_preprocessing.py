import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle

train = pd.read_csv("data/train/train.csv")
test = pd.read_csv("data/test/test.csv")
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

pd.DataFrame(train_scaled, columns=train.columns).to_csv(
    "data/train/train_scaled.csv", index=False
)
pd.DataFrame(test_scaled, columns=test.columns).to_csv(
    "data/test/test_scaled.csv", index=False
)

os.makedirs("model", exist_ok=True)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Data preprocessing completed")