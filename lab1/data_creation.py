import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

def generate_data(size, noise=1.0, anomaly=False):
    x = np.arange(size)
    y = 10 + 0.05 * x + np.random.normal(0, noise, size)
    if anomaly:
        idx = np.random.choice(size, size // 10, replace=False)
        y[idx] += np.random.normal(15, 5, len(idx))

    return pd.DataFrame({"day": x, "temperature": y})

train = generate_data(200, noise=1.0, anomaly=True)
test = generate_data(50, noise=1.0, anomaly=False)
train.to_csv("data/train/train.csv", index=False)
test.to_csv("data/test/test.csv", index=False)

print("Data creation completed")