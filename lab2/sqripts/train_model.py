import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

def train():
    X_train = pd.read_csv("lab2/data/processed/X_train.csv")
    y_train = pd.read_csv("lab2/data/processed/y_train.csv")

    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())

    os.makedirs("lab2/models", exist_ok=True)

    with open("lab2/models/model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train()