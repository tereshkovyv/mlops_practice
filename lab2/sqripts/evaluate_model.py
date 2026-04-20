import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

def evaluate():
    X_test = pd.read_csv("lab2/data/processed/X_test.csv")
    y_test = pd.read_csv("lab2/data/processed/y_test.csv")

    with open("lab2/models/model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate()