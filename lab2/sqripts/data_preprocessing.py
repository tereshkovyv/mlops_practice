import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess():
    df = pd.read_csv("lab2/data/raw/titanic.csv")

    df = df.drop(["Name", "Ticket", "Cabin"], axis=1)
    df = df.dropna()

    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs("lab2/data/processed", exist_ok=True)

    X_train.to_csv("lab2/data/processed/X_train.csv", index=False)
    X_test.to_csv("lab2/data/processed/X_test.csv", index=False)
    y_train.to_csv("lab2/data/processed/y_train.csv", index=False)
    y_test.to_csv("lab2/data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    preprocess()