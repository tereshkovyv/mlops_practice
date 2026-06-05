import pickle
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error, r2_score

@pytest.fixture
def model():
    with open("lab5/lab5/linear_model.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture
def test_clean_data():
    df = pd.read_csv("lab5/lab5/test_clean_3.csv")
    return df[["x1", "x2", "x3"]], df["y"]

@pytest.fixture
def test_noisy_data():
    df = pd.read_csv("lab5/lab5/test_noisy.csv")
    return df[["x1", "x2", "x3"]], df["y"]

def test_r2_on_clean_data(model, test_clean_data):
    X, y = test_clean_data
    r2 = r2_score(y, model.predict(X))
    assert r2 > 0.99, f"Качество упало на чистых данных, R2 = {r2:.4f}"

def test_model_coefficients(model):
    expected_coefs = [2.5, -4.0, 1.8]
    for i, expected in enumerate(expected_coefs):
        assert abs(model.coef_[i] - expected) < 1e-4, f"Ошибка в коэф {i}: {model.coef_[i]}"

def test_model_intercept(model):
    assert abs(model.intercept_ - 12.5) < 1e-4, f"Ошибка в сдвиге: {model.intercept_}"

def test_r2_on_noisy_data(model, test_noisy_data):
    X, y = test_noisy_data
    r2 = r2_score(y, model.predict(X))
    assert r2 > 0.95, f"На шумных данных R2 упал до: {r2:.4f}"

def test_mse_on_noisy_data(model, test_noisy_data):
    X, y = test_noisy_data
    mse = mean_squared_error(y, model.predict(X))
    assert mse < 5.0, f"Ошибка MSE выросла до: {mse:.4f}"