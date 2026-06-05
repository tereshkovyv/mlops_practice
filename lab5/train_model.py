import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Объединяем два чистых датасета для обучения
df1 = pd.read_csv("lab5/train_clean_1.csv")
df2 = pd.read_csv("lab5/train_clean_2.csv")
df_train = pd.concat([df1, df2], ignore_index=True)

X_train = df_train[["x1", "x2", "x3"]]
y_train = df_train["y"]

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

print("Результаты обучения на объединенных чистых данных")
print(f"Подобранные коэффициенты: {model.coef_}")
print(f"Подобранный Intercept: {model.intercept_:.4f}")

# Сохранение модели
with open("lab5/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Модель сохранена в lab5/linear_model.pkl")