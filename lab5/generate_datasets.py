import os
import pandas as pd
import numpy as np

np.random.seed(2026)

os.makedirs("lab5", exist_ok=True)

N = 250

def generate_base_data(N, x1_range, x2_range):
    # Генерируем два независимых базовых признака с непрерывным равномерным распределением
    x1 = np.random.uniform(x1_range[0], x1_range[1], N)
    x2 = np.random.uniform(x2_range[0], x2_range[1], N)
    # Синтезируем сложный нелинейный признак х3 на основе х1 (Feature Engineering)
    x3 = np.sin(x1) * 5
    return x1, x2, x3

# линейная зависимость y = 2.5*x1 - 4*x2 + 1.8*x3 + 12.5

# 1. Первый чистый датасет
x1_a, x2_a, x3_a = generate_base_data(N, (0, 10), (0, 10))
y_clean_a = 2.5 * x1_a - 4 * x2_a + 1.8 * x3_a + 12.5
df_clean_a = pd.DataFrame({"x1": x1_a, "x2": x2_a, "x3": x3_a, "y": y_clean_a})
df_clean_a.to_csv("lab5/train_clean_1.csv", index=False)

# 2. Второй чистый датасет
x1_b, x2_b, x3_b = generate_base_data(N, (5, 15), (2, 12))
y_clean_b = 2.5 * x1_b - 4 * x2_b + 1.8 * x3_b + 12.5
df_clean_b = pd.DataFrame({"x1": x1_b, "x2": x2_b, "x3": x3_b, "y": y_clean_b})
df_clean_b.to_csv("lab5/train_clean_2.csv", index=False)

# 3. Третий чистый датасет
x1_c, x2_c, x3_c = generate_base_data(N, (10, 20), (5, 15))
y_clean_c = 2.5 * x1_c - 4 * x2_c + 1.8 * x3_c + 12.5
df_clean_c = pd.DataFrame({"x1": x1_c, "x2": x2_c, "x3": x3_c, "y": y_clean_c})
df_clean_c.to_csv("lab5/test_clean_3.csv", index=False)

# 4. Датасет с сильным шумом, добавляем гетероскедастичный шум
noise = np.random.normal(0, 1.2 * x1_c, N)
y_noisy = 2.5 * x1_c - 4 * x2_c + 1.8 * x3_c + 12.5 + noise
df_noisy = pd.DataFrame({"x1": x1_c, "x2": x2_c, "x3": x3_c, "y": y_noisy})
df_noisy.to_csv("lab5/test_noisy.csv", index=False)

print("Все 4 датасета успешно созданы и сохранены в папку lab5/")