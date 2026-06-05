import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "lab5/linear_model.pkl", "rb") as f:
    model = pickle.load(f)

clean_df = pd.read_csv(BASE_DIR / "lab5/test_clean_3.csv")
noisy_df = pd.read_csv(BASE_DIR / "lab5/test_noisy.csv")

X_clean, y_clean = clean_df[["x1", "x2", "x3"]], clean_df["y"]
X_noisy, y_noisy = noisy_df[["x1", "x2", "x3"]], noisy_df["y"]

y_pred_clean = model.predict(X_clean)
y_pred_noisy = model.predict(X_noisy)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(y_clean, y_pred_clean, color="darkgreen", alpha=0.6, label="Точки данных")
ax1.plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 'r--', lw=2, label="Идеальное предсказание")
ax1.set_title("Чистые данные (Test Clean 3)\nИдеальное соответствие тренду", fontsize=12)
ax1.set_xlabel("Реальные значения (y_true)", fontsize=10)
ax1.set_ylabel("Предсказанные значения (y_pred)", fontsize=10)
ax1.grid(True)
ax1.legend()

ax2.scatter(y_noisy, y_pred_noisy, color="crimson", alpha=0.6, label="Точки данных")
ax2.plot([y_noisy.min(), y_noisy.max()], [y_noisy.min(), y_noisy.max()], 'r--', lw=2, label="Идеальное предсказание")
ax2.set_title("Зашумленные данные (Test Noisy)\nСильный разброс вокруг тренда", fontsize=12)
ax2.set_xlabel("Реальные значения (y_true)", fontsize=10)
ax2.set_ylabel("Предсказанные значения (y_pred)", fontsize=10)
ax2.grid(True)
ax2.legend()

plt.tight_layout()

output_image_path = BASE_DIR / "model_performance.png"
plt.savefig(output_image_path, dpi=300)
print(f"График успешно сохранен по пути: {output_image_path}")
plt.show()