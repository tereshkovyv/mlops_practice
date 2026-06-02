import pandas as pd

# Читаем текущую версию
df_v3 = pd.read_csv('lab4/penguins.csv')

# Применяем One-Hot Encoding к колонке sex
df_v3 = pd.get_dummies(df_v3, columns=['sex'], drop_first=False)

# Перезаписываем файл
df_v3.to_csv('lab4/penguins.csv', index=False)
print("One-Hot Encoding применен. Новые колонки:", df_v3.columns.tolist())