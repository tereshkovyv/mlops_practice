import pandas as pd

# Читаем текущую версию
df_v2 = pd.read_csv('lab4/penguins.csv')

# Находим среднее значение массы
mean_mass = df_v2['body_mass_g'].mean()

# Заполняем пропуски
df_v2['body_mass_g'] = df_v2['body_mass_g'].fillna(mean_mass)

# Перезаписываем файл
df_v2.to_csv('lab4/penguins.csv', index=False)
print("Пропуски заполнены средним. Осталось NaNs:", df_v2['body_mass_g'].isna().sum())