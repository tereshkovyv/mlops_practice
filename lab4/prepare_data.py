import seaborn as sns
import pandas as pd
import os

os.makedirs('lab4', exist_ok=True)

df = sns.load_dataset('penguins')

# species (вид), sex (пол), body_mass_g (масса с пропусками)
base_df = df[['species', 'sex', 'body_mass_g']]

base_df.to_csv('lab4/penguins.csv', index=False)
print("Базовый датасет сохранен. Пропусков в массе:", base_df['body_mass_g'].isna().sum())