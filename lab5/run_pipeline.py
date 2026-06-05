import os

# 1. Запуск генерации  данных
os.system("python lab5/generate_datasets")
# 2. Обучение модели на датасетах
os.system("python lab5/train_model.py")
# 3. Тестирование с помощью pytest
os.system("pytest lab5/test_model.py -v")