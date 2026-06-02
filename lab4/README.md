## Пошаговое выполнение лабораторной работы

1. В качестве хранилища датасетов в dvc использован [гугл диск](https://drive.google.com/drive/u/0/folders/1lf-rT8YPUxz1nsCZR9w6IBpqaMxX5wdN).
```shell
dvc remote add -d myrepo gdrive://1lf-rT8YPUxz1nsCZR9w6IBpqaMxX5wd
```
2. Три версии датасета после выполнения каждого этапа находятся [здесь](https://drive.google.com/drive/u/0/folders/18Er3og9LV86ekqfUS3bj4XICkVBo3Ieq).
3. Был использован датасет Palmer Penguinsspecies 
* species (Вид пингвина) — аналог класса пассажира (Pclass).
* sex (Пол) — строковый признак для последующего кодирования (Sex).
* body_mass_g (Масса тела) — числовой признак, содержащий пропуски (Age).
```shell
df = sns.load_dataset('penguins')
```
4. Скрипт prepare_data.py отобрал нужные колонки, мы сделали первый коммит и успешно запушели данные.
5. Скрипт process_data.py посчитал среднее и заполнил пустые ячейки в массе тела. Версия зафиксирована в Git и DVC.
6. Скрипт encode_data.py применил pd.get_dummies к текстовой колонке пола.
7. Для демонстрации успешного версионирования данных было выполнено переключение между коммитами Git:
```shell
git checkout 2887cdb
dvc checkout
```
В итоге файл lab4/penguins.csv вернулся к исходному состоянию (появились пропуски и текстовый пол). Чтобы вернуться обратно к финальной версии выполнили команду
```shell
git checkout master
dvc checkout
```

