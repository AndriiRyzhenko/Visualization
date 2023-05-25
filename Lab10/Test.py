import pandas as pd
from sklearn.datasets import load_digits, load_iris

# Чтение данных из файла
data = pd.read_csv('ecoli_new.csv', delimiter=',', header=None)


# Отображение данных

extracted_data = data.iloc[:, 1:7]

print(extracted_data.values.tolist())
