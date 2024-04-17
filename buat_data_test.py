import pandas as pd

# Membaca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed2.csv')

# Mengambil 100 baris terakhir
data_new = data.tail(1000)

# Menyimpan 100 baris terakhir ke dalam file CSV baru
data_new.to_csv('data_test2.csv', index=False)  # index=False untuk menghilangkan indeks baris dari output CSV
