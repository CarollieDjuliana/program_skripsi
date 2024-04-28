import pandas as pd

# Baca data dari file CSV
data = pd.read_csv('clean.csv')

# Hapus data yang duplikat berdasarkan semua kolom
data_cleaned = data.drop_duplicates()

# Simpan data yang telah dibersihkan ke dalam file CSV baru
data_cleaned.to_csv('data_mentah.csv', index=False)
