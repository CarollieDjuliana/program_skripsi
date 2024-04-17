import pandas as pd

# Membaca file CSV
df = pd.read_csv('merged_file.csv')

# Mengambil baris ke-2000 hingga ke-3500
df_filtered = df[1000:3500]

# Menyimpan data ke file CSV baru
df_filtered.to_csv('databaseUkt23.csv', index=False)