import pandas as pd

# Membaca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed3.csv')

# Mengonversi kolom "no_test" menjadi string
data['no_test'] = data['no_test'].astype(str)

# Memisahkan data ke dalam dua file
fileSatu = data[data['no_test'].str.len() == 9]
fileDua = data[data['no_test'].str.len() != 9]

# Menyimpan data ke dalam file baru
fileSatu.to_csv('fileSatu.csv', index=False)
fileDua.to_csv('fileDua.csv', index=False)