# import pandas as pd

# # Baca data dari file CSV
# data = pd.read_csv('data1_cleaned.csv')

# # Hapus data yang duplikat berdasarkan semua kolom
# data_cleaned = data.drop_duplicates()

# # Simpan data yang telah dibersihkan ke dalam file CSV baru
# data_cleaned.to_csv('data_mentah.csv', index=False)


# ...................................................................

import pandas as pd

# Membaca file CSV 1
data1 = pd.read_csv('data2_cleaned.csv')

# Membaca file CSV 2
data2 = pd.read_csv('misclassified_data_targeted.csv')

# Mengambil ID dari file CSV 2
ids_to_remove = data2['id'].tolist()

# Menghapus data pada file CSV 1 dengan ID yang ada di file CSV 2 dan nilai ukt_rev bukan 5
data1 = data1[~((data1['id'].isin(ids_to_remove)) & (data1['ukt_rev'] != 5))]
# data1 = data1[~data1['id'].isin(ids_to_remove)]
# Menyimpan data yang telah dihapus ke dalam file CSV baru
data1.to_csv('data3_cleaned.csv', index=False)