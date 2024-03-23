import pandas as pd

# Baca file CSV ke dalam DataFrame
# file1 = pd.read_csv('merge1.csv', delimiter=';', encoding='latin-1',index_col=False)
# file2 = pd.read_csv('merge2.csv', delimiter=';', encoding='latin-1', index_col=False)
# Membaca file Excel pertama
file1 = pd.read_excel('merge ukt_verifikasi.xlsx')

# Membaca file Excel kedua
file2 = pd.read_excel('merge ukt_verifikasi 2.xlsx')

# file1['no_test'] = file1['no_test'].astype(str).str.rstrip('.0')
# # Gabungkan kedua DataFrame berdasarkan kolom 'nama'
# merged_data = pd.merge(file1, file2, on='nama', how='left')

# # Perbarui kolom 'no_test' di DataFrame pertama dengan nilai kolom 'no_test' dari DataFrame kedua
# merged_data['no_test_x'] = merged_data['no_test_y']

# # Hapus kolom 'no_test_y' yang tidak diperlukan lagi
# merged_data.drop(columns=['no_test_y'], inplace=True)

# # Ubah nama kolom 'no_test_x' menjadi 'no_test'
# merged_data.rename(columns={'no_test_x': 'no_test'}, inplace=True)
# merged_data['no_test'] = merged_data['no_test'].astype(str).str.replace('.0', '')

# # Simpan DataFrame pertama yang sudah diperbarui ke dalam file CSV
# merged_data.to_csv('file1_updated.csv', index=False)


# ------

# Menggabungkan dua file berdasarkan kolom "no_test"
merged_df = pd.merge(file1, file2, on='no_test', how='inner')
merged_df = merged_df.drop(merged_df[(merged_df['fak_nama'] == 'D3') | (merged_df['ukt'] == 0)].index)

print(file1.dtypes)
print(file2.dtypes)
# Menyimpan hasil gabungan ke file CSV baru
merged_df.to_csv('merged_file.csv', index=False)

print("File CSV berhasil digabungkan!")