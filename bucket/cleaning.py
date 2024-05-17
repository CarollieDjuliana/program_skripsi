# import pandas as pd

# data = pd.read_csv('data_verifikasi_ukt2023.csv')

# # Hapus data yang duplikat berdasarkan semua kolom
# data_cleaned = data.drop_duplicates()

# # Simpan data yang telah dibersihkan ke dalam file CSV baru
# data_cleaned.to_csv('data_verifikasi_ukt2023.csv', index=False)


# ...................................................................

# import pandas as pd

# # Membaca file CSV 1
# data1 = pd.read_csv('data3_cleaned.csv')

# # Membaca file CSV 2
# data2 = pd.read_csv('misclassified_datahasil.csv')

# # Mengambil ID dari file CSV 2
# ids_to_remove = data2['id'].tolist()

# # Menghapus data pada file CSV 1 dengan ID yang ada di file CSV 2 dan nilai ukt_rev bukan 5
# data1 = data1[~((data1['id'].isin(ids_to_remove)))]
# # data1 = data1[~data1['id'].isin(ids_to_remove)]
# # Menyimpan data yang telah dihapus ke dalam file CSV baru
# data1.to_csv('data4_cleaned.csv', index=False)

# .....................................................................................

# import pandas as pd

# # Baca file CSV
# data = pd.read_csv('misclassified_data.csv')

# # Hapus data dengan nilai ukt_rev tertentu
# # nilai_ukt_rev = ''
# data = data[(data['ukt_rev'] == 6) & (data['predicted_ukt'] == 5)]

# # data = data[data['predicted_ukt'] == 5]


# # Simpan data yang sudah dihapus
# data.to_csv('misclassified_datahasil.csv', index=False)

# ...............................................................................................


# data_csv2 = pd.read_csv('data_ukt_rev_1.csv')
# data_csv1 = pd.read_csv('bucket/data_gabungan2.csv')

# # Ambil ID yang sama di kedua file
# same_ids = data_csv1[data_csv1['id'].isin(data_csv2['id'])]['id']

# # Filter data_csv1 berdasarkan ID yang sama dengan data_csv2
# filtered_data = data_csv1[data_csv1['id'].isin(same_ids)]

# # Simpan data yang sudah difilter ke dalam file CSV baru
# filtered_data.to_csv('ukt1.csv', index=False)
# print("Data yang memiliki ID yang sama telah disalin ke filtered_data.csv")

# ..................................................................................................

import pandas as pd

# # Baca file CSV
# df = pd.read_csv('data_verifikasi_ukt2023.csv')

# # Pilih kolom yang mau disalin
# kolom_tertentu = df[['id',  'prodi_nama', 'keberadaan_ayah', 'keberadaan_ibu',  'pekerjaan_ayah', 'pekerjaan_ibu',
#                      'jumlah_tanggungan', 'penghasilan_ayah', 'penghasilan_ibu',
#                      'kepemilikan_rumah', 'koreksi_pengeluaran_mhs_iuran_sekolah', 'kendaraan', 'listrik',
#                      'pajak_mobil', 'pajak_motor', 'ukt']]
# # Buat DataFrame baru hanya dengan kolom yang dipilih
# df_baru = pd.DataFrame(kolom_tertentu)

# # Simpan DataFrame baru ke file CSV
# df_baru.to_csv('data_verifikasi_ukt2023.csv', index=False)

# ...............................................................................
# import pandas as pd
# # Baca file CSV
# df = pd.read_csv('a_aUKT1.csv')

# # Seleksi data dengan kolom 'ukt_rev' bernilai 1
# df_ukt_rev_1 = df[df['ukt_rev'] == 1]
# # Simpan data yang sudah diseleksi ke dalam file CSV
# df_ukt_rev_1.to_csv('data_ukt_rev_1.csv', index=False)

# ............................................................................................
import pandas as pd

# Load dataset from CSV
data = pd.read_csv('after_preprocessing.csv')

selected_ids = [1406, 4322, 4098, 4112, 2698, 1637,
                3164, 1125, 3097, 1995, 821, 1766, 3836, 2930, 379]

# Salin data berdasarkan nilai ID
bootstrap_data_id = data[data['id'].isin(selected_ids)]

# Simpan data ke dalam file CSV
bootstrap_data_id.to_csv('bootstrap_data_id.csv', index=False)
