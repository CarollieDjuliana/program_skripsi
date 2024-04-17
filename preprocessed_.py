import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data 
data = pd.read_csv('databaseUkt23.csv', delimiter=',', encoding='latin-1')
# data = pd.read_excel('merge_file.xlsx')
# Menghapus data yang nilai ukt 0
data = data[data['ukt'] != 0]
# Tambahkan kondisi untuk menghapus data dengan penghasilan ayah dan ibu yang sama dengan 0
data = data[(data['penghasilan_ayah'] != 0) | (data['penghasilan_ibu'] != 0)]

# Kolom total_penghasilan'
data['total_pendapatan'] =( data['penghasilan_ayah'] + data['penghasilan_ibu'] )
data['pendapatan_class'] =( data['penghasilan_ayah'] + data['penghasilan_ibu'] )/data['jumlah_tanggungan']

# # fakultas
def map_to_fakultas(jurusan):
    if jurusan in ["Pendidikan Dokter", "Pendidikan Dokter Gigi", "Psikologi", "Ilmu Keperawatan"]:
        return 1
    elif jurusan in ["Agroekoteknologi","Agronomi", "Ilmu Tanah", "Agribisnis", "Teknik Pertanian", "Ilmu dan Teknologi Pangan", "Peternakan", "Budidaya Perairan (Akuakultur)", "Proteksi Tanaman", "Teknologi Hasil Perikanan", "Budidaya Perairan", "Teknologi Hasil Pertanian"]:
        return 2
    elif jurusan in ["Sistem Informasi", "Teknik Informatika", "Ilmu Komputer", "Sistem Komputer"]:
        return 3
    elif jurusan in ["Pendidikan Matematika", "Pendidikan Bahasa Inggris", "Pendidikan Bahasa Indonesia", "Keguruan & Pendidikan", "Pendidikan Guru Sekolah Dasar (PGSD)", "Pendidikan Jasmani, Kesehatan, dan Rekreasi", "Bimbingan dan Konseling", "Pendidikan Guru Anak Usia Dini (PGAUD)", "Pendidikan Biologi", "Pendidikan Kimia", "Pendidikan Fisika", "Pendidikan Teknik Mesin", "Pendidikan Ekonomi", "Pendidikan Masyarakat", "Pendidikan Sejarah", "Pendidikan Pancasila & Kewarganegaraan (PPKN)", "Pendidikan Bahasa, Sastra Indonesia & Daerah", "Pendidikan Jasmani dan Kesehatan"]:
        return 4
    elif jurusan in ["Manajemen", "Akuntansi", "Ekonomi Pembangunan"]:
        return 5
    elif jurusan in ["Gizi", "Ilmu Kesehatan Masyarakat", "Kesehatan Lingkungan"]:
        return 6
    elif jurusan in ["Teknik Sipil", "Teknik Pertambangan", "Teknik Mesin", "Teknik Kimia", "Teknik Elektro", "Arsitektur", "Teknik Geologi"]:
        return 7
    elif jurusan in ["ILMU HUBUNGAN INTERNASIONAL", "Sosiologi", "Ilmu Komunikasi", "Ilmu Administrasi Publik"]:
        return 8
    elif jurusan in ["Matematika", "Fisika", "Kimia", "Biologi", "Kelautan", "Farmasi", "Ilmu Kelautan"]:
        return 9
    elif jurusan == "Ilmu Hukum":
        return 10
    else:
        return "Tidak Diketahui"
    
# membuat kolom fakultas
data['fakultas'] = data['prodi_nama'].apply(map_to_fakultas)



# fungsi pendapatan_class
def map_to_pendapatan_class(total_penghasilan):
    if total_penghasilan < 500000:
        return 1
    elif 500000 <= total_penghasilan <= 650000:
        return 2
    elif 650001 <= total_penghasilan <= 800000:
        return 3
    elif 800001 <= total_penghasilan <= 950000:
        return 4
    elif 950001 <= total_penghasilan <= 1100000:
        return 5
    elif 1100001 <= total_penghasilan <= 1250000:
        return 6
    elif 1250001 <= total_penghasilan <= 1400000:
        return 7
    else:
        return 8

# kolom pendapatan_class 
data['pendapatan_class'] = data['pendapatan_class'].apply(map_to_pendapatan_class)
data['total_pendapatan'] = data['total_pendapatan'].apply(map_to_pendapatan_class)
# data['penghasilan_ibu'] = data['penghasilan_ibu'].map(map_to_pendapatan_class)
# data['penghasilan_ayah'] = data['penghasilan_ayah'].map(map_to_pendapatan_class)

mapping_pendapatan_perkapita = {
     '< 500000': 1,
    '>=500000, <=650000': 2,
    '>650000, <=800000': 3,
    '>800000, <=950000': 4,
    '>950000, <=1100000': 5,
    '>1100000, <=1250000': 6,
    '>1250000, <=1400000': 7,
    '>1400000': 8
}

# data['Penghasilan perkapita'] = data['Penghasilan perkapita'].map(mapping_pendapatan_perkapita)


# ubah status ayah dan status ibu jadi binner
# data['keberadaan_ayah'] = data['keberadaan_ayah'].map({'Almarhum':0, 'Masih Hidup':1})
# data['keberadaan_ibu'] = data['keberadaan_ibu'].map({'Almarhum':0, 'Masih Hidup':1})

# pekerjaan ayah dan ibu 
mapping_pekerjaan = {
    'Tidak Bekerja': 1,
    'Nelayan': 2,
    'Petani': 3,
    'Peternak': 4,
    'PNS / TNI / Polri': 5,
    'Karyawan Swasta': 6,
    'Pedagang Kecil': 7,
    'Pedagang Besar': 8,
    'Wiraswasta': 9,
    'Wirausaha': 10,
    'Buruh': 11,
    'Pensiunan': 12,
    'Sudah Meninggal': 13,
    'Lainnya': 14
}

data['pekerjaan_ayah']=data['pekerjaan_ayah'].map(mapping_pekerjaan)
data['pekerjaan_ibu']=data['pekerjaan_ibu'].map(mapping_pekerjaan)

# kepemilikan rumah 
mapping_kepemilikan_rumah = {
    'Tidak Memiliki (Rumah Sewa/Kontrak/Menumpang)': 1,
    'Rumah Sangat Sederhana Pribadi (Nilai < 50 Juta)': 2,
    'Rumah Sangat Sederhana Pribadi (Nilai < 100 Juta)': 3,
    'Rumah Sederhana Pribadi (Nilai 100 Juta - 150 Juta)': 4,
    'Rumah Sederhana Pribadi (Nilai 150 Juta - 200 Juta)': 5,
    'Rumah KPR Pribadi (Nilai 200 Juta - 250 Juta)': 6,
    'Rumah Pribadi (Nilai 250 Juta - 500 Juta) ': 7,
    'Rumah Pribadi (Nilai > 500 Juta) ': 8

}
# data['kepemilikan_rumah']=data['kepemilikan_rumah'].map(mapping_kepemilikan_rumah)

# kendaraan
mapping_kendaraan = {
    'Tidak memiliki kendaraan': 1,
    'Memiliki sepeda': 2,
    'Memiliki 1 (Satu) Sepeda Motor Pribadi': 3,
    'Memiliki 2 (Dua) Sepeda Motor Pribadi': 4,
    'Memiliki 3 (Tiga) Sepeda Motor Pribadi': 5,
    'Memiliki 1 (Satu) Mobil Pribadi (Nilai < 150 Juta)': 6,
    'Memiliki 1 (Satu) Mobil Pribadi (Nilai 150 s.d 200 Juta)': 7,
    'Memiliki 1 (Satu) Mobil Pribadi (Nilai > 200 Juta)': 8
}
# data['kendaraan']=data['kendaraan'].map(mapping_kendaraan)

# sekolah
mapping_sekolah = {
    'Biaya sekolah/bulan <= Rp. 50.000': 1,
    'Biaya sekolah/bulan Rp. 50.000 s/d Rp. 100.000': 2,
    'Biaya sekolah Rp. 100.000 s/d Rp. 200.000': 3,
    'Biaya sekolah Rp. 200.000 s/d Rp. 500.000': 4,
    'Biaya sekolah > Rp. 500.000': 5
}
# data['sekolah']=data['sekolah'].map(mapping_sekolah)

# listrik
mapping_listrik = {
    'VA 450 (Atau Menumpang Sambungan/Tidak Memiliki Sambungan Listrik)': 1,
    'VA 900': 2,
    'VA 1.300': 3,
    'VA 2.200': 4,
    'VA > 2.200': 5
}
# data['listrik']=data['listrik'].map(mapping_listrik)
# NAN
data = data.fillna(data.mode().iloc[0])

data['ukt'] = pd.to_numeric(data['ukt'], errors='coerce')

# Menggunakan opsi future.no_silent_downcasting
pd.set_option('future.no_silent_downcasting', True)

# Daftar kolom yang diambil
selected_columns = ['no_test', 'fakultas',  'pekerjaan_ayah', 
                   'pekerjaan_ibu', 'penghasilan_ayah', 'penghasilan_ibu','jumlah_tanggungan',
                    'total_pendapatan','pendapatan_class', 'kepemilikan_rumah', 
                    'kendaraan', 'sekolah', 'listrik','penghasilan','pajak_mobil', 'pajak_motor',  'ukt']
# selected_columns = ['no_test', 'fakultas', 'keberadaan_ayah', 'pekerjaan_ayah', 
#                     'keberadaan_ibu', 'pekerjaan_ibu', 'penghasilan_ayah', 'penghasilan_ibu',
#                     'total_pendapatan','pendapatan_class','jumlah_tanggungan', 'kepemilikan_rumah', 
#                     'kendaraan', 'sekolah', 'listrik','penghasilan','pajak_mobil', 'pajak_motor',  'ukt']

# selected_columns = ['no_test', 'fakultas', 'pekerjaan_ayah', 
#                     'pekerjaan_ibu', 'penghasilan_ayah', 'penghasilan_ibu',
#                     'total_pendapatan','pendapatan_class','jumlah_tanggungan', 'kepemilikan_rumah', 
#                     'kendaraan', 'sekolah', 'listrik','Penghasilan perkapita', 'ukt']

selected_data = data[selected_columns]

corr_matrix = selected_data.corr()
print(corr_matrix)
sns.heatmap(corr_matrix)
# Simpan data
selected_data.to_csv('databaseUkt23_preprocessing.csv', index=False)

# nan_values = selected_data[selected_data.isnull().any(axis=1)]
# print(nan_values)


# print(selected_data)
# unique_values = data['ukt'].unique()
# print(unique_values)
# data['ukt'] = pd.to_numeric(data['ukt'], errors='coerce')
# nan_rows = data[data['ukt'].isna()]
# print("Jumlah baris dengan nilai NaN pada kolom 'ukt':", nan_rows.shape[0])
# print(nan_rows)

# data_nan = data.isnull().sum()
# print(data_nan)

