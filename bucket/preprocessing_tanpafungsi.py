import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data_mentah.csv', delimiter=',', encoding='latin-1')

data = data[data['ukt_rev'] != 0]
data = data[data['fak_nama'] == 'S1']
data = data[data['koreksi_potensitambah_ukt'] == 0]
data = data[data['koreksi_potensitambah_ukt_catatan'].isnull()]
data['x'] = (data['koreksi_pengeluaran_mhs_iuran_sekolah'] * data['sekolah'])
data = data[(data['penghasilan_ayah'] != 0) | (data['penghasilan_ibu'] != 0)]
# Tambahkan kondisi untuk menghapus data dengan penghasilan ayah dan ibu yang sama dengan 0
data['keberadaan_orangtua'] = (
    data['keberadaan_ayah'] + data['keberadaan_ibu'])
data['pajak'] = (data['koreksi_harta_pajak_mobil'] +
                 data['koreksi_harta_pajak_motor'])

# Kolom total_penghasilan'
data['total_pendapatan'] = (
    data['koreksi_penghasilan_ayah'] + data['koreksi_penghasilan_ibu'])
data['pendapatan_class'] = (data['koreksi_penghasilan_ayah'] +
                            data['koreksi_penghasilan_ibu'])/data['jumlah_tanggungan']

# fakultas


def map_to_fakultas(jurusan):
    if jurusan in ["Pendidikan Dokter", "Pendidikan Dokter Gigi", "Psikologi", "Ilmu Keperawatan"]:
        return 1
    elif jurusan in ["Agroekoteknologi", "Agronomi", "Ilmu Tanah", "Agribisnis", "Teknik Pertanian", "Ilmu dan Teknologi Pangan", "Peternakan", "Budidaya Perairan (Akuakultur)", "Proteksi Tanaman", "Teknologi Hasil Perikanan", "Budidaya Perairan", "Teknologi Hasil Pertanian"]:
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
        return 999


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
data['pendapatan_class'] = data['pendapatan_class'].apply(
    map_to_pendapatan_class)


# pekerjaan ayah dan ibu
mapping_pekerjaan = {
    'Tidak Bekerja': 1,
    'Nelayan': 2,
    'Petani': 2,
    'Peternak': 2,
    'PNS / TNI / Polri': 9,
    'Karyawan Swasta': 6,
    'Pedagang Kecil': 3,
    'Pedagang Besar': 8,
    'Wiraswasta': 7,
    'Wirausaha': 8,
    'Buruh': 1,
    'Pensiunan': 6,
    'Sudah Meninggal': 0,
    'Lainnya': 5
}

data['pekerjaan_ayah'] = data['pekerjaan_ayah'].map(mapping_pekerjaan)
data['pekerjaan_ibu'] = data['pekerjaan_ibu'].map(mapping_pekerjaan)

# NAN
data = data.fillna(data.mode().iloc[0])

# Daftar kolom yang diambil
selected_columns = ['no_test', 'id', 'fakultas', 'pekerjaan_ayah', 'pekerjaan_ibu', 'pendapatan_class',
                    'penghasilan_ayah', 'penghasilan_ibu',
                    'kepemilikan_rumah', 'koreksi_pengeluaran_mhs_iuran_sekolah',
                    'kendaraan', 'sekolah', 'listrik', 'keberadaan_orangtua', 'pajak', 'ukt_rev']

selected_data = data[selected_columns]

corr_matrix = selected_data.corr()
print(corr_matrix)
sns.heatmap(corr_matrix)
# Simpan data
selected_data.to_csv('after_preprocessing.csv', index=False)
