import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def prepraration_data(data):
    # Filter data
    data = data[data['ukt_rev'] != 0]
    data['keberadaan_orangtua'] = data['keberadaan_ayah'] + data['keberadaan_ibu']
    data['pajak'] = data['koreksi_harta_pajak_mobil'] + data['koreksi_harta_pajak_motor']
    data['total_pendapatan'] = data['penghasilan_ayah'] + data['penghasilan_ibu']
    data['rata-rata_pendapatan'] = data['total_pendapatan'] / data['jumlah_tanggungan']

    # Mapping fakultas
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
            return 999

    data['fakultas'] = data['prodi_nama'].apply(map_to_fakultas)

    # Mapping pendapatan_class
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

    data['rata-rata_pendapatan'] = data['rata-rata_pendapatan'].apply(map_to_pendapatan_class)

    # Mapping pekerjaan ayah dan ibu
    mapping_pekerjaan = {
        'Sudah Meninggal': 0,
        'Tidak Bekerja': 1,
        'Buruh': 2,
        'Nelayan': 2,
        'Petani': 2,
        'Peternak': 2,
        'Pedagang Kecil': 3,
        'Karyawan Swasta': 4,
        'Lainnya': 5,
        'Wiraswasta': 6,
        'Wirausaha': 6,
        'Pedagang Besar': 7,
        'Pensiunan': 8,
        'PNS / TNI / Polri': 9,
        
    }

    data['pekerjaan_ayah'] = data['pekerjaan_ayah'].map(mapping_pekerjaan)
    data['pekerjaan_ibu'] = data['pekerjaan_ibu'].map(mapping_pekerjaan)

    # Isi nilai NaN
    data = data.fillna(data.mode().iloc[0])

    # Pilih kolom yang diambil
    selected_columns = ['no_test', 'id', 'fakultas', 'pekerjaan_ayah', 'pekerjaan_ibu', 'rata-rata_pendapatan',
                        'penghasilan_ayah', 'penghasilan_ibu', 'kepemilikan_rumah',
                        'koreksi_pengeluaran_mhs_iuran_sekolah', 'kendaraan', 'sekolah', 'listrik',
                        'keberadaan_orangtua', 'pajak', 'ukt_rev']

    selected_data = data[selected_columns]

    return selected_data
