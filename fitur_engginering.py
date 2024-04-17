import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Baca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed2.csv')

# Tentukan fitur kategorikal dan numerik
categorical_features = ['fakultas', 'pekerjaan_ayah','pekerjaan_ibu']
numeric_features = [ 'total_pendapatan', 'penghasilan_ayah', 'penghasilan_ibu', 'jumlah_tanggungan', 'kepemilikan_rumah', 'kendaraan', 'sekolah','listrik', 'penghasilan','pajak_mobil','pajak_motor']

# Ubah fitur kategorikal menjadi one-hot encoding
data_categorical = pd.get_dummies(data[categorical_features])

# Gabungkan fitur kategorikal dan numerik
X = pd.concat([data_categorical, data[numeric_features]], axis=1)
y = data['ukt']

# Hitung korelasi antar fitur
correlation_matrix = X.corr()

# Buat heatmap korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap Korelasi antar Fitur')
plt.show()
