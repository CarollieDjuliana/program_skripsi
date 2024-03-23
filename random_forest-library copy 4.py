import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# Baca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed3.csv')

# # Tentukan fitur kategorikal dan numerik
# categorical_features = ['fakultas', 'keberadaan_ayah', 'pekerjaan_ayah', 'keberadaan_ibu', 'pekerjaan_ibu', 'kepemilikan_rumah', 'sekolah', 'listrik', 'kendaraan']
# numeric_features = ['jumlah_tanggungan', 'total_pendapatan', 'penghasilan_ayah', 'penghasilan_ibu']

# # Ubah fitur kategorikal menjadi one-hot encoding
# data_categorical = pd.get_dummies(data[categorical_features])


# Pisahkan fitur dan label
X = data.drop(columns=['ukt', 'no_test']).values
y = data['ukt'].values


# Handle Imbalanced Data dengan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Definisikan hyperparameter yang ingin dioptimalkan
param_grid = {
    'n_estimators': [200],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# Inisialisasi model Random Forest
model = RandomForestClassifier()

# Inisialisasi Grid Search dengan model, parameter grid, dan metrik evaluasi
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Latih model dengan Grid Search untuk mencari kombinasi hyperparameter terbaik
grid_search.fit(X_train, y_train)

# Mendapatkan model terbaik setelah Grid Search
best_model = grid_search.best_estimator_

# Prediksi nilai UKT untuk data uji dengan model terbaik
predictions = best_model.predict(X_test)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, predictions)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy with Grid Search:", accuracy)

# # Seleksi fitur dengan SelectFromModel
# selector = SelectFromModel(best_model)
# selector.fit(X_train, y_train)
# X_train_selected = selector.transform(X_train)
# X_test_selected = selector.transform(X_test)

# # Latih ulang model pada fitur yang dipilih
# best_model.fit(X_train_selected, y_train)

# # Prediksi nilai UKT untuk data uji dengan model terbaik pada fitur yang dipilih
# predictions_selected = best_model.predict(X_test_selected)

# # Hitung akurasi prediksi pada fitur yang dipilih
# accuracy_selected = accuracy_score(y_test, predictions_selected)
# print("Accuracy with Feature Selection:", accuracy_selected)

X_df = pd.DataFrame(X, columns=data.drop(columns=['ukt', 'no_test']).columns)
# Menampilkan Feature Importance
importance = best_model.feature_importances_
for i in range(len(importance)):
    print("Feature", X_df.columns[i], ":", importance[i])