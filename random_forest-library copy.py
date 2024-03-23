# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# # Baca file CSV
# data = pd.read_csv('databaseUkt2023_preprocessed.csv')
# data = pd.get_dummies(data, columns=['fakultas', 'keberadaan_ayah', 'pekerjaan_ayah', 'keberadaan_ibu', 'pekerjaan_ibu', 'pendapatan_class', 'kepemilikan_rumah', 'kendaraan', 'sekolah', 'listrik'])
# # Pisahkan fitur dan label
# X = data.drop(columns=['ukt', 'no_test']).values
# y = data['ukt'].values

# # Split data menjadi data latih dan data uji
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from sklearn.tree import DecisionTreeClassifier

# # Inisialisasi dan latih model Decision Tree dari scikit-learn
# model = DecisionTreeClassifier(max_depth=10)
# model.fit(X_train, y_train)

# # Prediksi nilai UKT untuk data uji
# predictions = model.predict(X_test)
# # print(predictions)
# # Hitung akurasi prediksi
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: ", accuracy)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# Baca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed2.csv')

# Tentukan fitur kategorikal dan numerik
categorical_features = ['fakultas', 'keberadaan_ayah', 'pekerjaan_ayah', 'keberadaan_ibu', 'pekerjaan_ibu', 'kepemilikan_rumah', 'sekolah','listrik', 'kendaraan']
numeric_features = ['jumlah_tanggungan', 'total_pendapatan', 'penghasilan_ayah', 'penghasilan_ibu']

# Ubah fitur kategorikal menjadi one-hot encoding
data_categorical = pd.get_dummies(data[categorical_features])

# Pisahkan fitur dan label
X = pd.concat([data_categorical, data[numeric_features]], axis=1)
y = data['ukt']


# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Handle Imbalanced Data dengan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Definisikan hyperparameter yang ingin dioptimalkan
param_grid = {
    'max_depth': [10, 16, 14, 20, 30],
    'n_estimators': [100, 120, 130, 1000],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# Inisialisasi model Random Forest
model = RandomForestClassifier(criterion = 'gini')

# Inisialisasi Grid Search dengan model, parameter grid, dan metrik evaluasi
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Latih model dengan Grid Search untuk mencari kombinasi hyperparameter terbaik
grid_search.fit(X_train, y_train)

# Mendapatkan model terbaik setelah Grid Search
best_model = grid_search.best_estimator_

# Seleksi fitur dengan SelectFromModel
selector = SelectFromModel(best_model)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Latih ulang model pada fitur yang dipilih
best_model.fit(X_train_selected, y_train)

# Prediksi nilai UKT untuk data uji dengan model terbaik pada fitur yang dipilih
predictions_selected = best_model.predict(X_test_selected)

# Hitung akurasi prediksi pada fitur yang dipilih
accuracy_selected = accuracy_score(y_test, predictions_selected)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy with Grid Search:", accuracy_selected)
