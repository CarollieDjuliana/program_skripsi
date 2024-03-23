import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel, RFE

# Baca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed2.csv')

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
    # 'n_estimators': [100, 200, 300],
    # 'max_depth': [None, 10, 20, 30],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['auto', 'sqrt', 'log2']
    'n_estimators': [200],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# Inisialisasi model Random Forest
model = RandomForestClassifier()

# Definisikan jumlah fitur yang ingin dipertahankan
num_features_to_select = 7  # Atur jumlah fitur yang ingin dipertahankan

# Lakukan RFE pada model Random Forest yang belum dilatih
rfe = RFE(estimator=model, n_features_to_select=num_features_to_select, step=1)

# Lakukan fit RFE pada data latih
rfe.fit(X_train, y_train)

# Transformasikan data latih dan data uji dengan fitur yang dipilih oleh RFE
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Inisialisasi Grid Search dengan model, parameter grid, dan metrik evaluasi
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Latih model dengan Grid Search untuk mencari kombinasi hyperparameter terbaik
grid_search.fit(X_train_rfe, y_train)

# Menampilkan semua kombinasi hyperparameter yang diuji
print("All Hyperparameter Combinations:")
for params, mean_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    print("Hyperparameters:", params)
    print("Mean Score:", mean_score)
    print()

# Seleksi fitur dengan SelectFromModel
selector = SelectFromModel(grid_search.best_estimator_)
selector.fit(X_train_rfe, y_train)
X_train_selected = selector.transform(X_train_rfe)
X_test_selected = selector.transform(X_test_rfe)

# Latih ulang model pada fitur yang dipilih
grid_search.best_estimator_.fit(X_train_selected, y_train)

# Prediksi nilai UKT untuk data uji dengan model terbaik pada fitur yang dipilih
predictions_selected = grid_search.best_estimator_.predict(X_test_selected)

# Hitung akurasi prediksi pada fitur yang dipilih
accuracy_selected = accuracy_score(y_test, predictions_selected)
print("Accuracy with Feature Selection:", accuracy_selected)

X_df = pd.DataFrame(X, columns=data.drop(columns=['ukt', 'no_test']).columns)

# Menampilkan Feature Importance
importance = grid_search.best_estimator_.feature_importances_
for i in range(len(importance)):
    print("Feature", X_df.columns[i], ":", importance[i])

# Menampilkan fitur yang dipilih oleh RFE
print("Selected Features:")
for feature in X_df.columns[rfe.support_]:
    print(feature)