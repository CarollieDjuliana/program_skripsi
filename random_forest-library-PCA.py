import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# 1. Baca dan Persiapkan Data
data = pd.read_csv('databaseUkt2023_preprocessed2.csv')
data = data.head(3000)

# Pisahkan fitur dan label
X = data.drop(columns=['ukt', 'no_test']).values
y = data['ukt'].values

data.reset_index(drop=True, inplace=True)

# 2. Standarisasi Data (Opsional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Menerapkan PCA
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

# 4. Menerapkan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# 5. Print Hasil
result = pd.DataFrame(X_resampled, columns=['PCA Component '+str(i) for i in range(1, 9)])
result['Target'] = y_resampled
print(result)


# 6. Bagi Data menjadi Data Latih dan Data Uji
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [350],
    'max_depth': [39, 41],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# 7. Inisialisasi Model Random Forest dengan Randomized Search CV
rf_model = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, scoring='accuracy', cv=5, random_state=42)
random_search.fit(X_train, y_train)

# 8. Prediksi dengan Model Terbaik
best_rf_model = random_search.best_estimator_
y_pred_test = best_rf_model.predict(X_test)

# 9. Evaluasi dengan Data Uji
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_mat_test = confusion_matrix(y_test, y_pred_test)

print("Best Hyperparameters:", random_search.best_params_)
print("Accuracy on Test Set:", accuracy_test)
print("Confusion Matrix on Test Set:")
print(conf_mat_test)

# Save model to a .pkl file
joblib.dump(best_rf_model, 'best_model_q.pkl')
print("Model telah disimpan sebagai 'best_model.pkl'") 