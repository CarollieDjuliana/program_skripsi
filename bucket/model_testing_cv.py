import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split, cross_val_score

# 1. Baca dan Persiapkan Data
data = pd.read_csv('_merged_data.csv')

data = data.fillna(0)
X = data.drop(columns=['ukt_rev', 'no_test']).values
y = data['ukt_rev'].values

# 2. Bagi Data menjadi Data Latih dan Data Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Oversampling dengan ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# 4. Inisialisasi Model Random Forest tanpa pembobotan kelas
param_dist = {
    'n_estimators': 300,
    'max_depth': None,
    'random_state': 42
}
rf_model = RandomForestClassifier(**param_dist)

# 5. Evaluasi Model dengan Cross-Validation
cv_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=10)

# Cetak skor validasi silang
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# 6. Melatih model dengan data latih yang di-resample
rf_model.fit(X_resampled, y_resampled)

# 7. Evaluasi Model dengan Data Uji
y_pred_test = rf_model.predict(X_test)

# Confusion Matrix
conf_mat_test = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix on Test Set:")
print(conf_mat_test)

# Akurasi
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Accuracy on Test Set:", accuracy_test)

# Precision
precision = precision_score(y_test, y_pred_test, average='weighted')
print("Precision on Test Set:", precision)

# Recall
recall = recall_score(y_test, y_pred_test, average='weighted')
print("Recall on Test Set:", recall)

# F1-score
f1 = f1_score(y_test, y_pred_test, average='weighted')
print("F1-score on Test Set:", f1)
