from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import ADASYN
from preprocessing import preprocessing_data


def print_class_counts(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_counts_dict = dict(zip(unique_classes, class_counts))

    for class_label, count in class_counts_dict.items():
        print(f"Jumlah sampel pada kelas {class_label}: {count}")


# 1. Baca dan Persiapkan Data
data = 'data/data_mentah.csv'
data = preprocessing_data(data)
data = data.fillna(0)
X = data.drop(columns=['ukt_rev', 'no_test']).values
y = data['ukt_rev'].values

# 2. Bagi Data menjadi Data Latih dan Data Uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33)
print_class_counts(y)
# 4. Feature Engineering dengan Polynomial Features dan Transformasi Fitur Non-linear
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_interact = poly.fit_transform(X_train)
X_test_interact = poly.transform(X_test)


smote = SMOTE()
X_resampled, y_train_resampled = smote.fit_resample(X_train_interact, y_train)

# # 3. Oversampling dengan ADASYN
# adasyn = ADASYN(random_state=42)
# X_resampled, y_train_resampled = adasyn.fit_resample(X_train_interact, y_train)

# 5. Inisialisasi Model Random Forest tanpa pembobotan kelas
param_dist = {
    'n_estimators': 300,
    'max_depth': None,
    'random_state': 42
}
rf_model = RandomForestClassifier(**param_dist)

# 6. Melatih model dengan data latih yang di-resample
rf_model.fit(X_resampled, y_train_resampled)

# 7. Evaluasi Model dengan Data Uji
y_pred_test = rf_model.predict(X_test_interact)

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

# Membuat DataFrame untuk data yang salah diklasifikasikan
misclassified_data = pd.DataFrame(
    X_test, columns=data.drop(columns=['ukt_rev', 'no_test']).columns)
misclassified_data['true_label'] = y_test
misclassified_data['predicted_label'] = y_pred_test
misclassified_data = misclassified_data[misclassified_data['true_label']
                                        != misclassified_data['predicted_label']]
misclassified_data_targeted = misclassified_data[(
    (misclassified_data['true_label'] == 2) & (misclassified_data['predicted_label'] != 2))]
misclassified_data_targeted.to_csv(
    'misclassified_data_targeted.csv', index=False)
# Menyimpan data yang salah diklasifikasikan ke dalam file CSV
# misclassified_data.to_csv('misclassified_data.csv', index=False)
