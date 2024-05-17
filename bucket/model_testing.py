import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from preprocessing import preprocessing_data


def print_class_counts(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_counts_dict = dict(zip(unique_classes, class_counts))

    for class_label, count in class_counts_dict.items():
        print(f"Jumlah sampel pada kelas {class_label}: {count}")


# 1. Baca dan Persiapkan Data
# data = 'data/data_mentah.csv'
data = 'hasil_filter.csv'
data = preprocessing_data(data)
# data = pd.read_csv('after_preprocessing2.csv')
# data = data.head(1000)
data = data.fillna(0)
X = data.drop(columns=['ukt_rev', 'no_test']).values
y = data['ukt_rev'].values

# 2. Bagi Data menjadi Data Latih dan Data Uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=34)
print_class_counts(y)


param_dist = {
    'n_estimators': 45,
    'max_depth': 32,
    'random_state': 42,
    'n_jobs': -1
}
rf_model = RandomForestClassifier(**param_dist)

# 6. Melatih model dengan data latih yang di-resample
rf_model.fit(X_train, y_train)

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
