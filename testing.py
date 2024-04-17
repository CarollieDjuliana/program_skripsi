import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import joblib

# Memuat data pengujian
data_test = pd.read_csv('databaseUkt2023_preprocessed2.csv')
data_test = data_test.head(1000)

# Separate features and labels
X_test = data_test.drop(columns=['ukt', 'no_test'])
y_test = data_test['ukt']

# Memuat kembali model dari file pkl
loaded_model = joblib.load('best_model_q.pkl')

# Memuat kembali objek lda dari file pkl (jika telah disimpan sebelumnya)
lda = joblib.load('lda_model.pkl')

# Menerapkan LDA pada data pengujian
X_test_lda = lda.transform(X_test.values)

# Lakukan prediksi dengan model yang dimuat
predictions = loaded_model.predict(X_test_lda)

# Hitung akurasi
accuracy = accuracy_score(y_test, predictions)
print("Accuracy with Loaded Model:", accuracy)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, predictions))

# Confusion Matrix
confusion_mat = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(confusion_mat)

# F1-score (Weighted)
f1_weighted = f1_score(y_test, predictions, average='weighted')
print("F1-score (Weighted):", f1_weighted)

# F1-score (Micro)
f1_micro = f1_score(y_test, predictions, average='micro')
print("F1-score (Micro):", f1_micro)

# Hitung jumlah nilai yang diprediksi dengan benar
correct_predictions = sum(predictions == y_test)
accuracy = correct_predictions / len(y_test)
print("Accuracy with Loaded Model (Manual Calculation):", accuracy)