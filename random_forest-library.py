import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib
# Baca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed2.csv')

# Ambil sampel acak sebanyak 2000
data = data.head(2000)

# Pisahkan fitur dan label
X = data.drop(columns=['ukt', 'no_test'])
y = data['ukt']

# Identifikasi outlier pada data
Q1 = X.quantile(0.1)
Q3 = X.quantile(0.9)
IQR = Q3 - Q1
outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)

# Hapus outlier dari dataset
X_no_outliers = X[~outliers]
y_no_outliers = y[~outliers]

# Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_no_outliers, y_no_outliers)

# Inisialisasi LDA
lda = LDA()

# Melatih LDA dan transformasi data
X_lda = lda.fit_transform(X_resampled, y_resampled)

# Inisialisasi Random Forest Classifier
model = RandomForestClassifier()

# Tentukan hyperparameter yang akan dioptimalkan
param_dist = {
    'n_estimators': [350],  
    'max_depth': [39],       
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}
  
# Inisialisasi Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inisialisasi Randomized Search dengan model, distribusi parameter, dan evaluasi metrik
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, scoring='accuracy', random_state=42)

# Latih model dengan Randomized Search untuk mencari kombinasi hyperparameter terbaik
random_search.fit(X_lda, y_resampled)

# Print the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the best model with the best hyperparameters
best_model = RandomForestClassifier(**best_params)

# Perform cross-validation
y_pred = cross_val_predict(best_model, X_lda, y_resampled, cv=skf)

# Calculate accuracy
accuracy = accuracy_score(y_resampled, y_pred)
print("Accuracy with Cross-Validation:", accuracy)

# Classification Report
print("Classification Report:")
print(classification_report(y_resampled, y_pred)) 

# Confusion Matrix
conf_mat = confusion_matrix(y_resampled, y_pred)
print("Confusion Matrix:")
print(conf_mat)


# Save model to a .pkl file
joblib.dump(best_model, 'best_model4.pkl')
print("Model telah disimpan sebagai 'best_model.pkl'")