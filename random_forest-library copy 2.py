import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

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

# Inisialisasi model Random Forest
base_model = RandomForestClassifier(n_estimators=200, max_depth=20)

# Inisialisasi model AdaBoost dengan base model Random Forest
boosting_model = AdaBoostClassifier(base_model, n_estimators=50, learning_rate=1.0)

# Latih model AdaBoost
boosting_model.fit(X_train, y_train)

# Prediksi nilai UKT untuk data uji dengan model AdaBoost
predictions_boosting = boosting_model.predict(X_test)

# Hitung akurasi prediksi dengan model AdaBoost
accuracy_boosting = accuracy_score(y_test, predictions_boosting)
print("Accuracy with AdaBoost (Random Forest):", accuracy_boosting)