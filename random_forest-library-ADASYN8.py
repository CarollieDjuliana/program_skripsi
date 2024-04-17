import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from scipy.stats import randint
import time

start_time = time.time()
# Read CSV file
data = pd.read_csv('databaseUkt2023_preprocessed2_tanpakoreksi.csv')
data = data.sample(n=4000, random_state=42)

# Separate features and labels
X = data.drop(columns=['ukt', 'no_test'])
y = data['ukt']

# Calculate the original number of samples in each class
class_counts = y.value_counts()

# Define the desired ratio for each class
sampling_strategy = {
    1: max(class_counts) + 300,
    2: max(class_counts) + 300,
    3: max(class_counts) + 300,
    4: max(class_counts) + 300,
    5: max(class_counts) + 300,
    6: max(class_counts) + 300,
    7: max(class_counts) + 300,
    8: max(class_counts) + 300
}

# Handle Imbalanced Data with ADASYN
adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Initialize Random Forest Classifier
model = RandomForestClassifier()

# Define hyperparameters to be optimized
param_dist = {
    'n_estimators': [350],
    'max_depth': [30, 39],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# Initialize Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize Randomized Search with model, parameter distribution, and evaluation metric
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=skf, scoring='accuracy', random_state=42)

# Train model with Randomized Search to find the best hyperparameter combination
random_search.fit(X_resampled, y_resampled)

# Print the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the best model with the best hyperparameters
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_resampled, y_resampled)

# Perform cross-validation
cv_accuracy = cross_val_score(best_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_accuracy)

# Predict UKT values for testing data with the best model
predictions = best_model.predict(X_resampled)

# Calculate accuracy
accuracy = accuracy_score(y_resampled, predictions)
print("Accuracy with Best Hyperparameters:", accuracy)

# Classification Report
print("Classification Report:")
print(classification_report(y_resampled, predictions))

# Confusion Matrix
confusion_mat = confusion_matrix(y_resampled, predictions)
print("Confusion Matrix:")
print(confusion_mat)

# F1-score (Weighted)
f1_weighted = f1_score(y_resampled, predictions, average='weighted')
print("F1-score (Weighted):", f1_weighted)

# F1-score (Micro)
f1_micro = f1_score(y_resampled, predictions, average='micro')
print("F1-score (Micro):", f1_micro)

# Save model to a .pkl file
joblib.dump(best_model, 'best_model4.pkl')
print("Model telah disimpan sebagai 'best_model.pkl'")

end_time = time.time()
execution_time = end_time - start_time
print("Total Execution Time:", execution_time, "seconds")