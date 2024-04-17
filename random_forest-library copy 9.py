import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import time

start_time = time.time()
# Read CSV file
data = pd.read_csv('databaseUkt2023_preprocessed3.csv')
data = data.head(4000)

# Separate features and labels
X = data.drop(columns=['ukt', 'no_test'])
y = data['ukt']

# One-Hot Encoding
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X).toarray()

# Handle Imbalanced Data with SMOTE and RandomUnderSampler
smote = SMOTE()
rus = RandomUnderSampler()

# Apply SMOTE and RandomUnderSampler to balance the data
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

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

# Train the model with Randomized Search to find the best hyperparameter combination
model = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=5, scoring='accuracy')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model to a .pkl file
joblib.dump(model, 'best_model.pkl')
print("Model telah disimpan sebagai 'best_model.pkl'")

# Save encoder to a .pkl file
joblib.dump(encoder, 'encoder.pkl')
print("Encoder telah disimpan sebagai 'encoder.pkl'")

end_time = time.time()
execution_time = end_time - start_time
print("Total Execution Time:", execution_time, "seconds")