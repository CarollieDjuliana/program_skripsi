import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import randint

# Read CSV file
data = pd.read_csv('databaseUkt2023_preprocessed2.csv')
data = data.head(2000)

# Separate features and labels
X = data.drop(columns=['ukt', 'no_test'])
y = data['ukt']


# Calculate the original number of samples in each class
class_counts = y.value_counts()

# Define the desired ratio for each class
sampling_strategy = {
    1: min(class_counts) + 4000,
    2: min(class_counts) + 4000,
    3: min(class_counts) + 4000,
    4: min(class_counts) + 4000,
    5: min(class_counts) + 4000,
    6: min(class_counts) + 4000,
    7: min(class_counts) + 4000,
    8: min(class_counts) + 4000
}

# Handle Imbalanced Data with ADASYN
adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
model = RandomForestClassifier()

# Perform Recursive Feature Elimination
num_features_to_select = 12
rfe = RFE(estimator=model, n_features_to_select=num_features_to_select, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Add Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_rfe)
X_test_poly = poly.transform(X_test_rfe)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Initialize Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameters to be optimized
param_dist = {
    'n_estimators': [350],
    'max_depth': [39],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# Initialize Randomized Search with model, parameter distribution, and evaluation metric
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=skf, scoring='accuracy', random_state=42)

# Train model with Randomized Search to find the best hyperparameter combination
random_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the best model with the best hyperparameters
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train_scaled, y_train)

# Predict UKT values for testing data with the best model
predictions = best_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy with Best Hyperparameters:", accuracy)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, predictions))

# F1-score (Weighted)
f1_weighted = f1_score(y_test, predictions, average='weighted')
print("F1-score (Weighted):", f1_weighted)

# F1-score (Micro)
f1_micro = f1_score(y_test, predictions, average='micro')
print("F1-score (Micro):", f1_micro)
