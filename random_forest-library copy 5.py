import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
# Read CSV file
data = pd.read_csv('databaseUkt2023_preprocessed2.csv')
data = data.head(1000)

# Separate features and label
X = data.drop(columns=['ukt', 'no_test']).values
y = data['ukt'].values

# Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define hyperparameters to be optimized
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [30, 35],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# Initialize model Random Forest
model = RandomForestClassifier()

# Define number of features to retain
num_features_to_select = 12

# Perform RFE on the untrained Random Forest model
rfe = RFE(estimator=model, n_features_to_select=num_features_to_select, step=1)

# Fit RFE on training data
rfe.fit(X_train, y_train)

# Transform training and testing data with features selected by RFE
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Initialize Grid Search with model, parameter grid, and evaluation metric
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Train model with Grid Search to find the best hyperparameter combination
grid_search.fit(X_train_rfe, y_train)

# Print the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the best model with the best hyperparameters
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train_rfe, y_train)

# Predict UKT values for testing data with the best model
predictions = best_model.predict(X_test_rfe)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy with Best Hyperparameters:", accuracy)
