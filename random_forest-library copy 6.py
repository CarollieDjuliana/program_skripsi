import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read CSV file
data = pd.read_csv('databaseUkt2023_preprocessed2_tanpakoreksi.csv')

# Randomly sample 4000 data points
data = data.sample(n=4000, random_state=42)

# Separate features and labels
X = data.drop(columns=['ukt', 'no_test'])
y = data['ukt']

# Identify outliers in the data
Q1 = X.quantile(0.1)
Q3 = X.quantile(0.9)
IQR = Q3 - Q1
outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)

# Remove outliers from dataset
X_no_outliers = X[~outliers]
y_no_outliers = y[~outliers]

# Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_no_outliers, y_no_outliers)

# Check class distribution after oversampling
class_distribution_resampled = y_resampled.value_counts()

# Display class distribution after oversampling
print("Class Distribution after Oversampling:")
print(class_distribution_resampled)

# Initialize Gradient Boosting Classifier
model = GradientBoostingClassifier()

# Fit the model
model.fit(X_resampled, y_resampled)

# Initialize Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
y_pred = cross_val_predict(model, X_resampled, y_resampled, cv=skf)

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