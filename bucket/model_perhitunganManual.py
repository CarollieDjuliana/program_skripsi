import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


class model:
    @staticmethod
    def train_random_forest(X, y, n_estimators, max_depth):
        # 1. Preprocessing dengan Polynomial Features
        X_interact, feature_names = model.preprocessing(X)

        # 2. Inisialisasi dan Latih Model Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_interact, y)

        # 3. Cetak detail model
        model.print_model_details_to_file(rf_model, feature_names)

        # 4. Evaluasi model
        return rf_model

    @staticmethod
    def preprocessing(X):
        poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False)
        X_interact = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(input_features=X.columns)
        return X_interact, feature_names

    @staticmethod
    def print_model_details_to_file(model, feature_names):
        # Tulis detail Random Forest ke dalam file
        with open('model_details.txt', 'w') as f:
            f.write("Random Forest Details:\n")
            f.write(f"Number of Trees (Estimators): {model.n_estimators}\n")
            f.write(f"Max Depth of Trees: {model.max_depth}\n\n")

            # Tulis detail setiap pohon dalam Random Forest ke dalam file
            f.write("Decision Trees Details:\n")
            for i, tree in enumerate(model.estimators_):
                tree_rules = export_text(tree, feature_names=feature_names)
                f.write(f"Tree {i+1}:\n")
                f.write(tree_rules + '\n\n')

    @staticmethod
    def evaluate_model(model, X, y):
        # Lakukan evaluasi model dengan cross-validation
        print("Model Evaluation:")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("Cross-Validation Scores:", cv_scores)
        print("Mean Accuracy:", np.mean(cv_scores))

        # Hitung dan cetak metrik evaluasi lainnya
        y_pred = model.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        # average='micro' digunakan karena targetnya multiclass
        print("Precision:", precision_score(y, y_pred, average='micro'))
        # average='micro' digunakan karena targetnya multiclass
        print("Recall:", recall_score(y, y_pred, average='micro'))
        # average='micro' digunakan karena targetnya multiclass
        print("F1 Score:", f1_score(y, y_pred, average='micro'))

        # # _class='ovr'
        # print("ROC AUC Score:", roc_auc_score(
        #     y, y_pred, multi_class='ovr', average='macro'))


data = pd.read_csv('after_preprocessing2.csv')
# data = preprocessing_data(data)
# data = data.fillna(0)
X = data.drop(columns=['ukt_rev', 'no_test'])
y = data['ukt_rev']
# Contoh penggunaan:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33)

rf_model = model.train_random_forest(
    X_train, y_train, n_estimators=10, max_depth=5)
# Simpan detail model ke dalam file
X_interact, feature_names = model.preprocessing(X_train)
model.print_model_details_to_file(rf_model, feature_names)

# Evaluasi model menggunakan data test
X_test_processed, _ = model.preprocessing(X_test)
model.evaluate_model(rf_model, X_test_processed, y_test)
