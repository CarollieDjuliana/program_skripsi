import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import ADASYN


def train_random_forest(X, y, n_estimators, max_depth):
# Setelah membagi X_train dan y_train
    print_class_counts(y)
    # 1. Oversampling dengan ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # 2. Feature Engineering dengan Polynomial Features dan Transformasi Fitur Non-linear
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interact = poly.fit_transform(X_resampled)

    # 3. Inisialisasi dan Latih Model Random Forest
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_interact, y_resampled)

    # 4. Mengembalikan model yang telah dilatih
    return rf_model

def print_class_counts(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_counts_dict = dict(zip(unique_classes, class_counts))

    for class_label, count in class_counts_dict.items():
        print(f"Jumlah sampel pada kelas {class_label}: {count}")