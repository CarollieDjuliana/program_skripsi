import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import ADASYN


def train_random_forest(X, y, n_estimators, max_depth):

    # 2. Feature Engineering dengan Polynomial Features 
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interact = poly.fit_transform(X)
    
    # Oversampling dengan ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_interact, y)


    # 3. Inisialisasi dan Latih Model Random Forest
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_resampled, y_resampled)

    # 4. Mengembalikan model yang telah dilatih
    return rf_model
