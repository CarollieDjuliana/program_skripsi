import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
import time
from sklearn.metrics import accuracy_score

start_time = time.time()

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = np.unique(y)

        # if all labels are the same, return a leaf node
        if len(num_labels) == 1:
            return {'predicted_class': num_labels[0]}

        # if max depth is reached, return a leaf node
        if self.max_depth is not None and depth >= self.max_depth:
            return {'predicted_class': max(num_labels, key=list(y).count)}

        best_feat, best_thresh = None, None
        best_gini = np.inf
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_feat, best_thresh = feature, threshold
                    best_gini = gini

        # if no split could reduce gini impurity, return a leaf node
        if best_feat is None or best_thresh is None:
            return {'predicted_class': max(num_labels, key=list(y).count)}

        left_indices = np.where(X[:, best_feat] <= best_thresh)[0]
        right_indices = np.where(X[:, best_feat] > best_thresh)[0]
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        return {'feature': best_feat, 'threshold': best_thresh, 'left': left, 'right': right}

    def _predict(self, inputs, node=None):
        if node is None:
            node = self.tree

        if 'predicted_class' in node:
            return node['predicted_class']

        if inputs[node['feature']] <= node['threshold']:
            return self._predict(inputs, node['left'])
        else:
            return self._predict(inputs, node['right'])

    def _gini_impurity(self, left_labels, right_labels):
        def calc_gini(labels):
            classes, counts = np.unique(labels, return_counts=True)
            proba = counts / len(labels)
            gini = 1 - np.sum(proba**2)
            return gini

        n = len(left_labels) + len(right_labels)
        gini_left, gini_right = calc_gini(left_labels), calc_gini(right_labels)
        gini = (len(left_labels) / n) * gini_left + (len(right_labels) / n) * gini_right
        return gini
    
class RandomForest:
    def __init__(self, n_trees=None, max_depth=None, bootstrap_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            bootstrap_indices = np.random.choice(len(X), size=int(self.bootstrap_ratio * len(X)), replace=True)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(predictions, axis=0))

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_params(self, deep=True):
        return {'n_trees': self.n_trees, 'max_depth': self.max_depth, 'bootstrap_ratio': self.bootstrap_ratio}



# Baca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed.csv')
data = pd.get_dummies(data, columns=['fakultas', 'keberadaan_ayah', 'pekerjaan_ayah', 'keberadaan_ibu', 'pekerjaan_ibu', 'pendapatan_class', 'kepemilikan_rumah', 'kendaraan', 'sekolah', 'listrik'])
# Pisahkan fitur dan label
X = data.drop(columns=['ukt', 'no_test']).values
y = data['ukt'].values

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest dengan hyperparameter yang sudah ditentukan
model = RandomForest(n_trees=100, max_depth=10)

# Latih model dengan data latih
model.fit(X_train, y_train)

# Lakukan validasi silang untuk mengevaluasi kinerja model
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# Tampilkan skor validasi silang
print("Cross Validation Scores:", cv_scores)

# Prediksi nilai UKT untuk data uji
predictions = model.predict(X_test)

# Lakukan cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = cross_val_score(model, X, y, cv=kf)
print("Accuracy:", accuracies)

prediction_time = time.time() - start_time
print("Prediction Time:", prediction_time, "seconds")
