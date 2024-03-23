import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

start_time = time.time()

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(len(np.unique(y)))]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'predicted_class': predicted_class}

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        best_split = None
        best_gini = 1.0
        n_features = X.shape[1]

        for feature in range(n_features):
            feature_values = np.unique(X[:, feature])
            for threshold in feature_values:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                       continue

                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_split = {'feature': feature, 'threshold': threshold, 
                                      'left_indices': left_indices, 'right_indices': right_indices}
                    best_gini = gini

        if best_gini < 1.0:
                left_subtree = self._grow_tree(X[best_split['left_indices'], :], y[best_split['left_indices']], depth + 1)
                right_subtree = self._grow_tree(X[best_split['right_indices'], :], y[best_split['right_indices']], depth + 1)
                node['left'] = left_subtree
                node['right'] = right_subtree
                node['split_feature'] = best_split['feature']
                node['split_threshold'] = best_split['threshold']
        return node

    def _predict(self, inputs, node=None):
        if node is None:
            node = self.tree

        if 'predicted_class' in node:
            return node['predicted_class']

        if inputs[node['split_feature']] <= node['split_threshold']:
            return self._predict(inputs, node['left'])
        else:
            return self._predict(inputs, node['right'])

    def _gini_impurity(self, left_y, right_y):
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = len(right_y) / (len(left_y) + len(right_y))
        gini_left = 1.0 - sum((np.sum(left_y == c) / len(left_y))**2 for c in np.unique(left_y))
        gini_right = 1.0 - sum((np.sum(right_y == c) / len(right_y))**2 for c in np.unique(right_y))
        gini = p_left * gini_left + p_right * gini_right
        return gini

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, bootstrap_ratio=0.8):
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

# Baca file CSV
data = pd.read_csv('databaseUkt2023_preprocessed.csv')

# Pisahkan fitur dan label
X = data.drop(columns=['ukt']).values
y = data['ukt'].values

# Split data menjadi data latih dan data uji

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model Random Forest
model = RandomForest(n_trees=100, max_depth=10)
model.fit(X_train, y_train) 

# Prediksi nilai UKT untuk data uji
predictions = model.predict(X_test)
print(predictions)

prediction_time = time.time() - start_time
print("Prediction Time:", prediction_time, "seconds")