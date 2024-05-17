import csv
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import time
from sklearn.metrics import accuracy_score
from preprocessing import preprocessing_data

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

        if len(num_labels) == 1:
            print(f"At depth {depth}, reached leaf node. Predicted class: {
                  num_labels[0]}")
            return {'predicted_class': num_labels[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            print(f"At depth {depth}, reached max depth. Predicted class: {
                  max(num_labels, key=list(y).count)}")
            return {'predicted_class': max(num_labels, key=list(y).count)}

        best_feat, best_thresh = None, None
        best_gini = np.inf
        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            for threshold in unique_values:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_feat, best_thresh = feature, threshold
                    best_gini = gini
                print(f"At depth {depth}, feature {feature}, threshold {
                      threshold}, Gini impurity {gini}")

        if best_feat is None or best_thresh is None:
            print(f"At depth {depth}, reached leaf node. Predicted class: {
                  max(num_labels, key=list(y).count)}")
            return {'predicted_class': max(num_labels, key=list(y).count)}

        print(f"At depth {depth}, selected feature {best_feat} with threshold {
              best_thresh} with Gini impurity {best_gini}")

        left_indices = np.where(X[:, best_feat] <= best_thresh)[0]
        right_indices = np.where(X[:, best_feat] > best_thresh)[0]
        print(f"At depth {depth}, splitting data. Left node: {
              len(left_indices)} samples, Right node: {len(right_indices)} samples")

        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :],
                                y[right_indices], depth + 1)

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
        gini_left = calc_gini(left_labels)
        gini_right = calc_gini(right_labels)
        weight_left, weight_right = len(left_labels) / n, len(right_labels) / n
        gini = weight_left * gini_left + weight_right * gini_right
        return gini


class RandomForest:
    def __init__(self, n_trees=None, max_depth=None, bootstrap_ratio=0.8, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.bootstrap_ratio = bootstrap_ratio
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        rng = np.random.default_rng(
            self.random_state) if self.random_state is not None else np.random.default_rng()

        for _ in range(self.n_trees):
            # Memilih 10 data secara acak dengan penggantian
            bootstrap_indices = rng.choice(len(X), size=15, replace=True)
            print(f"x: {bootstrap_indices}")
            # Mengambil data berdasarkan indeks bootstrap
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            # print(f"x: {X_bootstrap}")

            # Membuat decision tree
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(predictions, axis=0))

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_params(self, deep=True):
        return {'n_trees': self.n_trees, 'max_depth': self.max_depth, 'bootstrap_ratio': self.bootstrap_ratio, 'random_state': self.random_state}


def print_class_counts(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_counts_dict = dict(zip(unique_classes, class_counts))

    for class_label, count in class_counts_dict.items():
        print(f"Jumlah sampel pada kelas {class_label}: {count}")


# Baca file CSV
# data = 'after_preprocessing.csv'
# data = 'data_verifikasi_ukt2023.csv'
# data = preprocessing_data(data)
# data.to_csv('testdata.csv', index=False)
data = pd.read_csv('after_preprocessing.csv')
# data = pd.read_csv('data3_cleaned.csv')
# data = data.head(1000,)
X = data.drop(columns=['ukt'
                       ]).values
y = data['ukt'].values
indices = data.index.values


# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=34)
print_class_counts(y)

# Inisialisasi model Random Forest dengan hyperparameter yang baru
model = RandomForest(n_trees=3, max_depth=6)


# Latih model dengan data latih
model.fit(X_train, y_train)

# Prediksi nilai UKT untuk data uji
predictions = model.predict(X_test)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)


# 1. Hitung confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

correctly_classified_indices = indices_test[y_test == predictions]
correctly_classified_data = data.loc[correctly_classified_indices]
correctly_classified_data.to_csv('correctly_classified_data.csv', index=False)
print("Correctly classified data saved to correctly_classified_data.csv")
