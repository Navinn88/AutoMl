import numpy as np

class DecisionTreeNode:
    """Node in decision tree storing split feature, threshold, and child nodes."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    """Binary decision tree classifier using Gini impurity for splits."""
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Build decision tree by recursively finding best splits using Gini impurity."""
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """Predict class labels by traversing tree from root to leaf nodes."""
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def get_params(self):
        """Return tree parameters including max depth and minimum samples for split."""
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split}

    def _gini(self, y):
        """Calculate Gini impurity for a set of labels."""
        classes = np.unique(y)
        impurity = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    def _best_split(self, X, y):
        """Find best feature and threshold for splitting data using Gini impurity."""
        m, n = X.shape
        best_gini = 1.0
        best_idx, best_thr = None, None
        for idx in range(n):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left = y[X[:, idx] <= thr]
                right = y[X[:, idx] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = (len(left) * self._gini(left) + len(right) * self._gini(right)) / m
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr
        return best_idx, best_thr

    def _build_tree(self, X, y, depth):
        """Recursively build tree by finding best splits until stopping criteria met."""
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._majority_class(y)
            return DecisionTreeNode(value=leaf_value)
        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            leaf_value = self._majority_class(y)
            return DecisionTreeNode(value=leaf_value)
        left_idxs = X[:, feat_idx] <= threshold
        right_idxs = X[:, feat_idx] > threshold
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return DecisionTreeNode(feature_index=feat_idx, threshold=threshold, left=left, right=right)

    def _majority_class(self, y):
        """Return most common class label in a set of labels."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict(self, inputs, node):
        """Recursively traverse tree to predict class for single input."""
        if node.value is not None:
            return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right) 