import numpy as np

class LogisticRegression:
    """Logistic regression classifier using gradient descent with sigmoid activation."""
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, random_state: int = 42):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        np.random.seed(random_state)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit logistic regression using gradient descent optimization."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1.0 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using sigmoid function."""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels using 0.5 probability threshold."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self) -> dict:
        """Return model parameters including weights and bias term."""
        return { "weights": self.weights, "intercept": self.bias } 