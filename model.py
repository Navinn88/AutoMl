import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept: bool = True):
        """Initialize linear regression model with optional intercept."""
        self.fit_intercept = fit_intercept
        self.weights = None
        self.intercept = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to feature matrix."""
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """Fit model using normal equation and return self for chaining."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.fit_intercept:
            X = self._add_intercept(X)
            
        try:
            theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            
            if self.fit_intercept:
                self.intercept = theta[0]
                self.weights = theta[1:]
            else:
                self.intercept = 0.0
                self.weights = theta
                
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error in computing normal equation: {str(e)}")
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values using fitted model."""
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.fit_intercept:
            return np.dot(X, self.weights) + self.intercept
        else:
            return np.dot(X, self.weights)

    def get_params(self) -> dict:
        """Return model parameters including weights and intercept."""
        return {
            'weights': self.weights,
            'intercept': self.intercept,
            'fit_intercept': self.fit_intercept
        } 