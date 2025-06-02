import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize linear regression model.
        
        Args:
            fit_intercept: Whether to calculate the intercept for this model.
                          If False, no intercept will be used in calculations.
        """
        self.fit_intercept = fit_intercept
        self.weights = None  # Coefficients
        self.intercept = None  # Intercept term

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add a column of ones to X for the intercept term."""
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear regression model using the normal equation.
        
        Args:
            X: Training data, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)
            
        Returns:
            self: Returns the instance itself
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.fit_intercept:
            X = self._add_intercept(X)
            
        # Normal equation: θ = (X^T X)^(-1) X^T y
        # Using numpy's linear algebra solver for better numerical stability
        try:
            # Solve the normal equation using numpy's lstsq
            # This is more numerically stable than computing the inverse directly
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
        """
        Predict using the linear model.
        
        Args:
            X: Samples, shape (n_samples, n_features)
            
        Returns:
            Predicted values, shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.fit_intercept:
            return np.dot(X, self.weights) + self.intercept
        else:
            return np.dot(X, self.weights)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'weights': self.weights,
            'intercept': self.intercept,
            'fit_intercept': self.fit_intercept
        } 