import numpy as np

class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None
        self.history = []  # To store the cost at each iteration

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features + 1)
        X = np.hstack((np.ones((n_samples, 1)), X))
        y = y.reshape(-1, 1)  # Ensure y is a column vector

        for _ in range(self.iterations):
            predictions = self._predict(X).reshape(-1, 1)  # Ensure predictions is a column vector
            errors = predictions - y
            self.history.append(np.mean(errors**2))  # MSE
            for i in range(n_features + 1):
                update_value = self.learning_rate * (1/n_samples) * np.dot(errors.flatten(), X[:, i])
                self.coefficients[i] -= update_value

    def predict(self, X):
        n_samples = X.shape[0]
        X = np.hstack((np.ones((n_samples, 1)), X))
        return self._predict(X).flatten()

    def _predict(self, X):
        return np.dot(X, self.coefficients)

    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)