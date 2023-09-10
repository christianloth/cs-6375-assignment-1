import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None

    def fit(self, X, y):
        # Number of samples and features
        n_samples, n_features = X.shape

        # Initialize coefficients: one for each feature and one for the intercept
        self.coefficients = np.zeros(n_features + 1)

        # Add a column of ones for the intercept term
        X = np.hstack((np.ones((n_samples, 1)), X))

        # Gradient Descent
        for _ in range(self.iterations):
            predictions = self._predict(X)
            errors = predictions - y

            # Update the coefficients
            for i in range(n_features + 1):
                self.coefficients[i] -= self.learning_rate * (1/n_samples) * np.dot(errors, X[:, i])

    def predict(self, X):
        n_samples = X.shape[0]
        # Add a column of ones for the intercept term
        X = np.hstack((np.ones((n_samples, 1)), X))
        return self._predict(X)

    def _predict(self, X):
        return np.dot(X, self.coefficients)


# Generate some dummy data for testing

# Generate a dataset with 100 samples, where each sample has 3 features.
np.random.seed(42)  # Seed for reproducibility
X_dummy = np.random.rand(100, 3)

# Generate a target variable y as a linear combination of the features with some noise.
true_coefficients = np.array([2.5, -1.5, 3.0])
y_dummy = np.dot(X_dummy, true_coefficients) + 1.5 + np.random.normal(0, 0.5, 100)

# Split the data into training and test sets
split_idx = int(0.8 * len(X_dummy))
X_train_dummy = X_dummy[:split_idx]
y_train_dummy = y_dummy[:split_idx]
X_test_dummy = X_dummy[split_idx:]
y_test_dummy = y_dummy[split_idx:]

# Initialize and train the custom linear regression model
model = SimpleLinearRegression(learning_rate=0.01, iterations=5000)
model.fit(X_train_dummy, y_train_dummy)

# Make predictions
y_pred_dummy = model.predict(X_test_dummy)

# Return the trained coefficients and the first 5 predicted values for verification
print("Coefficients: ", model.coefficients)
print("First 5 predictions: ", y_pred_dummy[:5])
