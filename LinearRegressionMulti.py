import numpy as np


class LinearRegressionMulti:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Fit the model to the data
    def fit(self, X, y):
        # Number of training examples
        m = X.shape[0]

        # Add a column of ones for the bias term
        X_bias = np.c_[np.ones((m, 1)), X]

        # Initialize weights
        self.theta = np.zeros((X_bias.shape[1], 1))

        # Gradient Descent
        for _ in range(self.iterations):
            gradients = 1 / m * X_bias.T.dot(X_bias.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients

    # Predict using the trained model
    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias.dot(self.theta)

    # Compute Mean Squared Error
    def mse(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

# Sample usage:
# model = LinearRegressionMulti(learning_rate=0.01, iterations=1000)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# error = model.mse(y_test, y_pred)

