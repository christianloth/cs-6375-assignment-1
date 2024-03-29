import numpy as np


def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Perform gradient descent with multiple features.

    Args:
    - X (numpy.ndarray): The feature matrix (m x n), where m is the number of examples and n is the number of features.
    - y (numpy.ndarray): The actual values vector (m x 1), where m is the number of examples.
    - learning_rate (float): The learning rate
    - num_iterations (int): The number of iterations

    Returns:
    - numpy.ndarray: The optimized parameters/coefficients for linear regression.
    - list: The history of cost values for each iteration.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))  # (n x 1)
    cost_history = []

    for _ in range(num_iterations):
        y_pred = X.dot(theta)  # (m x 1)
        error = y_pred - y  # (m x 1)
        gradient = (X.T.dot(error)) / m  # (n x 1)
        theta -= learning_rate * gradient  # (n x 1)

        cost = np.sum(np.square(error)) / (2 * m)
        cost_history.append(cost)

    return theta, cost_history


def predict(X, theta):
    return X.dot(theta)
