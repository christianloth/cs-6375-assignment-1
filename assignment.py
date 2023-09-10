
# Required Libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from CustomLinearRegression import CustomLinearRegression


def preprocess():
    # 1. Load the dataset
    data = pd.read_csv("./computer+hardware/machine.data", header=None)

    # Column names based on the provided information and data preview
    column_names = [
        "vendor_name", "model_name", "machine cycle time in nanoseconds", "minimum main memory in kilobytes", "maximum main memory in kilobytes", "cache memory in kilobytes", "minimum channels in units", "maximum channels in units", "published relative performance", "estimated relative performance from the original article"
    ]
    data.columns = column_names

    # 2. Remove null or NA values (though it's unlikely for this dataset)
    data.dropna(inplace=True)

    # 3. Remove any redundant rows
    data.drop_duplicates(inplace=True)

    # 4. Convert categorical variables to numerical variables
    # Exclude vendor_name and model_name
    data_encoded = pd.get_dummies(data.drop(["vendor_name", "model_name", "minimum channels in units", "maximum channels in units", "estimated relative performance from the original article"], axis=1))

    # Split data 80/20 (train/test)
    train_size = int(0.8 * len(data_encoded))

    train_data = data_encoded.iloc[:train_size]
    test_data = data_encoded.iloc[train_size:]

    X_train = train_data.drop("published relative performance", axis=1)
    y_train = train_data["published relative performance"]

    X_test = test_data.drop("published relative performance", axis=1)
    y_test = test_data["published relative performance"]

    return X_train, y_train, X_test, y_test


def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Perform gradient descent to optimize linear regression with multiple features.

    Args:
    - X (numpy.ndarray): The feature matrix (m x n), where m is the number of examples and n is the number of features.
    - y (numpy.ndarray): The target vector (m x 1), where m is the number of examples.
    - learning_rate (float): The learning rate for gradient descent.
    - num_iterations (int): The number of iterations for gradient descent.

    Returns:
    - numpy.ndarray: The optimized parameters/coefficients for linear regression.
    - list: The history of cost values for each iteration.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    for _ in range(num_iterations):
        y_pred = X.dot(theta)
        error = y_pred - y
        gradient = (X.T.dot(error)) / m
        theta -= learning_rate * gradient

        cost = np.sum(np.square(error)) / (2 * m)
        cost_history.append(cost)

    return theta, cost_history


def predict(X, theta):
    return X.dot(theta)



if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocess()


    # Problem 1
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Add a column of ones for the bias term
    X_train_bias = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_test_bias = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

    # Convert y_train and y_test to matrix form
    y_train_matrix = y_train.values.reshape(-1, 1)
    y_test_matrix = y_test.values.reshape(-1, 1)

    # Train the model using gradient descent
    theta, cost_history = gradient_descent(X_train_bias, y_train_matrix, learning_rate=0.01, num_iterations=1000)

    # Predict values for the test set
    y_pred = predict(X_test_bias, theta)

    # Calculate R^2 score for the predictions
    r2 = 1 - (np.sum(np.square(y_test_matrix - y_pred)) / np.sum(np.square(y_test_matrix - y_test_matrix.mean())))
    print(r2)
    print("---------------------------- Problem 1 ----------------------------")
    print("Coefficients: ", ['{:.3f}'.format(i) for i in theta.flatten()[1:]])
    print("Bias: {:.3f}".format(theta[0][0]))
    print("MSE: {:.3f}".format(np.mean((y_pred - y_test_matrix) ** 2)))
    print("R^2: {:.3f}".format(r2))
    print("\nPredicted Values: ", ['{:.3f}'.format(i) for i in y_pred.flatten()])
    print("Actual Values: ", ['{:.3f}'.format(i) for i in y_test_matrix.flatten()])
    print("-------------------------------------------------------------------")








    # Problem 2
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    coef_matrix = reg.coef_ # vector
    intercept = reg.intercept_ # scalar
    r2 = r2_score(y_test, y_pred)

    print("---------------------------- Problem 2 ----------------------------")

    print("Coefficients: ", ['{:.3f}'.format(i) for i in coef_matrix])
    print("Bias: {:.3f}".format(intercept))

    print("MSE: {:.3f}".format(np.mean((y_pred - y_test) ** 2)))
    print("R^2: {:.3f}".format(r2))

    print("\nPredicted Values: ", ['{:.3f}'.format(i) for i in y_pred])
    print("Actual Values: ", ['{:.3f}'.format(i) for i in y_test])

    print("-------------------------------------------------------------------")