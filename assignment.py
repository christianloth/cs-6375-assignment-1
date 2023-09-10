# Required Libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

from gradient_descent import gradient_descent, predict


def preprocess():
    # 1. Load the dataset
    data = pd.read_csv("./computer+hardware/machine.data", header=None)

    # Column names based on the provided information and data preview
    column_names = [
        "vendor_name", "model_name", "machine cycle time in nanoseconds", "minimum main memory in kilobytes", "maximum main memory in kilobytes", "cache memory in kilobytes",
        "minimum channels in units", "maximum channels in units", "published relative performance", "estimated relative performance from the original article"
    ]
    data.columns = column_names

    # 2. Remove null or NA values (though it's unlikely for this dataset)
    data.dropna(inplace=True)

    # 3. Remove any redundant rows
    data.drop_duplicates(inplace=True)

    # 4. Remove unnecessary columns
    data_encoded = pd.get_dummies(
        data.drop(["vendor_name", "model_name", "minimum channels in units", "maximum channels in units", "estimated relative performance from the original article"], axis=1))

    # Split data 80/20 (train/test)
    train_size = int(0.8 * len(data_encoded))

    train_data = data_encoded.iloc[:train_size]
    test_data = data_encoded.iloc[train_size:]

    X_train = train_data.drop("published relative performance", axis=1)
    y_train = train_data["published relative performance"]

    X_test = test_data.drop("published relative performance", axis=1)
    y_test = test_data["published relative performance"]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocess()

    # Problem 1
    means = X_train.mean()
    std_devs = X_train.std()

    # Scale the training data
    X_train_scaled = (X_train - means) / std_devs

    # Scale the test data using the mean and standard deviation from the training set
    X_test_scaled = (X_test - means) / std_devs

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
    coef_matrix = reg.coef_  # vector
    intercept = reg.intercept_  # scalar
    r2 = r2_score(y_test, y_pred)

    print("---------------------------- Problem 2 ----------------------------")
    print("Coefficients: ", ['{:.3f}'.format(i) for i in coef_matrix])
    print("Bias: {:.3f}".format(intercept))
    print("MSE: {:.3f}".format(np.mean((y_pred - y_test) ** 2)))
    print("R^2: {:.3f}".format(r2))

    print("\nPredicted Values: ", ['{:.3f}'.format(i) for i in y_pred])
    print("Actual Values: ", ['{:.3f}'.format(i) for i in y_test])
    print("-------------------------------------------------------------------")
