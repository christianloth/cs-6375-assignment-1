import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

import utils
from gradient_descent import gradient_descent, predict

LEARNING_RATE_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
NUM_ITERATIONS_VALUES = [500, 1000, 1500, 2000]


def preprocess():
    # 1. Load the dataset
    data = pd.read_csv("https://raw.githubusercontent.com/christianloth/cs-6375-public-files/main/computer%2Bhardware/machine.data", header=None)

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
    results = []

    # Clear old log file
    with open('log.txt', 'w') as log_file:
        pass

    X_train, y_train, X_test, y_test = preprocess()

    # -------------------- Problem 1 -------------------- #
    means = X_train.mean()
    std_devs = X_train.std()

    # Scale the training data
    X_train_scaled = (X_train - means) / std_devs

    # Scale the test data using the mean and standard deviation from the training set
    X_test_scaled = (X_test - means) / std_devs

    # Add a column of ones for the bias term
    X_train_matrix_with_bias = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_test_matrix_with_bias = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

    # Convert y_train and y_test to from pandas series object to numpy ndarray
    y_train_matrix = y_train.values.reshape(-1, 1)
    y_test_matrix = y_test.values.reshape(-1, 1)

    utils.stdout_output("---------------------------- Problem 1 ----------------------------")
    for learning_rate in LEARNING_RATE_VALUES:
        for num_iterations in NUM_ITERATIONS_VALUES:
            # Train the model using gradient descent
            theta, cost_history = gradient_descent(X_train_matrix_with_bias, y_train_matrix, learning_rate=learning_rate, num_iterations=num_iterations)

            # Predict values for the test set
            y_pred = predict(X_test_matrix_with_bias, theta)

            utils.stdout_output("Learning Rate: {}".format(learning_rate))
            utils.stdout_output("Number of Iterations: {}".format(num_iterations))
            utils.stdout_output()

            utils.stdout_output("Coefficients: ", ['{:.3f}'.format(i) for i in theta.flatten()[1:]])
            utils.stdout_output("Bias: {:.3f}".format(theta[0][0]))
            mse = np.mean((y_pred - y_test_matrix) ** 2)
            utils.stdout_output("MSE: {:.3f}".format(mse))
            #utils.stdout_output("Cost History: ", ['{:.3f}'.format(i) for i in cost_history])

            utils.stdout_output("R^2: {:.3f}".format(1 - (np.sum(np.square(y_test_matrix - y_pred)) / np.sum(np.square(y_test_matrix - y_test_matrix.mean())))))
            utils.stdout_output()

            utils.stdout_output("Predicted Values: ", ['{:.3f}'.format(i) for i in y_pred.flatten()])
            utils.stdout_output("Actual Values: ", ['{:.3f}'.format(i) for i in y_test_matrix.flatten()])
            utils.stdout_output("Error Values: ", ['{:.3f}'.format(i) for i in (y_pred - y_test_matrix).flatten()])
            utils.stdout_output("-------------------------------------------------------------------")
            utils.stdout_output()

            final_cost = cost_history[-1]
            results.append((learning_rate, num_iterations, final_cost, cost_history))
    # ------------------------------------------------------ #



    # -------------------- Problem 2 -------------------- #
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    coef_matrix = reg.coef_  # vector
    intercept = reg.intercept_  # scalar
    r2 = r2_score(y_test, y_pred)

    utils.stdout_output("---------------------------- Problem 2 ----------------------------")
    utils.stdout_output("Coefficients: ", ['{:.3f}'.format(i) for i in coef_matrix])
    utils.stdout_output("Bias: {:.3f}".format(intercept))
    utils.stdout_output("MSE: {:.3f}".format(np.mean((y_pred - y_test) ** 2)))
    utils.stdout_output("R^2: {:.3f}".format(r2))
    utils.stdout_output()

    utils.stdout_output("Predicted Values: ", ['{:.3f}'.format(i) for i in y_pred])
    utils.stdout_output("Actual Values: ", ['{:.3f}'.format(i) for i in y_test])
    utils.stdout_output("Error Values: ", ['{:.3f}'.format(i) for i in (y_pred - y_test)])
    utils.stdout_output("-------------------------------------------------------------------")
    # ------------------------------------------------------ #

    utils.stdout_output('\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

    # Plotting the results
    learning_rates_used = []
    num_iterations_used = []
    final_costs = []

    for lr, ni, cost, _ in results:
        learning_rates_used.append(lr)
        num_iterations_used.append(ni)
        final_costs.append(cost)


    plt.figure(figsize=(10, 6))
    plt.scatter(learning_rates_used, final_costs, c=num_iterations_used, cmap='viridis', s=100)
    plt.colorbar(label='Number of Iterations')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Cost')
    plt.title('Performance of Gradient Descent for Different Hyperparameters')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.show()

    plt.figure(figsize=(15, 10))

    for i, lr in enumerate(LEARNING_RATE_VALUES, start=1):
        result_for_lr = next((r for r in results if r[0] == lr and r[1] == 2000), None)

        _, ni, _, cost_history = result_for_lr

        plt.subplot(2, 3, i)  # Assuming 2 rows and 3 columns for 5 plots
        plt.plot(range(len(cost_history)), [2 * ch for ch in cost_history])  # Multiplying by 2 to get MSE from cost
        plt.xlabel('Number of Iterations')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'MSE for {ni} iterations (lr={lr})')
        plt.grid(True)

    plt.tight_layout()
    plt.show()