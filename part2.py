import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

import utils
from preprocess import preprocess

if __name__ == "__main__":
    results = []

    X_train, y_train, X_test, y_test = preprocess()

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