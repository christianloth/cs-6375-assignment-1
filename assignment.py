
# Required Libraries
import pandas as pd
import numpy as np

from LinearRegressionMulti import LinearRegressionMulti


def problem1():
    # 1. Load the dataset
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv("adult/adult.data", header=None, names=column_names, sep=r'\s*,\s*', engine='python')

    # 2. Remove null or NA values
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # 3. Remove any redundant rows
    data.drop_duplicates(inplace=True)

    # 4. Convert categorical variables to numerical variables
    data_encoded = pd.get_dummies(data, drop_first=True)  # Using one-hot encoding

    # 5. Attribute Selection (optional)
    #    We can remove attributes based on correlation. For the sake of this example, let's just view the correlation matrix.
    #correlation_matrix = data_encoded.corr()

    # Split data 80/20 (train/test)
    train_size = int(0.8 * len(data_encoded))

    train_data = data_encoded.iloc[:train_size]
    test_data = data_encoded.iloc[train_size:]

    X_train = train_data.drop("income_>50K", axis=1)
    y_train = train_data["income_>50K"]

    X_test = test_data.drop("income_>50K", axis=1)
    y_test = test_data["income_>50K"]

    model = LinearRegressionMulti(learning_rate=0.01, iterations=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = model.mse(y_test, y_pred)
    print(error)


if __name__ == "__main__":
    problem1()

