import pandas as pd


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
