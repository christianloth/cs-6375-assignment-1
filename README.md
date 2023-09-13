# CS6375 Assignment

This is for the gradient descent assignment for CS6375 by Christian Loth.

## Requirements

- Python 3.11 or higher
- `pipenv` for managing dependencies and virtual environment. This is what I have chosen for my python virtual environment to manage dependencies.

## Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`

## Setup

1. cd into the cs-6375-assignment-1 directory:
    ```bash
    cd cs-6375-assignment-1
    ```
2. Using `pipenv` to manage dependencies and virtual environment:
    ```bash
    pipenv install
    ```

3. Activate the virtual environment:
    ```bash
    pipenv shell
    ```

## Running the Program

After setting up the environment and installing the necessary dependencies:

1. Run the main file:
    ```bash
    python assignment.py
    ```

This will execute the code and generate the required outputs, logs, and plots.

## Outputs

- The program will display various metrics such as the coefficients, bias, MSE, and R^2 for different configurations.
- The program will generate and display plots for analyzing the performance of the gradient descent algorithm.
- A `log.txt` file will be created in the root directory of the project that contains the log of the program execution.
  - I have included a `log.txt` file in the submission. However, it will be overwritten if the program is run again.