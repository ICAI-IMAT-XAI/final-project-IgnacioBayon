import numpy as np


def mase(y_train, y_test, y_pred) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    Parameters:
        y_train (array-like): Training target values.
        y_test (array-like): True target values for the test set.
        y_pred (array-like): Predicted target values for the test set.
    Returns:
        float: MASE value.
    """
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    naive_errors = np.abs(y_train[1:] - y_train[:-1])
    naive_mae = np.mean(naive_errors)

    model_errors = np.abs(y_test - y_pred)
    model_mae = np.mean(model_errors)

    return model_mae / naive_mae
