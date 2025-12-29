import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


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


# --- Helper Function for Reconstruction & Evaluation ---
def evaluate_diff_model(preds_diff, y_prev_raw, y_actual_raw, model_name, dates, ticker="TSLA"):
    """
    Reconstructs actual values from predicted differences and calculates metrics.
    
    Args:
        preds_diff: The model's predicted changes (y_t - y_{t-1})
        y_prev_raw: The actual values at t-1 (for reconstruction)
        y_actual_raw: The actual values at t (ground truth)
        model_name: Name for plotting
        dates: Dates for plotting
    """
    # Reconstruction: Pred_t = Actual_{t-1} + Pred_Diff_t
    preds_reconstructed = y_prev_raw + preds_diff
    
    # Calculate Metrics
    mae = mean_absolute_error(y_actual_raw, preds_reconstructed)
    rmse = np.sqrt(mean_squared_error(y_actual_raw, preds_reconstructed))
    
    print(f"--- {model_name} Performance ---")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_actual_raw, label='Actual RV', color='blue', alpha=0.6)
    plt.plot(dates, preds_reconstructed, label=f'{model_name} Prediction', color='red', linestyle='--', alpha=0.8)
    plt.title(f"{model_name}: Reconstructed Predictions vs Actuals ({ticker})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return mae, rmse, preds_reconstructed