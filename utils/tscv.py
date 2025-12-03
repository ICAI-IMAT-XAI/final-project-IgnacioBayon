from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.lstm import LSTMRegressor, train_lstm, evaluate_lstm
from src.data import TimeSeriesDataset


def run_tscv(
    model_class: object,
    model_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
    n_splits: int = 5,
    gap: int = 0,
    verbose: bool = False
):
    """
    Generic Time-Series Cross-Validation for sequence forecasting (RF, XGB, etc.)

    model_class: estimator class, e.g. RandomForestRegressor
    model_params: dict of model params passed to model_class(**model_params)
    X, y: raw unscaled data
    dates: array of datetime-like entries
    seq_len: window size for forming samples
    n_splits: TSCV splits
    gap: optional gap between train and test
    verbose: whether to print fold results
    """

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    mae_scores, rmse_scores = [], []

    print(f"--- Starting TSCV for {model_class.__name__} ---")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):

        # Slice data
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Scale features ONLY using training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # Build sequences
        X_train_seq, y_train_seq = [], []
        for i in range(len(X_train_scaled) - seq_len):
            X_train_seq.append(X_train_scaled[i : i + seq_len].flatten())
            y_train_seq.append(y_train_fold[i + seq_len])

        X_test_seq, y_test_seq = [], []
        for i in range(len(X_test_scaled) - seq_len):
            X_test_seq.append(X_test_scaled[i : i + seq_len].flatten())
            y_test_seq.append(y_test_fold[i + seq_len])

        X_train_seq = np.array(X_train_seq)
        y_train_seq = np.array(y_train_seq)
        X_test_seq = np.array(X_test_seq)
        y_test_seq = np.array(y_test_seq)

        # Instantiate and train model
        model = model_class(**model_params)
        model.fit(X_train_seq, y_train_seq)

        preds = model.predict(X_test_seq)
        mae = mean_absolute_error(y_test_seq, preds)
        rmse = np.sqrt(mean_squared_error(y_test_seq, preds))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        # Date reporting
        test_dates_fold = dates[test_index]
        start_date = pd.to_datetime(test_dates_fold[0]).date()
        end_date = pd.to_datetime(test_dates_fold[-1]).date()

        if verbose:
            print(
                f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, Dates: {start_date} - {end_date}"
            )

    print(f"\nAverage MAE across folds: {np.mean(mae_scores):.4f}")
    print(f"Average RMSE across folds: {np.mean(rmse_scores):.4f}")

    return mae_scores, rmse_scores


def run_fixed_window_tscv(
    model_class: object,
    model_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
    train_size: int,
    test_size: int,
    verbose: bool = False
):
    """
    Walk-forward validation with fixed-size train and test windows.
    Best for volatility forecasting.
    """

    n_samples = len(X)
    mae_scores, rmse_scores = [], []

    fold = 0
    start = 0

    print(f"--- Starting FIXED-WINDOW TSCV for {model_class.__name__} ---")

    while True:
        train_start = start
        train_end = start + train_size
        test_end = train_end + test_size

        if test_end > n_samples:
            break  # no more full folds

        # Slice data
        X_train_fold = X[train_start:train_end]
        y_train_fold = y[train_start:train_end]
        X_test_fold = X[train_end:test_end]
        y_test_fold = y[train_end:test_end]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # Build sequences
        X_train_seq, y_train_seq = [], []
        for i in range(len(X_train_scaled) - seq_len):
            X_train_seq.append(X_train_scaled[i : i + seq_len].flatten())
            y_train_seq.append(y_train_fold[i + seq_len])

        X_test_seq, y_test_seq = [], []
        for i in range(len(X_test_scaled) - seq_len):
            X_test_seq.append(X_test_scaled[i : i + seq_len].flatten())
            y_test_seq.append(y_test_fold[i + seq_len])

        X_train_seq = np.array(X_train_seq)
        y_train_seq = np.array(y_train_seq)
        X_test_seq = np.array(X_test_seq)
        y_test_seq = np.array(y_test_seq)

        # Train
        model = model_class(**model_params)
        model.fit(X_train_seq, y_train_seq)

        preds = model.predict(X_test_seq)
        mae = mean_absolute_error(y_test_seq, preds)
        rmse = np.sqrt(mean_squared_error(y_test_seq, preds))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        if verbose:
            start_date = pd.to_datetime(dates[train_end]).date()
            end_date = pd.to_datetime(dates[test_end - 1]).date()
            print(
                f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, Test Dates: {start_date} - {end_date}"
            )

        fold += 1
        start += test_size  # slide forward

    print(f"\nAverage MAE: {np.mean(mae_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

    return mae_scores, rmse_scores


def build_sequences(X_scaled, y_raw, seq_len):
    """Convert scaled features + raw target into (X_seq, y_seq) for LSTM."""
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i : i + seq_len])
        y_seq.append(y_raw[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


def run_lstm_tscv(
    X,
    y,
    dates,
    seq_len,
    n_splits=5,
    gap=0,
    lstm_params=None,
    batch_size=64,
    num_epochs=50,
    lr=1e-3,
    patience=10,
    val_ratio=0.2,
    verbose=False
):
    """
    Time-series cross-validation for your LSTM using train_lstm / evaluate_lstm.

    X, y: np.ndarray (raw features and target, time-ordered)
    dates: array-like of same length as X for logging
    seq_len: lookback window
    n_splits, gap: passed to TimeSeriesSplit
    lstm_params: dict with arguments for LSTMRegressor (input_size, hidden_size, num_layers, dropout)
    batch_size, num_epochs, lr, patience: training config (passed to train_lstm)
    val_ratio: fraction of *sequence* data used as validation inside each fold
    """

    if lstm_params is None:
        lstm_params = {}

    input_size = X.shape[1]
    if "input_size" not in lstm_params:
        lstm_params["input_size"] = input_size
    if lstm_params.get("num_layers") == 1:
        lstm_params["dropout"] = 0.0  # no dropout if single layer

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    mae_scores, rmse_scores = [], []

    print(f"--- Starting LSTM TSCV with {n_splits} splits ---")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        if verbose:
            print(f"\n=== Fold {fold} ===")

        # 1) Slice raw data
        X_train_raw, X_test_raw = X[train_index], X[test_index]
        y_train_raw, y_test_raw = y[train_index], y[test_index]

        # 2) Scale features using only training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 3) Build LSTM sequences
        X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train_raw, seq_len)
        X_test_seq, y_test_seq = build_sequences(X_test_scaled, y_test_raw, seq_len)

        # Guard in case a fold is too small
        if len(X_train_seq) < 10 or len(X_test_seq) < 1:
            print(f"Fold {fold}: not enough sequences, skipping.")
            continue

        # 4) Train/val split inside the training fold (for early stopping)
        n_train = int(len(X_train_seq) * (1 - val_ratio))
        X_tr, X_val = X_train_seq[:n_train], X_train_seq[n_train:]
        y_tr, y_val = y_train_seq[:n_train], y_train_seq[n_train:]

        # 5) Build DataLoaders
        train_dataset = TimeSeriesDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr).float(),
        )
        val_dataset = TimeSeriesDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
        )
        test_dataset = TimeSeriesDataset(
            torch.from_numpy(X_test_seq).float(),
            torch.from_numpy(y_test_seq).float(),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 6) Create a fresh LSTM model for this fold
        model = LSTMRegressor(**lstm_params)

        # Unique model name per fold so they don't overwrite each other
        model_name = f"lstm_tscv_fold{fold}.pth"

        # 7) Train using your existing helper
        _ = train_lstm(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
            patience=patience,
            model_name=model_name,
        )

        # 8) Evaluate on this fold's test set using your evaluate_lstm
        preds, mae, rmse = evaluate_lstm(model, test_loader, y_true=y_test_seq)

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        # 9) Log dates for sanity
        if verbose:
            test_dates_fold = dates[test_index]
            start_date = pd.to_datetime(test_dates_fold[0]).date()
            end_date = pd.to_datetime(test_dates_fold[-1]).date()
            print(
                f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, Dates: {start_date} - {end_date}"
            )

    print("\n=== LSTM TSCV Summary ===")
    print(f"Average MAE:  {np.mean(mae_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

    return mae_scores, rmse_scores
