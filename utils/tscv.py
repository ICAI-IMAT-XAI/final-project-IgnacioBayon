from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.lstm import LSTMDataset, LSTMRegressor, train_lstm, evaluate_lstm
from src.data import create_sequences


def build_sequences_np(X, y, seq_len) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw/scaled numpy arrays into (X_seq, y_seq)."""
    n = len(X)
    if n <= seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))

    X_seq = np.array([X[i : i + seq_len] for i in range(n - seq_len)], dtype=np.float32)
    y_seq = y[seq_len:].astype(np.float32)

    return X_seq, y_seq


def run_tscv(
    model_class: object,
    model_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
    n_splits: int = 5,
    gap: int = 0,
    verbose: bool = False,
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

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):

        # 1. Raw slicing
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y[train_idx], y[test_idx]

        # 2. Scale using training data only
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        X_test_sc = scaler.transform(X_test_raw)

        # 3. Build sequences (not flattened yet)
        X_train_seq, y_train_seq = build_sequences_np(X_train_sc, y_train_raw, seq_len)
        X_test_seq, y_test_seq = build_sequences_np(X_test_sc, y_test_raw, seq_len)

        # Skip fold if too small
        if len(X_train_seq) < 10 or len(X_test_seq) == 0:
            print(f"Fold {fold} skipped (not enough sequences)")
            continue

        # 4.Flatten sequences for traditional ML models
        X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
        X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

        # 5. Train and predict
        model = model_class(**model_params)
        model.fit(X_train_flat, y_train_seq)
        preds = model.predict(X_test_flat)

        # 6. Evaluate
        mae = mean_absolute_error(y_test_seq, preds)
        rmse = np.sqrt(mean_squared_error(y_test_seq, preds))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        # Date reporting
        if verbose:
            start_date = dates[test_idx][0]
            end_date = dates[test_idx][-1]
            print(
                f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, Dates: {start_date} - {end_date}"
            )

    print(f"\nAverage MAE across {n_splits} folds: {np.mean(mae_scores):.4f}")
    print(f"Average RMSE across {n_splits} folds: {np.mean(rmse_scores):.4f}")

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
    verbose: bool = False,
):
    """
    Walk-forward validation with fixed-size train and test windows.
    Best for volatility forecasting.
    """

    mae_scores, rmse_scores = [], []
    n_samples = len(X)
    fold = 0
    start = 0

    print(f"--- Starting FIXED-WINDOW TSCV for {model_class.__name__} ---")

    while True:

        train_start = start
        train_end = start + train_size
        test_end = train_end + test_size

        if test_end > n_samples:
            break  # no more full folds

        # 1. Fold slices
        X_train_fold = X[train_start:train_end]
        y_train_fold = y[train_start:train_end]
        X_test_fold = X[train_end:test_end]
        y_test_fold = y[train_end:test_end]

        # 2. Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # 3. Build sequences
        X_train_seq, y_train_seq = build_sequences_np(
            X_train_scaled, y_train_fold, seq_len
        )
        X_test_seq, y_test_seq = build_sequences_np(X_test_scaled, y_test_fold, seq_len)

        if len(X_train_seq) < 10 or len(X_test_seq) == 0:
            print(f"Fold {fold} skipped (not enough sequences)")
            start += test_size  # slide forward
            continue

        # 4. Flatten for ML models
        X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
        X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)

        # 5. Train and predict
        model = model_class(**model_params)
        model.fit(X_train_flat, y_train_seq)
        preds = model.predict(X_test_flat)

        # 6. Evaluate
        mae = mean_absolute_error(y_test_seq, preds)
        rmse = np.sqrt(mean_squared_error(y_test_seq, preds))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        if verbose:
            start_date = dates[train_end]
            end_date = dates[test_end - 1]
            print(
                f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, Test Dates: {start_date} - {end_date}"
            )

        fold += 1
        start += test_size  # slide forward

    print(f"\nAverage MAE: {np.mean(mae_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

    return mae_scores, rmse_scores


def run_lstm_tscv(
    X,
    y,
    dates,
    seq_len,
    lstm_params,
    n_splits=5,
    gap=0,
    batch_size=32,
    num_epochs=50,
    lr=1e-3,
    patience=10,
    val_ratio=0.2,
    verbose=False,
):
    """
    Time-series cross-validation for LSTM
    """

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    mae_scores, rmse_scores = [], []

    # Ensure input size is correct
    lstm_params = dict(lstm_params)

    print(f"--- Starting LSTM TSCV with {n_splits} splits ---")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        if verbose:
            print(f"\n=== Fold {fold} ===")

        # 1. Slice raw data
        X_train_raw, X_test_raw = X[train_index], X[test_index]
        y_train_raw, y_test_raw = y[train_index], y[test_index]

        # 2. Scale features using only training data
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        X_test_sc = scaler.transform(X_test_raw)

        # 3. Build LSTM sequences
        X_train_seq, y_train_seq = build_sequences_np(X_train_sc, y_train_raw, seq_len)
        X_test_seq, y_test_seq = build_sequences_np(X_test_sc, y_test_raw, seq_len)

        if len(X_train_seq) < 10 or len(X_test_seq) < 1:
            print(f"Fold {fold}: not enough sequences, skipping.")
            continue

        # 4. Train/val split inside the training fold (for early stopping)
        n_train = int(len(X_train_seq) * (1 - val_ratio))
        X_tr, X_val = X_train_seq[:n_train], X_train_seq[n_train:]
        y_tr, y_val = y_train_seq[:n_train], y_train_seq[n_train:]

        # 5. Build DataLoaders
        train_dataset = LSTMDataset(X_tr, y_tr)
        val_dataset = LSTMDataset(X_val, y_val)
        test_dataset = LSTMDataset(X_test_seq, y_test_seq)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 6. Train a new LSTM model
        model = LSTMRegressor(**lstm_params)

        _ = train_lstm(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
            patience=patience,
            model_name=f"lstm_tscv_fold{fold}.pth",
        )

        # 7. Evaluate on test set
        preds, mae, rmse = evaluate_lstm(model, test_loader, y_true=y_test_seq)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        # 9) Log dates for sanity
        if verbose:
            start_date = dates[test_index][0]
            end_date = dates[test_index][-1]
            print(
                f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, Dates: {start_date} - {end_date}"
            )

    print("\n=== LSTM TSCV Summary ===")
    print(f"Average MAE:  {np.mean(mae_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

    return mae_scores, rmse_scores
