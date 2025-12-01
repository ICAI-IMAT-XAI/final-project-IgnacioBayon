import pandas as pd
import numpy as np
import yfinance as yf
import math
import sklearn.preprocessing
import typing

import torch
from torch.utils.data import Dataset, DataLoader


class VolatilityDataset(Dataset):
    def __init__(self, X, y, dates):
        """
        Args:
            X (np.ndarray): Feature sequences of shape (num_samples, seq_len, num_features).
            y (np.ndarray): Target values of shape (num_samples,).
            dates (np.ndarray): Dates corresponding to each sample of shape (num_samples,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dates = dates  # keep as numpy array for XAI

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.dates[idx]


class TimeSeriesDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.clone().detach()
        self.y = y.clone().detach().squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def download_data(
    ticker: str, start_date: str = "2000-01-01", end_date: str | None = None
) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data in 'YYYY-MM-DD' format.
        end_date (str | None): End date for data in 'YYYY-MM-DD' format. If None, uses current date.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data. (Close, High, Low, Open, Volume)

    """
    return yf.download(
        ticker, start=start_date, end=end_date, auto_adjust=True
    ).dropna()


def compute_features(df: pd.DataFrame, trading_periods: int = 252) -> pd.DataFrame:
    """
    Compute features for volatility prediction.
    Args:
        df (pd.DataFrame): DataFrame containing historical stock data with 'Open', 'High', 'Low', 'Close' columns.
        trading_periods (int): Number of trading periods in a year (e.g., 252 for daily data).
    Returns:
        pd.DataFrame: DataFrame with additional feature columns.
            1. Log returns
            2. Realised Volatility (target)
            3. Yang-Zhang Volatility (feature)
            4. Close/Open return (intraday log return)
            5. High/Low log return
            6. RSI (14)
            7. Bollinger Band Width (20)
            8. MACD (12, 26)
    """
    df = df.copy()

    # 1. Log returns
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    # 2. Realised Volatility (target)
    df["RealisedVolatility"] = (
        df["LogReturn"].rolling(21).std() * np.sqrt(trading_periods)
    )

    # 3. Yang–Zhang Volatility (feature)
    df["YZVolatility"] = yang_zhang_vol(df, window=21, trading_periods=trading_periods)

    # 4. Close/Open return (intraday log return)
    df["Log_CO"] = np.log(df["Close"] / df["Open"])

    # 5. High/Low log return
    df["Log_HL"] = np.log(df["High"] / df["Low"])

    # 6. RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 7. Bollinger Band Width (20)
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Width"] = (2 * std20) / sma20

    # 8. MACD (12, 26)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    return df.dropna()



def yang_zhang_vol(
    df: pd.DataFrame, window: int = 21, trading_periods: int = 252
) -> pd.Series:
    """
    Yang-Zhang volatility is the combination of the overnight (close-to-open volatility),
    a weighted average of the Rogers-Satchell volatility and the day's open-to-close volatility.
    (Source: https://www.pyquantnews.com/the-pyquant-newsletter/how-to-compute-volatility-6-ways)

    Args:
        df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', 'Close' columns.
        window (int): Rolling window size.
        trading_periods (int): Number of trading periods in a year (e.g., 252 for daily data).
    """
    log_ho = (df["High"] / df["Open"]).apply(np.log)
    log_lo = (df["Low"] / df["Open"]).apply(np.log)
    log_co = (df["Close"] / df["Open"]).apply(np.log)

    log_oc = (df["Open"] / df["Close"].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2

    log_cc = (df["Close"] / df["Close"].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (
        1.0 / (window - 1.0)
    )
    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (
        1.0 / (window - 1.0)
    )
    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(
        np.sqrt
    ) * math.sqrt(trading_periods)

    return result.dropna()


def create_sequences(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of features and targets for time series modeling.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        feature_cols (list[str]): List of column names to be used as features.
        target_col (str): Column name to be used as the target.
        seq_len (int): Length of the sequences.

    Returns:
        X (np.ndarray): Array of shape (num_sequences, seq_len, num_features).
        y (np.ndarray): Array of shape (num_sequences,).
        dates (np.ndarray): Array of shape (num_sequences,).
    """
    df = df.copy().dropna()

    X_list, y_list, date_list = [], [], []
    for i in range(len(df) - seq_len):
        X_list.append(df.iloc[i : i + seq_len][feature_cols].values)
        y_list.append(df.iloc[i + seq_len][target_col])
        date_list.append(df.index[i + seq_len])

    X = np.array(X_list)
    y = np.array(y_list)
    dates = np.array(date_list)

    return X, y, dates


def split_sequences(X, y, dates, train_size=0.7, test_size=0.15):
    """
    Split sequences into training and testing sets
    """
    n = len(X)
    assert len(y) == n
    assert len(dates) == n

    train_end = int(n * train_size)

    # Only train and test splits
    if abs(train_size + test_size - 1) < 1e-6:
        return (
            (X[:train_end], y[:train_end], dates[:train_end]),
            None,
            (X[train_end:], y[train_end:], dates[train_end:]),
        )

    # Train, validation, and test splits
    else:
        val_end = int(n * (train_size + test_size))
        return (
            (X[:train_end], y[:train_end], dates[:train_end]),
            (X[train_end:val_end], y[train_end:val_end], dates[train_end:val_end]),
            (X[val_end:], y[val_end:], dates[val_end:]),
        )


def load_data(
    ticker: str,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    start_date: str,
    end_date: str | None = None,
    train_size: float = 0.7,
    test_size: float = 0.3,
    # batch_size: int = 32,
):
    """
    Load dataset, compute features, create sequences, and split into train/val/test sets.

    Args:
        ticker (str): Stock ticker symbol.
        feature_cols (list[str]): List of column names to be used as features.
        target_col (str): Column name to be used as the target.
        seq_len (int): Length of the sequences.
        train_size (float): Proportion of data to use for training.
        test_size (float): Proportion of data to use for testing.
        batch_size (int): Batch size for DataLoader.
    Returns:
        train_dataset (Dataset): Dataset for training set.
        val_dataset (Dataset | None): Dataset for validation set. If no validation set, returns None.
        test_dataset (Dataset): Dataset for testing set.
    """
    df = download_data(ticker, start_date, end_date)
    df = compute_features(df)

    X, y, dates = create_sequences(df, feature_cols, target_col, seq_len)

    train_split, val_split, test_split = split_sequences(
        X, y, dates, train_size, test_size
    )

    train_dataset = VolatilityDataset(*train_split)
    test_dataset = VolatilityDataset(*test_split)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # If no validation split, return None for val_dataloader
    if val_split is None:
        return train_dataset, None, test_dataset
    # Otherwise construct validation dataloader
    val_dataset = VolatilityDataset(*val_split)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset
