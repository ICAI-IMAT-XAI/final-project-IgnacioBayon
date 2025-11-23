import pandas as pd
import numpy as np
import yfinance as yf
import sklearn.preprocessing


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


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns and realised volatility to the DataFrame."""
    df = df.copy()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    # Multiply by sqrt(252) to annualize the volatility (assuming 252 trading days in a year)
    # Use a rolling window of 21 days (approximately one month)
    df["RealisedVolatility"] = df["LogReturn"].rolling(window=21).std() * np.sqrt(252)
    df.dropna(inplace=True)
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to the DataFrame.
    Indicators added:
    - Relative Strength Index (RSI) for 14 days
    - Bollinger Band Width for 20 days
    - Moving Average Convergence Divergence (MACD)
    """
    df = df.copy()

    # --- 1. RSI: Relative Strength Index ---
    # Captures momentum by comparing recent gains to recent losses
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- 2. Bollinger Band Width ---
    # Measures volatility by calculating the width of the Bollinger Bands
    # Bollinger Bands = SMA ± (2 * standard deviation)
    sma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()
    df["BB_Width"] = (2 * std_20) / sma_20

    # --- 3. MACD: Moving Average Convergence Divergence ---
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26

    return df


def create_sequences(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of features and targets for time series modeling.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        feature_cols (list[str]): List of column names to be used as features.
        target_col (str): Column name to be used as the target.
        seq_length (int): Length of the sequences.

    Returns:
        X (np.ndarray): Array of shape (num_sequences, seq_length, num_features).
        y (np.ndarray): Array of shape (num_sequences,).
    """
    df = df.copy().dropna()

    X_list, y_list = [], []
    for i in range(len(df) - seq_length):
        X_list.append(df.iloc[i : i + seq_length][feature_cols].values)
        y_list.append(df.iloc[i + seq_length][target_col])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def split_sequences(
    X: np.ndarray, y: np.ndarray, train_size: float = 0.7, val_size: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into training, validation, and test sets.

    Args:
        X (np.ndarray): Feature sequences.
        y (np.ndarray): Target values.
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of data to use for validation.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    total_size = len(X)
    train_end = int(total_size * train_size)
    val_end = int(total_size * (train_size + val_size))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset(
    ticker: str,
    feature_cols: list[str],
    target_col: str,
    seq_length: int,
    start_date: str,
    end_date: str | None = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset, compute features, create sequences, and split into train/val/test sets.

    Args:
        ticker (str): Stock ticker symbol.
        feature_cols (list[str]): List of column names to be used as features.
        target_col (str): Column name to be used as the target.
        seq_length (int): Length of the sequences.
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of data to use for validation.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    df = download_data(ticker, start_date, end_date)
    df = compute_log_returns(df)
    df = compute_technical_indicators(df)
    X, y = create_sequences(df, feature_cols, target_col, seq_length)
    # Normalize features
    # X.shape = (num_sequences, seq_length, num_features)
    num_features = X.shape[2]
    for i in range(num_features):
        scaler = sklearn.preprocessing.StandardScaler()
        X[:, :, i] = scaler.fit_transform(X[:, :, i])

    return split_sequences(X, y, train_size, val_size)
