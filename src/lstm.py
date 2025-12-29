import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import utils.metrics as metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    """
    LSTM Dataset for volatility prediction.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


class Attention(nn.Module):
    """
    Computes a weighted average of the LSTM outputs (all time steps).
    The weights are learned dynamically.
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # A simple linear layer to calculate a score for each time step
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (Batch, Seq_Len, Hidden_Size)

        # 1. Calculate scores for each time step
        # Shape: (Batch, Seq_Len, 1)
        scores = self.attn(lstm_outputs)

        # 2. Normalize scores to probabilities (weights) summing to 1
        # Shape: (Batch, Seq_Len, 1)
        attn_weights = F.softmax(scores, dim=1)

        # 3. Multiply each time step by its weight and sum them up
        # Context shape: (Batch, Hidden_Size)
        context = torch.sum(attn_weights * lstm_outputs, dim=1)

        return context, attn_weights


class LSTMAttentionRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMAttentionRegressor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. LSTM Layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        # 2. Attention Layer
        self.attention = Attention(hidden_size)

        # 3. Fully Connected Layer (Regressor)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)

        # Get output for ALL time steps, not just the last one
        # lstm_out shape: (Batch, Seq_Len, Hidden_Size)
        lstm_out, _ = self.lstm(x)

        # Apply Attention
        # We get the 'context' (weighted summary) and the weights (for visualization)
        context, attn_weights = self.attention(lstm_out)

        # Predict using the context vector
        out = self.fc(context)

        return (
            out.squeeze()
        )  # , attn_weights (Return this if you want to plot the attention!)


def train_lstm(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader = None,
    num_epochs=100,
    lr=1e-3,
    patience=15,
    model_name="best_lstm_model.pth",
):
    model_path = os.path.join("../models", model_name)
    if not os.path.exists("../models"):
        os.makedirs("../models")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")

    early_stopping_counter = 0
    best_val_epoch = 0

    train_loss_history, val_loss_history = [], []
    train_rmse_history, val_rmse_history = [], []
    val_mase_history = []

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        train_loss_history.append(avg_train_loss)
        train_rmse_history.append(np.sqrt(avg_train_loss))

        if val_loader:
            model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)

                    val_losses.append(loss.item())
                    val_preds.extend(outputs.squeeze().numpy())
                    val_targets.extend(y_batch.numpy())

            avg_val_loss = np.mean(val_losses)
            val_loss_history.append(avg_val_loss)
            val_rmse_history.append(np.sqrt(avg_val_loss))

            mase_value = metrics.mase(
                val_targets,
                val_targets,
                val_preds,
            )
            val_mase_history.append(mase_value)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_epoch = epoch + 1
                torch.save(model.state_dict(), model_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1}. Best Val RMSE: {np.sqrt(best_val_loss):.6f}, at epoch {best_val_epoch}."
                    )
                    break

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train RMSE: {train_rmse_history[-1]:.6f} | "
                    f"Val RMSE: {val_rmse_history[-1]:.6f}"
                )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train RMSE: {train_rmse_history[-1]:.6f}"
            )

    # Print it has been saved
    if val_loader:
        print(
            f"\nTraining finished. Best model saved to {model_path}",
            f"\nBest Val RMSE: {np.sqrt(best_val_loss):.6f} at epoch {best_val_epoch}\n",
        )
    else:
        # If no validation, save the final model
        torch.save(model.state_dict(), model_path)
        print(f"\nTraining finished. Model saved to {model_path}")
        val_loss_history = None
        val_rmse_history = None
        val_mase_history = None

    return (
        train_loss_history,
        train_rmse_history,
        val_loss_history,
        val_rmse_history,
        val_mase_history,
    )


def evaluate_lstm(
    model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, y_true: np.ndarray
):
    model.eval()
    preds = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            preds.extend(outputs.squeeze().cpu().numpy())

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    preds = np.array(preds)

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))

    return preds, mae, rmse
