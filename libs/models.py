import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import t
import numpy as np
import torch.nn.init as init
import copy
import time

class MLPdataset(Dataset):
    """
    PyTorch Dataset class for tabular (non-sequential) data.
    """
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Return features and target as tensors.
        X = self.features[idx]
        y = self.targets[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

class MLPModel(nn.Module):
    """
    Dynamic MLP for regression (or other tasks).
    
    Args:
        input_dim (int):   Number of input features.
        depth (int):       Number of hidden layers.
        width (int or list of int):
                           If int, all hidden layers have this many units.
                           If list, its length must == depth, and each entry
                           defines the size of that hidden layer.
        dropout (float):   Dropout probability (default: 0.0).
        output_dim (int):  Number of outputs (default: 1).
        activation (nn.Module class):
                           Activation to use after each hidden layer
                           (default: nn.ReLU).
    """
    def __init__(self,
                 input_dim: int,
                 depth: int,
                 width,
                 dropout: float = 0.0,
                 output_dim: int = 1,
                 activation: type = nn.ReLU):
        super().__init__()
        
        # normalize width into a list of layer sizes
        if isinstance(width, int):
            widths = [width] * depth
        else:
            if len(width) != depth:
                raise ValueError(f"Length of width list ({len(width)}) must equal depth ({depth})")
            widths = list(width)
        
        layers = []
        in_features = input_dim
        
        # build hidden layers
        for hidden_units in widths:
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation())
            in_features = hidden_units
        
        # final output layer
        layers.append(nn.Linear(in_features, output_dim))
        
        # bundle into a Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def l1_regularization(model, lambda_):
    return lambda_ * sum(p.abs().sum() for p in model.parameters() if p.requires_grad)

def l2_regularization(model, lambda_):
    return lambda_ * sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)

def train_mlp(
    train_dataset,
    val_dataset,
    model,
    criterion,
    epochs: int,
    patience: int,
    print_freq: int,
    device,
    optimizer: torch.optim.Optimizer | None = None,
    learning_rate: float | None = 1e-3,          # used only when optimizer is None
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    batch_size: int | None = None,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    save_path: str = None,
    timing = False, # if True, measure training time
):
    """
    Train a PyTorch MLP model with L1/L2 regularization and early stopping.

    Args:
        train_dataset (Dataset): Training set.
        val_dataset   (Dataset): Validation set.
        model         (nn.Module):          Your MLPModel instance.
        criterion     (callable):           Loss fn (e.g. nn.MSELoss()).
        epochs        (int):                Max epochs.
        patience      (int):                Early-stop patience.
        print_freq    (int):                Print every N epochs.
        device        (torch.device|str):   'cuda' or 'cpu'.
        optimizer     (Optimizer|None):     Pre-built optimizer. If None, an Adam
                                            optimizer is created from model
                                            parameters and ``learning_rate``.
        learning_rate (float|None):         LR for the fallback Adam optimizer.
        lambda_l1     (float):              L1 coefficient.
        lambda_l2     (float):              L2 coefficient.
        batch_size    (int|None):           If None, uses full dataset.
        shuffle_train (bool):               Shuffle train loader.
        shuffle_val   (bool):               Shuffle val loader.
        save_path     (str|None):          If given, saves the model to this path.

    Returns:
        best_model (nn.Module):  Model re-loaded with best weights.
        history    (dict):       {'train_loss': [...], 'val_loss': [...]}
    """

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle_train)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=shuffle_val)

    # build a default optimizer if none was supplied
    if optimizer is None:
        if learning_rate is None:
            raise ValueError("Either provide an optimizer or a learning_rate.")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_state = None
    counter = 0
    history = {'train_loss': [], 'val_loss': []}
    if timing:
        time_ = []
    for epoch in range(1, epochs + 1):
        if timing:
            t0 = time.perf_counter()
        model.to(device).train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss += l1_regularization(model, lambda_l1)
            loss += l2_regularization(model, lambda_l2)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item() * X.size(0)

        val_loss /= len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % print_freq == 0:
            print(f"Epoch {epoch}/{epochs}  "
                  f"- Train Loss: {train_loss:.5E}  "
                  f"- Val Loss: {val_loss:.5E}")
        if timing:
            t1 = time.perf_counter()
            time_.append(t1 - t0)
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best val loss: {best_val_loss:.5E}")

    # save the model
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    if timing:
        return model, history, time_
    else:
        return model, history

def predict_mlp(
    model: torch.nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray = None,
    scaler = None,
    batch_size: int = None,
    device: torch.device = torch.device('cpu'),
) -> np.ndarray:
    """
    Run a trained MLP model on test data and return (optionally inverse-scaled) predictions.

    Args:
        model      (torch.nn.Module): Your trained model (already loaded with best weights).
        x_test     (np.ndarray):      Test features.
        y_test     (np.ndarray, opt): Dummy or real targets (only to satisfy the Dataset interface).
        scaler     (sklearn scaler, opt): If provided, must implement inverse_transform().
        batch_size (int, opt):        DataLoader batch size; if None, uses full dataset.
        device     (torch.device):    “cpu” or “cuda”.

    Returns:
        np.ndarray: 1D array of predictions (inverse-transformed if `scaler` given).
    """
    # build dataset & loader
    dummy_y = np.zeros_like(x_test[:, :1]) if y_test is None else y_test
    test_dataset = MLPdataset(x_test, dummy_y)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device).eval()
    all_preds = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)                # shape: (batch_size, 1)
            preds = outputs.cpu().numpy().flatten()
            all_preds.extend(preds)

    preds_arr = np.array(all_preds)
    if scaler is not None:
        # expects shape (n_samples, 1)
        preds_arr = scaler.inverse_transform(preds_arr.reshape(-1, 1)).flatten()
    return preds_arr

