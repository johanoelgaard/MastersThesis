import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import t
import numpy as np


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
    
# class MLPModel(nn.Module):
#     """
#     MLP model for regression.
#     """
#     def __init__(self, input_dim):
#         super(MLPModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(64, 32)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(32, 1)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         return out


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
        output_dim (int):  Number of outputs (default: 1).
        activation (nn.Module class):
                           Activation to use after each hidden layer
                           (default: nn.ReLU).
    """
    def __init__(self,
                 input_dim: int,
                 depth: int,
                 width,
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
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation())
            in_features = hidden_units
        
        # final output layer
        layers.append(nn.Linear(in_features, output_dim))
        
        # bundle into a Sequential
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# regularization functions
def l1_regularization(model, lambda_l1):
    """
    Compute L1 regularization loss.
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())

    # normalize by number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    return lambda_l1 * l1_norm / num_params

def l2_regularization(model, lambda_l2):
    """
    Compute L2 regularization loss.
    """
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())

    # normalize by number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    return lambda_l2 * l2_norm / num_params

import numpy as np
import torch
from torch.utils.data import DataLoader

def train_mlp(
    train_dataset,
    val_dataset,
    model,
    criterion,
    learning_rate: float,
    lambda_l1: float,
    lambda_l2: float,
    epochs: int,
    patience: int,
    print_freq: int,
    device,
    batch_size=None,
    shuffle_train=True,
    shuffle_val=False,
):
    """
    Train a PyTorch MLP model with L1/L2 regularization and early stopping.

    Args:
        train_dataset (torch.utils.data.Dataset): Training set.
        val_dataset   (torch.utils.data.Dataset): Validation set.
        model         (torch.nn.Module):        Your MLPModel instance.
        criterion     (callable):               Loss function (e.g. nn.MSELoss()).
        l1_regularization_fn (callable):        fn(model, λ) → scalar L1 penalty.
        l2_regularization_fn (callable):        fn(model, λ) → scalar L2 penalty.
        lambda_l1     (float):                  L1 coeff.
        lambda_l2     (float):                  L2 coeff.
        learning_rate (float):                  Adam LR.
        epochs        (int):                    Max epochs.
        patience      (int):                    Early‐stop patience.
        print_freq    (int):                    Print every N epochs.
        device        (torch.device or str):    “cuda” or “cpu”.
        batch_size    (int or None):            If None, uses full dataset.
        shuffle_train (bool):                   Shuffle train loader.
        shuffle_val   (bool):                   Shuffle val loader.

    Returns:
        best_model    (torch.nn.Module):        Model re‐loaded with best weights.
        history       (dict):                   {'train_loss': [...], 'val_loss': [...]}
    """
    # reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=shuffle_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    best_state = None
    counter = 0

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs+1):
        model.to(device).train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(X)
            loss = criterion(out, y)
            loss = loss + l1_regularization(model, lambda_l1)
            loss = loss + l2_regularization(model, lambda_l2)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                loss = loss + l1_regularization(model, lambda_l1)
                loss = loss + l2_regularization(model, lambda_l2)
                val_loss += loss.item() * X.size(0)

        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % print_freq == 0:
            print(f"Epoch {epoch}/{epochs}  "
                  f"- Train Loss: {train_loss:.5E}  "
                  f"- Val Loss: {val_loss:.5E}")

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.5E}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
