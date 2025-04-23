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

