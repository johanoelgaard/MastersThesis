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
    
class MLPModel(nn.Module):
    """
    MLP model for regression.
    """
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# 5. Define Regularization Functions
def l1_regularization(model, lambda_l1):
    """
    Compute L1 regularization loss.
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

def l2_regularization(model, lambda_l2):
    """
    Compute L2 regularization loss.
    """
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return lambda_l2 * l2_norm