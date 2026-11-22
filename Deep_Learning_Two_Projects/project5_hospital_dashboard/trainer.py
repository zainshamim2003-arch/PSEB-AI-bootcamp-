"""Training engine for Project 5. Independent of Streamlit."""
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CancerMLP(nn.Module):
    def __init__(self, input_size=30, hidden_size=16, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train(cfg, data, on_epoch=None):
    """Train one run and optionally call on_epoch(epoch, history) each epoch."""
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_val, y_val = data
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    model = CancerMLP(
        input_size=X_train.shape[1],
        hidden_size=int(cfg["hidden_size"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()
    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"]) if cfg["early_stopping"] else max_epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(X_train), y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_eval = float(loss_fn(model(X_train), y_train).item())
            val_eval = float(loss_fn(model(X_val), y_val).item())

        history["train"].append(train_eval)
        history["val"].append(val_eval)

        if val_eval < best_val:
            best_val = val_eval
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if on_epoch is not None:
            on_epoch(epoch, history)

        if cfg["early_stopping"] and wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_val, best_epoch, epoch, device

def diagnose(history):
    """Course-style curve rule, based on the last and best validation losses."""
    train_end = history["train"][-1]
    val_values = history["val"]
    best_val = min(val_values)
    val_end = val_values[-1]

    if train_end > 0.3:
        return "Underfitting"
    if val_end > best_val * 1.30:
        return "Overfitting"
    return "Healthy"
