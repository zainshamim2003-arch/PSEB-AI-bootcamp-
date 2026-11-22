"""Project 1: Little Data, Big Model – Courier Company.

Trains three MLP models on only 200 labelled handwritten digit images:
1) Small linear model (underfitting baseline)
2) Big MLP (can overfit)
3) Big MLP with dropout, weight decay, and early stopping

Run: python train.py
"""
from pathlib import Path
import copy
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 0
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 30
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_data():
    """Return fixed 200 / 798 / 799 train/validation/test split."""
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    # First reserve exactly 200 labelled examples for training.
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, train_size=200, random_state=SEED, stratify=y
    )
    # Split the remaining 1,597 examples into 798 validation and 799 test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining, y_remaining, test_size=799, random_state=SEED,
        stratify=y_remaining
    )
    return (
        torch.tensor(X_train), torch.tensor(y_train),
        torch.tensor(X_val), torch.tensor(y_val),
        torch.tensor(X_test), torch.tensor(y_test),
    )

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(64, 10)
    def forward(self, x):
        return self.net(x)

class BigNet(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(64, 256), nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(256, 256), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(256, 10))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def evaluate_loss(model, X, y, loss_fn, device):
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(X.to(device)), y.to(device)).item())

def train_model(name, model, optimizer, data, loss_fn, device,
                epochs=EPOCHS, patience=None):
    X_train, y_train, X_val, y_val = data[:4]
    model.to(device)
    train_ds = TensorDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Measure both curves in eval mode so dropout is disabled for both.
        train_loss = evaluate_loss(model, X_train, y_train, loss_fn, device)
        val_loss = evaluate_loss(model, X_val, y_val, loss_fn, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save(best_state, OUT / f"best_{name}.pt")
        else:
            wait += 1

        if patience is not None and wait >= patience:
            break

    # Use the best validation checkpoint for final evaluation.
    model.load_state_dict(best_state)
    return model, history, best_val, best_epoch, epoch

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_data()
    X_test, y_test = data[4], data[5]
    loss_fn = nn.CrossEntropyLoss()

    specs = [
        ("small", SmallNet(), "sgd", 0.01, 0.0, None),
        ("big", BigNet(), "adam", 0.001, 0.0, None),
        ("regularised", BigNet(dropout=0.3), "adamw", 0.001, 0.05, PATIENCE),
    ]

    histories = {}
    rows = []
    for name, model, opt_name, lr, decay, patience in specs:
        set_seed(SEED)  # make comparisons repeatable
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=decay
            )

        model, hist, best_val, best_epoch, epochs_run = train_model(
            name, model, optimizer, data, loss_fn, device,
            patience=patience
        )
        histories[name] = hist
        model.eval()
        with torch.no_grad():
            predictions = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
        test_acc = accuracy_score(y_test.numpy(), predictions)
        final_train = hist["train"][-1]
        rows.append({
            "model": name,
            "final_training_loss": final_train,
            "best_validation_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "test_accuracy": test_acc,
        })
        torch.save(model.state_dict(), OUT / f"{name}_final.pt")
        print(f"{name}: best val loss={best_val:.4f} at epoch {best_epoch}; "
              f"test accuracy={test_acc:.4f}; epochs run={epochs_run}")

        if name == "regularised":
            report = classification_report(
                y_test.numpy(), predictions, digits=3, zero_division=0
            )
            (OUT / "classification_report.txt").write_text(report, encoding="utf-8")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(hist["train"], label="Training loss")
        ax.plot(hist["val"], label="Validation loss", linestyle="--")
        best_epoch = int(np.argmin(hist["val"]))
        ax.axvline(best_epoch, linestyle=":", label=f"Best epoch {best_epoch + 1}")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Project 1 – Training and validation loss")
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=160)
    plt.close(fig)

    config = {
        "seed": SEED, "device": str(device), "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "patience": PATIENCE,
        "split": {"train": 200, "validation": 798, "test": 799},
        "input": "64 normalised pixel values (8x8 image flattened)",
        "classes": 10,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nResults:\n", results.to_string(index=False))
    print(f"\nSaved outputs to: {OUT.resolve()}")

if __name__ == "__main__":
    main()
