
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import ECGNet


def main():

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    data = np.load("data/processed/ecg_data.npz")
    X, Y = data["X"], data["Y"]

    if Y.shape[1] == 0:
        raise ValueError("No diagnostic classes found.")

    indices = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42
    )

    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=42
    )

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    np.save("results/train_indices.npy", train_idx)
    np.save("results/val_indices.npy", val_idx)
    np.save("results/results_test_indices.npy", test_idx)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X[train_idx]),
            torch.tensor(Y[train_idx])
        ),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X[val_idx]),
            torch.tensor(Y[val_idx])
        ),
        batch_size=16,
        num_workers=0
    )

    model = ECGNet(Y.shape[1]).to(device)

    positives = Y[train_idx].sum(axis=0)
    negatives = len(train_idx) - positives

    pos_weight = np.clip(
        negatives / np.maximum(positives, 1), 1, 20
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight, dtype=torch.float32
        ).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    best_loss = float("inf")
    patience = 5
    wait = 0

    for epoch in range(30):

        model.train()
        train_loss = 0

        for xb, yb in train_loader:

            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb, yb = xb.to(device), yb.to(device)

                val_loss += criterion(
                    model(xb), yb
                ).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/30 | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss
            wait = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": Y.shape[1]
                },
                "models/ecg_model.pt"
            )

        else:

            wait += 1

            if wait >= patience:
                print("Early stopping.")
                break

    print("Training completed.")
    print("Model saved: models/ecg_model.pt")


if __name__ == "__main__":
    main()

