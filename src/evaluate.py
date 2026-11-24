
import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import (
    f1_score,
    classification_report,
    recall_score
)

from model import ECGNet
from baseline import baseline_predict


def main():

    data = np.load("data/processed/ecg_data.npz")
    X, Y = data["X"], data["Y"]

    with open("data/processed/labels.json") as f:
        labels = json.load(f)

    test_idx = np.load(
        "results/results_test_indices.npy"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ECGNet(len(labels)).to(device)

    checkpoint = torch.load(
        "models/ecg_model.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():

        probabilities = torch.sigmoid(
            model(
                torch.tensor(
                    X[test_idx],
                    dtype=torch.float32
                ).to(device)
            )
        ).cpu().numpy()

    actual = Y[test_idx].astype(int)
    predicted = (probabilities >= 0.5).astype(int)

    baseline = np.array([
        baseline_predict(X[i], labels)
        for i in test_idx
    ])

    rows = []

    for i, code in enumerate(labels):

        rows.append({
            "Class": code,
            "Model_F1": f1_score(
                actual[:, i],
                predicted[:, i],
                zero_division=0
            ),
            "Baseline_F1": f1_score(
                actual[:, i],
                baseline[:, i],
                zero_division=0
            )
        })

    rows.append({
        "Class": "MACRO-F1",
        "Model_F1": f1_score(
            actual, predicted,
            average="macro",
            zero_division=0
        ),
        "Baseline_F1": f1_score(
            actual, baseline,
            average="macro",
            zero_division=0
        )
    })

    Path("results").mkdir(exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/comparison.csv", index=False)

    print("\nF1 comparison:")
    print(df.to_string(index=False))

    # Preliminary urgent referral definition:
    # Positive if any configured urgent code is predicted.
    urgent_codes = {
        "164865005",
        "164947007",
        "27885002"
    }

    urgent_indices = [
        i for i, code in enumerate(labels)
        if code in urgent_codes
    ]

    if urgent_indices:

        actual_urgent = (
            actual[:, urgent_indices].max(axis=1)
        )

        predicted_urgent = (
            predicted[:, urgent_indices].max(axis=1)
        )

        print(
            "\nPreliminary urgent referral recall:",
            recall_score(
                actual_urgent,
                predicted_urgent,
                zero_division=0
            )
        )

    else:
        print("\nNo configured urgent classes were selected.")

    print("\nSaved results/comparison.csv")


if __name__ == "__main__":
    main()
