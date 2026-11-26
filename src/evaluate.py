
import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from model import ECGNet
from baseline import baseline_predict


DATA_PATH = "data/processed/ecg_data.npz"
LABELS_PATH = "data/processed/labels.json"
MODEL_PATH = "models/ecg_model.pt"
TEST_INDICES_PATH = "results/results_test_indices.npy"

OUTPUT_DIR = Path("results")

URGENT_CODES = {
    "164865005",
    "164947007",
    "27885002"
}

THRESHOLD = 0.5
BATCH_SIZE = 32


def load_data():

    print("\nLoading dataset...")

    data = np.load(DATA_PATH)

    X = data["X"]
    Y = data["Y"]

    with open(LABELS_PATH, "r") as f:
        labels = [str(label) for label in json.load(f)]

    test_idx = np.load(TEST_INDICES_PATH)

    print("ECG dataset shape:", X.shape)
    print("Label matrix shape:", Y.shape)
    print("Number of classes:", len(labels))
    print("Number of test recordings:", len(test_idx))

    if X.ndim != 3:
        raise ValueError("Expected X to have 3 dimensions.")

    if Y.ndim != 2:
        raise ValueError("Expected Y to have 2 dimensions.")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y contain different numbers of recordings.")

    if Y.shape[1] != len(labels):
        raise ValueError(
            "Number of labels does not match Y columns."
        )

    if len(test_idx) == 0:
        raise ValueError("Test index file is empty.")

    if np.any(test_idx < 0) or np.any(test_idx >= len(X)):
        raise ValueError("Test indices contain invalid positions.")

    if len(np.unique(test_idx)) != len(test_idx):
        raise ValueError("Test indices contain duplicates.")

    return X, Y, labels, test_idx



def load_model(num_classes, device):

    print("\nLoading trained model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    if checkpoint["num_classes"] != num_classes:
        raise ValueError(
            "Checkpoint class count does not match labels.json."
        )

    model = ECGNet(num_classes).to(device)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    print("Model loaded successfully.")

    return model


def get_predictions(model, X_test, device):

    print("\nGenerating model predictions...")

    probability_batches = []

    with torch.no_grad():

        for start in range(0, len(X_test), BATCH_SIZE):

            batch = X_test[start:start + BATCH_SIZE]

            batch_tensor = torch.tensor(
                batch,
                dtype=torch.float32
            ).to(device)

            logits = model(batch_tensor)

            probabilities = torch.sigmoid(logits)

            probability_batches.append(
                probabilities.cpu().numpy()
            )

    probabilities = np.concatenate(
        probability_batches,
        axis=0
    )

    predicted = (
        probabilities >= THRESHOLD
    ).astype(int)

    return probabilities, predicted


def get_baseline_predictions(X_test, labels):

    print("\nGenerating baseline predictions...")

    baseline = np.asarray([
        baseline_predict(recording, labels)
        for recording in X_test
    ])

    if baseline.shape != (len(X_test), len(labels)):
        raise ValueError(
            "Baseline predictions have an unexpected shape. "
            f"Expected {(len(X_test), len(labels))}, "
            f"got {baseline.shape}."
        )

    return (baseline >= 0.5).astype(int)


def evaluate_classes(actual, predicted, probabilities, baseline, labels):

    print("\n" + "=" * 75)
    print("CLASS-WISE EVALUATION")
    print("=" * 75)

    rows = []

    for i, code in enumerate(labels):

        actual_class = actual[:, i]
        predicted_class = predicted[:, i]
        baseline_class = baseline[:, i]

        model_f1 = f1_score(
            actual_class,
            predicted_class,
            zero_division=0
        )

        baseline_f1 = f1_score(
            actual_class,
            baseline_class,
            zero_division=0
        )

        actual_count = int(actual_class.sum())
        predicted_count = int(predicted_class.sum())

        true_positives = int(
            ((actual_class == 1) & (predicted_class == 1)).sum()
        )

        rows.append({
            "Class": code,
            "Actual_Positive": actual_count,
            "Predicted_Positive": predicted_count,
            "True_Positive": true_positives,
            "Max_Probability": round(
                float(probabilities[:, i].max()), 4
            ),
            "Model_F1": round(float(model_f1), 4),
            "Baseline_F1": round(float(baseline_f1), 4)
        })

    df = pd.DataFrame(rows)

    print(df.to_string(index=False))

    return df

def evaluate_overall(actual, predicted, baseline):

    print("\n" + "=" * 75)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 75)

    metrics = {
        "Metric": [
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Micro Precision",
            "Micro Recall",
            "Micro F1",
            "Exact Match Accuracy"
        ],

        "Model": [
            precision_score(
                actual, predicted,
                average="macro", zero_division=0
            ),
            recall_score(
                actual, predicted,
                average="macro", zero_division=0
            ),
            f1_score(
                actual, predicted,
                average="macro", zero_division=0
            ),
            precision_score(
                actual, predicted,
                average="micro", zero_division=0
            ),
            recall_score(
                actual, predicted,
                average="micro", zero_division=0
            ),
            f1_score(
                actual, predicted,
                average="micro", zero_division=0
            ),
            np.mean(np.all(actual == predicted, axis=1))
        ],

        "Baseline": [
            precision_score(
                actual, baseline,
                average="macro", zero_division=0
            ),
            recall_score(
                actual, baseline,
                average="macro", zero_division=0
            ),
            f1_score(
                actual, baseline,
                average="macro", zero_division=0
            ),
            precision_score(
                actual, baseline,
                average="micro", zero_division=0
            ),
            recall_score(
                actual, baseline,
                average="micro", zero_division=0
            ),
            f1_score(
                actual, baseline,
                average="micro", zero_division=0
            ),
            np.mean(np.all(actual == baseline, axis=1))
        ]
    }

    metrics_df = pd.DataFrame(metrics)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return metrics_df

def print_classification_report(actual, predicted, labels):

    print("\n" + "=" * 75)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 75)

    report = classification_report(
        actual,
        predicted,
        target_names=labels,
        zero_division=0
    )

    print(report)


def evaluate_urgent_referral(actual, predicted, labels):

    print("\n" + "=" * 75)
    print("PRELIMINARY URGENT REFERRAL EVALUATION")
    print("=" * 75)

    urgent_indices = [
        i for i, code in enumerate(labels)
        if code in URGENT_CODES
    ]

    if not urgent_indices:

        print("WARNING: No configured urgent codes found in labels.")
        print("Check URGENT_CODES against labels.json.")

        return None

    found_codes = [
        labels[i] for i in urgent_indices
    ]

    missing_codes = URGENT_CODES - set(labels)

    print("Configured urgent codes found:", found_codes)

    if missing_codes:
        print("Configured codes missing from dataset:", missing_codes)

    # A recording is considered urgent if it has at least
    # one of the configured urgent labels.

    actual_urgent = (
        actual[:, urgent_indices].max(axis=1)
    )

    predicted_urgent = (
        predicted[:, urgent_indices].max(axis=1)
    )

    actual_count = int(actual_urgent.sum())
    predicted_count = int(predicted_urgent.sum())

    correctly_identified = int(
        (
            (actual_urgent == 1) &
            (predicted_urgent == 1)
        ).sum()
    )

    urgent_recall = recall_score(
        actual_urgent,
        predicted_urgent,
        zero_division=0
    )

    urgent_precision = precision_score(
        actual_urgent,
        predicted_urgent,
        zero_division=0
    )

    print("\nActual urgent recordings:", actual_count)
    print("Predicted urgent recordings:", predicted_count)
    print("Correctly identified urgent recordings:", correctly_identified)

    print(f"Urgent referral recall: {urgent_recall:.4f}")
    print(f"Urgent referral precision: {urgent_precision:.4f}")

    if actual_count == 0:
        print(
            "\nWARNING: No urgent recordings were present "
            "in the test set. Recall cannot be meaningfully assessed."
        )

    return {
        "Actual_Urgent": actual_count,
        "Predicted_Urgent": predicted_count,
        "Correctly_Identified": correctly_identified,
        "Urgent_Recall": urgent_recall,
        "Urgent_Precision": urgent_precision
    }



def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load data
    X, Y, labels, test_idx = load_data()

    X_test = X[test_idx]
    actual = Y[test_idx].astype(int)

    if not np.all(np.isin(actual, [0, 1])):
        raise ValueError("Y contains values other than 0 and 1.")

    # Load model
    model = load_model(len(labels), device)

    # Generate predictions
    probabilities, predicted = get_predictions(
        model,
        X_test,
        device
    )

    # Generate baseline predictions
    baseline = get_baseline_predictions(
        X_test,
        labels
    )

    # Class-wise evaluation
    class_df = evaluate_classes(
        actual,
        predicted,
        probabilities,
        baseline,
        labels
    )

    # Overall evaluation
    metrics_df = evaluate_overall(
        actual,
        predicted,
        baseline
    )

    # Detailed classification report
    print_classification_report(
        actual,
        predicted,
        labels
    )

    # Urgent referral evaluation
    urgent_results = evaluate_urgent_referral(
        actual,
        predicted,
        labels
    )

    # Save results
    class_df.to_csv(
        OUTPUT_DIR / "comparison.csv",
        index=False
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "overall_metrics.csv",
        index=False
    )

    if urgent_results is not None:

        pd.DataFrame([urgent_results]).to_csv(
            OUTPUT_DIR / "urgent_referral_metrics.csv",
            index=False
        )

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

    print("Results saved in the results/ folder:")
    print("- comparison.csv")
    print("- overall_metrics.csv")
    print("- urgent_referral_metrics.csv (if applicable)")


if __name__ == "__main__":
    main()
