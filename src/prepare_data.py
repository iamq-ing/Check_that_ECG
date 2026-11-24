
import argparse
import json

import numpy as np

from pathlib import Path
from collections import Counter
from tqdm import tqdm

from data_loader import (
    find_headers,
    read_labels,
    load_ecg
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/raw"
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=3000
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=20
    )

    args = parser.parse_args()

    root = Path(args.data)

    headers = find_headers(root)

    if not headers:

        raise RuntimeError(
            "No .hea files found in data/raw"
        )

    print("Headers found:", len(headers))

    counts = Counter()

    for header in tqdm(headers):

        counts.update(
            set(read_labels(header))
        )

    selected = [

        code

        for code, count in counts.most_common()

        if count >= args.min_count

    ][:20]

    urgent_codes = [
        "164865005",
        "164947007",
        "27885002"
    ]

    for code in [
        "426177001",
        "427084000"
    ] + urgent_codes:

        if (
            counts[code] >= 10
            and code not in selected
        ):

            selected.append(code)

    selected = sorted(selected)

    print("Selected classes:", len(selected))

    X, Y, ids = [], [], []

    for header in tqdm(
        headers,
        desc="Loading ECG recordings"
    ):

        if len(X) >= args.max_records:
            break

        labels = read_labels(header)

        if not labels:
            continue

        try:

            signal = load_ecg(header)

            target = np.array(
                [
                    int(code in labels)
                    for code in selected
                ],
                dtype=np.float32
            )

            X.append(signal)
            Y.append(target)

            ids.append(
                str(header.relative_to(root))
            )

        except Exception as e:

            print("Skipping:", header, str(e))

    if len(X) < 50:

        raise RuntimeError(
            "Insufficient valid ECG recordings"
        )

    output = Path("data/processed")

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output / "ecg_data.npz",
        X=np.asarray(X, dtype=np.float32),
        Y=np.asarray(Y, dtype=np.float32),
        ids=np.asarray(ids)
    )

    with open(
        output / "labels.json",
        "w"
    ) as f:

        json.dump(
            selected,
            f,
            indent=2
        )

    print("Prepared recordings:", len(X))

    print("Signal shape:", X[0].shape)

    print("Saved processed dataset")


if __name__ == "__main__":

    main()
