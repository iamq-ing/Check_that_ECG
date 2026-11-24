
from pathlib import Path
import re
import numpy as np
import wfdb
from scipy.signal import resample_poly
from fractions import Fraction

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

TARGET_FS = 250
DURATION = 10



def read_labels(header_path):

    labels = []

    with open(
        header_path,
        "r",
        encoding="utf-8-sig",
        errors="ignore"
    ) as f:

        for line in f:

            match = re.match(
                r"^\s*#\s*Dx\s*:\s*(.*)",
                line,
                re.IGNORECASE
            )

            if match:

                value = match.group(1)

                labels = [
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                ]

                break

    return labels




def load_ecg(header_path):

    header_path = Path(header_path)
    base = header_path.with_suffix("")

    record = wfdb.rdrecord(str(base))

    signal = record.p_signal

    if signal is None:
        raise ValueError("Unable to read ECG signal")

    # Normalize lead names
    names = [
        str(x).strip().upper()
        for x in record.sig_name
    ]

    expected_leads = [
        "I", "II", "III", "AVR", "AVL", "AVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    # Check for missing leads
    missing = [
        lead for lead in expected_leads
        if lead not in names
    ]

    if missing:
        raise ValueError(
            f"Missing standard ECG leads: {missing}. "
            f"Available leads: {names}"
        )

    indices = [
        names.index(lead)
        for lead in expected_leads
    ]

    signal = signal[:, indices].T

    fs = float(record.fs)

    if fs != TARGET_FS:

        ratio = Fraction(
            TARGET_FS / fs
        ).limit_denominator(1000)

        signal = resample_poly(
            signal,
            ratio.numerator,
            ratio.denominator,
            axis=1
        )

    target_length = TARGET_FS * DURATION

    if signal.shape[1] > target_length:

        signal = signal[:, :target_length]

    elif signal.shape[1] < target_length:

        signal = np.pad(
            signal,
            ((0, 0), (0, target_length - signal.shape[1]))
        )

    signal = signal.astype(np.float32)

    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)

    signal = (
        signal - mean
    ) / np.maximum(std, 1e-6)

    return signal



def find_headers(root):

    return sorted(
        Path(root).rglob("*.hea")
    )
