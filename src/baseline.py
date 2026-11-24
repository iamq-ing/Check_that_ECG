
import numpy as np

from scipy.signal import find_peaks


def estimate_heart_rate(signal, fs=250):

    lead = signal[1]

    peaks, _ = find_peaks(
        lead,
        distance=int(0.30 * fs),
        prominence=max(np.std(lead) * 0.5, 0.1)
    )

    if len(peaks) < 2:
        return np.nan

    rr = np.diff(peaks) / fs

    rr = rr[
        (rr >= 0.30) &
        (rr <= 2.0)
    ]

    if len(rr) == 0:
        return np.nan

    return 60 / np.median(rr)


def baseline_predict(signal, labels):

    prediction = np.zeros(
        len(labels),
        dtype=int
    )

    heart_rate = estimate_heart_rate(signal)

    if np.isnan(heart_rate):
        return prediction

    for i, code in enumerate(labels):

        if code == "426177001":

            prediction[i] = int(
                heart_rate < 60
            )

        elif code == "427084000":

            prediction[i] = int(
                heart_rate > 100
            )

    return prediction
