
import json
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from src.model import ECGNet
from src.data_loader import load_ecg


st.set_page_config(
    page_title="ECG AI Analyzer",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ AI-Based 12-Lead ECG Classification")

st.caption(
    "PhysioNet Challenge 2020 Research Project"
)

st.warning(
    "Research prototype only. Not intended for "
    "clinical diagnosis or emergency decisions."
)


@st.cache_resource
def load_model():

    with open(
        "data/processed/labels.json"
    ) as f:

        labels = json.load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = ECGNet(
        len(labels)
    ).to(device)

    checkpoint = torch.load(
        "models/ecg_model.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, labels, device


def plot_ecg(signal):

    names = [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    fig, axes = plt.subplots(
        6, 2,
        figsize=(16, 12)
    )

    time = np.arange(
        signal.shape[1]
    ) / 250

    for i, ax in enumerate(axes.flat):

        ax.plot(
            time,
            signal[i],
            color="darkred",
            linewidth=0.7
        )

        ax.set_title(names[i])

        ax.set_xlabel("Time (seconds)")

        ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig


try:

    model, labels, device = load_model()

except Exception as e:

    st.error(
        "Model not found. Please prepare data and "
        "train the model first."
    )

    st.code(str(e))

    st.stop()


st.sidebar.header("ECG Input")

mode = st.sidebar.radio(
    "Choose ECG source",
    [
        "Prepared dataset",
        "Upload ECG files"
    ]
)

signal = None


if mode == "Prepared dataset":

    data_path = Path(
        "data/processed/ecg_data.npz"
    )

    if not data_path.exists():

        st.error("Processed ECG data not found.")

        st.stop()

    data = np.load(data_path)

    sample = st.sidebar.number_input(
        "Sample number",
        min_value=0,
        max_value=len(data["X"]) - 1,
        value=0
    )

    signal = data["X"][sample]

    st.write(
        "Record:", data["ids"][sample]
    )


else:

    hea_file = st.file_uploader(
        "Upload ECG header",
        type=["hea"]
    )

    mat_file = st.file_uploader(
        "Upload ECG signal",
        type=["mat"]
    )

    if hea_file and mat_file:

        with tempfile.TemporaryDirectory() as temp:

            header_path = (
                Path(temp) / hea_file.name
            )

            signal_path = (
                Path(temp) / mat_file.name
            )

            header_path.write_bytes(
                hea_file.getvalue()
            )

            signal_path.write_bytes(
                mat_file.getvalue()
            )

            try:

                signal = load_ecg(header_path)

            except Exception as e:

                st.error(str(e))

                st.stop()

    else:

        st.info(
            "Upload both matching .hea and .mat files."
        )

        st.stop()


if signal is not None:

    st.subheader("12-Lead ECG Waveform")

    st.pyplot(
        plot_ecg(signal)
    )

    x = torch.tensor(
        signal[np.newaxis, :, :],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        probabilities = torch.sigmoid(
            model(x)
        ).cpu().numpy()[0]

    threshold = st.sidebar.slider(
        "Prediction threshold",
        0.1, 0.9, 0.5, 0.05
    )

    selected = np.where(
        probabilities >= threshold
    )[0]

    st.subheader("Predicted Abnormalities")

    if len(selected) == 0:

        st.info(
            "No class exceeded the selected threshold."
        )

    else:

        for i in selected:

            st.write(
                f"**{labels[i]}** — "
                f"{probabilities[i] * 100:.1f}%"
            )

    st.subheader("All Class Probabilities")

    df = pd.DataFrame({

        "Diagnostic code": labels,

        "Probability (%)":
            probabilities * 100

    }).sort_values(
        "Probability (%)",
        ascending=False
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Experimental Referral Screening")

    urgent_codes = {
        "164865005",
        "164947007",
        "27885002"
    }

    urgent_indices = [
        i for i, code in enumerate(labels)
        if code in urgent_codes
    ]

    urgent_flag = any(
        probabilities[i] >= threshold
        for i in urgent_indices
    )

    if urgent_flag:

        st.error(
            "Configured urgent-reference class predicted."
        )

    else:

        st.info(
            "No configured urgent-reference class "
            "crossed the threshold. This does not "
            "rule out an emergency."
        )
