import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Bio import AlignIO
from io import StringIO, BytesIO

CODE_TO_AA = {
    0: "-", 1: "A", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "K", 10: "L",
    11: "M", 12: "N", 13: "P", 14: "Q", 15: "R", 16: "S", 17: "T", 18: "V", 19: "W", 20: "Y",
    21: "X", 22: "B", 23: "J", 24: "O", 25: "Z",
}

def msa_to_image(alignment_text: str, format: str) -> tuple:
    """
    Converts Multiple Sequence Alignment (MSA) to numerical image data and amino acid array.

    Parameters:
        alignment_text (str): The MSA text in the specified format
        format (str): The format of the MSA text (e.g., 'fasta', 'clustal')

    Returns:
        tuple: (msa_image, msa_letters) - numerical representation and letter representation
    """
    try:
        alignment = AlignIO.read(StringIO(alignment_text), format)
    except Exception as e:
        st.error(f"An error occurred while parsing the MSA alignment: {e}")
        return None, None

    AA_CODES = {
        "-": 0, "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
        "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
        "X": 21, "B": 22, "J": 23, "O": 24, "Z": 25,
    }

    msa_image = np.zeros((len(alignment), alignment.get_alignment_length()), dtype=int)
    msa_letters = np.empty((len(alignment), alignment.get_alignment_length()), dtype=object)

    for i, record in enumerate(alignment):
        for j, aa in enumerate(str(record.seq)):
            code = AA_CODES.get(aa.upper(), 0)
            msa_image[i, j] = code
            msa_letters[i, j] = aa.upper()

    return msa_image, msa_letters


def plot_msa_image(msa_image: np.ndarray, msa_letters: np.ndarray, plot_method: str):
    """
    Plots the Multiple Sequence Alignment (MSA) as a heatmap.

    Parameters:
        msa_image (np.ndarray): Numerical representation of the MSA
        msa_letters (np.ndarray): Letter representation of the MSA
        plot_method (str): The plotting method ('Plotly (Interactive)' or 'Matplotlib (Static)')
    """
    if msa_image is None or msa_letters is None:
        st.error("No MSA image data to plot.")
        return

    if msa_image.size == 0 or msa_letters.size == 0:
        st.error("MSA image data is empty.")
        return

    if msa_image.shape != msa_letters.shape:
        st.error("Mismatch between msa_image and msa_letters dimensions.")
        return

    if plot_method == "Plotly (Interactive)":
        plot_msa_image_plotly(msa_image, msa_letters)
    elif plot_method == "Matplotlib (Static)":
        plot_msa_image_matplotlib(msa_image, msa_letters)
    else:
        st.error("Unsupported plotting method selected.")


def plot_msa_image_plotly(msa_image: np.ndarray, msa_letters: np.ndarray):
    """
    Plots the MSA as an interactive heatmap using Plotly.

    Parameters:
        msa_image (np.ndarray): Numerical representation of the MSA
        msa_letters (np.ndarray): Letter representation of the MSA
    """
    msa_image_list = msa_image.tolist()
    msa_letters_list = msa_letters.tolist()

    hover_text = [
        [f"Position: {x}<br>Sequence: {y}<br>Amino Acid: {aa}"
         for x, aa in enumerate(row, start=1)]
        for y, row in enumerate(msa_letters_list, start=1)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=msa_image_list,
            text=hover_text,
            hoverinfo="text",
            colorscale="Spectral",
            showscale=False
        )
    )

    fig.update_traces(hovertemplate="%{text}<extra></extra>")
    fig.update_layout(
        title="Multiple Sequence Alignment Heatmap (Plotly)",
        xaxis_title="MSA Residue Position",
        yaxis_title="Sequence Number",
        xaxis=dict(ticks='', showticklabels=True),
        yaxis=dict(ticks='', showticklabels=True),
        plot_bgcolor='white',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    try:
        img_bytes = fig.to_image(format="png")
    except Exception as e:
        st.warning("High-resolution image export failed (Kaleido might be missing). Falling back to standard resolution.")

        fig_mpl, ax = plt.subplots(figsize=(20, 7))
        cax = ax.imshow(msa_image, cmap='Spectral', aspect='auto', interpolation='nearest')
        cmap = plt.get_cmap('Spectral', 26)
        ax.set_title("Multiple Sequence Alignment", fontsize=14)
        ax.set_xlabel("MSA Residue Position", fontsize=12)
        ax.set_ylabel("Sequence Number", fontsize=12)

        unique_values = np.unique(msa_image)
        handles = []
        labels = []
        for val in unique_values:
            label = CODE_TO_AA.get(val, "?")
            handles.append(mpatches.Patch(color=cmap(val), label=label))
            labels.append(label)

        ax.legend(handles=handles, labels=labels, title="Amino Acids", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        plt.close(fig_mpl)
        buf.seek(0)
        img_bytes = buf

    st.download_button(
        label="📥 Download MSA Heatmap (PNG)",
        data=img_bytes,
        file_name="msa_heatmap.png",
        mime="image/png"
    )


def plot_msa_image_matplotlib(msa_image: np.ndarray, msa_letters: np.ndarray):
    """
    Plots the MSA as a static heatmap using Matplotlib.

    Parameters:
        msa_image (np.ndarray): Numerical representation of the MSA
        msa_letters (np.ndarray): Letter representation of the MSA
    """
    img_buf = _plot_msa_image_matplotlib_subset(msa_image, msa_letters, 0, 0)
    if img_buf:
        st.download_button(
            label="📥 Download MSA Heatmap (PNG)",
            data=img_buf,
            file_name="msa_heatmap.png",
            mime="image/png"
        )

def _plot_msa_image_matplotlib_subset(msa_image_subset: np.ndarray, msa_letters_subset: np.ndarray, seq_offset: int, res_offset: int):
    """
    Helper function to plot a subset of the MSA using Matplotlib.

    Parameters:
        msa_image_subset (np.ndarray): Subset of the numerical MSA representation
        msa_letters_subset (np.ndarray): Subset of the letter MSA representation
        seq_offset (int): Sequence offset for labeling
        res_offset (int): Residue offset for labeling
    """
    fig, ax = plt.subplots(figsize=(20, 7))
    cax = ax.imshow(msa_image_subset, cmap='Spectral', aspect='auto', interpolation='nearest')
    cmap = plt.get_cmap('Spectral', 26)
    ax.set_title("Multiple Sequence Alignment", fontsize=14)
    ax.set_xlabel("MSA Residue Position", fontsize=12)
    ax.set_ylabel("Sequence Number", fontsize=12)

    unique_values = np.unique(msa_image_subset)
    handles = []
    labels = []
    for val in unique_values:
        label = CODE_TO_AA.get(val, "?")
        handles.append(mpatches.Patch(color=cmap(val), label=label))
        labels.append(label)

    ax.legend(handles=handles, labels=labels, title="Amino Acids", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    st.image(buf, caption="MSA Heatmap", use_container_width=True)
    return buf
