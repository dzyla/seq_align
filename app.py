import streamlit as st
from Bio import AlignIO
from Bio import Align
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder
from io import StringIO
import traceback
import numpy as np
import plotly.graph_objects as go
from typing import List
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from io import BytesIO
import os

# Import pyFAMSA
from pyfamsa import Aligner as PyFAMSAAligner, Sequence as PyFAMSASequence

def msa_to_image(alignment_text: str, format: str) -> tuple:
    """
    Converts Multiple Sequence Alignment (MSA) to numerical image data and amino acid array.

    Args:
        alignment_text (str): MSA result as a string.
        format (str): Format of the MSA (e.g., "fasta", "clustal").

    Returns:
        tuple: (msa_image as numpy.ndarray, msa_letters as numpy.ndarray)

    Raises:
        Exception: If any error occurs during conversion.
    """
    try:
        # Parse the alignment text into a MultipleSeqAlignment object
        alignment = AlignIO.read(StringIO(alignment_text), format)
    except Exception as e:
        st.error(f"An error occurred while parsing the MSA alignment: {e}")
        raise e

    AA_CODES = {
        "-": 0,
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "X": 21,
        "B": 22,
        "J": 23,
        "O": 24,
        "Z": 25,
    }

    msa_image = np.zeros((len(alignment), alignment.get_alignment_length()), dtype=int)
    msa_letters = np.empty(
        (len(alignment), alignment.get_alignment_length()), dtype=object
    )

    for i, record in enumerate(alignment):
        for j, aa in enumerate(str(record.seq)):
            code = AA_CODES.get(aa.upper(), 0)
            msa_image[i, j] = code
            msa_letters[i, j] = aa.upper()

    return msa_image, msa_letters


def plot_msa_image(msa_image: np.ndarray, msa_letters: np.ndarray, plot_method: str):
    """
    Plots the Multiple Sequence Alignment (MSA) as a heatmap.
    Offers two plotting methods: Plotly (interactive) and Matplotlib (static and faster for large MSAs).

    Args:
        msa_image (numpy.ndarray): Numerical representation of MSA.
        msa_letters (numpy.ndarray): Array of amino acid letters.
        plot_method (str): The plotting method selected by the user.
    """
    if msa_image is None or msa_letters is None:
        st.error("No MSA image data to plot.")
        return

    # Ensure that msa_image and msa_letters have compatible shapes
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

    Args:
        msa_image (numpy.ndarray): Numerical representation of MSA.
        msa_letters (numpy.ndarray): Array of amino acid letters.
    """
    # Convert msa_image and msa_letters to lists for Plotly
    msa_image_list = msa_image.tolist()
    msa_letters_list = msa_letters.tolist()

    # Create hover text by combining X, Y, and amino acid letter
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
            showscale=False  # Hide the color scale to remove extra traces
        )
    )

    # Update hovertemplate to display X, Y, and Text
    fig.update_traces(
        hovertemplate="%{text}<extra></extra>"  # <extra></extra> removes the trace info
    )

    fig.update_layout(
        title="Multiple Sequence Alignment Heatmap (Plotly)",
        xaxis_title="MSA Residue Position",
        yaxis_title="Sequence Number",
        xaxis=dict(ticks='', showticklabels=True),
        yaxis=dict(ticks='', showticklabels=True),
        plot_bgcolor='white',
        height=600
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_msa_image_matplotlib(msa_image: np.ndarray, msa_letters: np.ndarray):
    """
    Plots the MSA as a static heatmap using Matplotlib.

    Args:
        msa_image (numpy.ndarray): Numerical representation of MSA.
        msa_letters (numpy.ndarray): Array of amino acid letters.
    """
    _plot_msa_image_matplotlib_subset(msa_image, msa_letters, 0, 0)


CODE_TO_AA = {
    0: "-",
    1: "A",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "P",
    14: "Q",
    15: "R",
    16: "S",
    17: "T",
    18: "V",
    19: "W",
    20: "Y",
    21: "X",
    22: "B",
    23: "J",
    24: "O",
    25: "Z",
}


def _plot_msa_image_matplotlib_subset(msa_image_subset: np.ndarray, msa_letters_subset: np.ndarray, seq_offset: int, res_offset: int):
    """
    Helper function to plot a subset of the MSA using Matplotlib.

    Args:
        msa_image_subset (numpy.ndarray): Numerical representation of MSA subset.
        msa_letters_subset (numpy.ndarray): Array of amino acid letters subset.
        seq_offset (int): Starting index of sequences.
        res_offset (int): Starting index of residues.
    """
    # # Define a colormap
    # cmap = ListedColormap([
    #     "#FFFFFF",  # 0: '-'
    #     "#1f77b4",  # 1: 'A'
    #     "#ff7f0e",  # 2: 'C'
    #     "#2ca02c",  # 3: 'D'
    #     "#d62728",  # 4: 'E'
    #     "#9467bd",  # 5: 'F'
    #     "#8c564b",  # 6: 'G'
    #     "#e377c2",  # 7: 'H'
    #     "#7f7f7f",  # 8: 'I'
    #     "#bcbd22",  # 9: 'K'
    #     "#17becf",  # 10: 'L'
    #     "#aec7e8",  # 11: 'M'
    #     "#ffbb78",  # 12: 'N'
    #     "#98df8a",  # 13: 'P'
    #     "#ff9896",  # 14: 'Q'
    #     "#c5b0d5",  # 15: 'R'
    #     "#c49c94",  # 16: 'S'
    #     "#f7b6d2",  # 17: 'T'
    #     "#c7c7c7",  # 18: 'V'
    #     "#dbdb8d",  # 19: 'W'
    #     "#9edae5",  # 20: 'Y'
    #     "#bcbd22",  # 21: 'X'
    #     "#17becf",  # 22: 'B'
    #     "#ff7f0e",  # 23: 'J'
    #     "#2ca02c",  # 24: 'O'
    #     "#d62728",  # 25: 'Z'
    # ])

    # Create a figure and axis
    # make the plot scale properly based on the image size, taking under consideration the shape of the image

    fig, ax = plt.subplots(figsize=(20,7))
    
    # Display the heatmap
    cax = ax.imshow(msa_image_subset, cmap='Spectral', aspect='auto', interpolation='nearest')

    cmap = plt.get_cmap('Spectral', 26)
    # Set labels
    ax.set_title("Multiple Sequence Alignment", fontsize=14)
    ax.set_xlabel("MSA Residue Position", fontsize=12)
    ax.set_ylabel("Sequence Number", fontsize=12)

    # Create a legend for amino acids
    unique_values = np.unique(msa_image_subset)
    handles = []
    labels = []
    for val in unique_values:
        if val in CODE_TO_AA:
            label = CODE_TO_AA[val]
        else:
            label = "?"
        handles.append(mpatches.Patch(color=cmap(val), label=label))
        labels.append(label)
    
    # Add legend outside the plot
    ax.legend(handles=handles, labels=labels, title="Amino Acids", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    # Convert the Matplotlib figure to a PNG image in memory
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    img = buf.getvalue()

    # Display the image in Streamlit
    st.image(img, caption="MSA Heatmap", use_column_width=True)


def main():
    st.set_page_config(page_title="Advanced Sequence Alignment Tool", layout="wide")
    st.title("🔬 Advanced Sequence Alignment and Format Conversion Tool")
    st.write("""
    ### Welcome to the Advanced Sequence Alignment Tool!

    This application allows you to perform **pairwise** and **multiple sequence alignments (MSA)** on **DNA** or **Protein** sequences. 
    You can choose between **Global**, **Local**, and **Overlap** alignments. Additionally, you can convert sequence files between various formats.
    For MSA, point mutations relative to a reference sequence are reported.

    **How to Use:**
    1. **Upload your sequences** either by uploading a file or pasting them directly.
    2. **Select your preferences** from the sidebar.
    3. **Run** the desired analysis.
    4. **Download** your results!
    """)

    # Sidebar for user inputs
    st.sidebar.header("📥 Upload and Settings")

    # Sequence Format Selection for Upload
    input_format = st.sidebar.selectbox(
        "Select Input Format",
        ("Text (FASTA)", "FASTA", "Clustal", "Phylip", "EMBL", "GenBank", "Newick", "PDB", "mmCIF"),  # Added "PDB" and "mmCIF"
        help="Choose the format of your input sequences."
    )

    sequences = None  # Initialize sequences
    tree = None       # Initialize tree
    uploaded_file = None  # To hold the uploaded file for naming purposes

    if input_format == "Text (FASTA)":
        # Use text_area for multi-line input
        seq_text = st.sidebar.text_area(
            "Paste Sequences Here (FASTA Format)",
            height=250,
            placeholder=">Sequence1\nATGCGTA...\n>Sequence2\nATGCGTC..."
        )
        if seq_text:
            sequences, error = parse_sequences_from_text(seq_text)
            if error:
                st.sidebar.error(error)
            else:
                st.sidebar.success(f"Successfully loaded {len(sequences)} sequences.")
    elif input_format in ["PDB", "mmCIF"]:  # Handle PDB and mmCIF formats
        # Handle PDB/mmCIF file uploads
        uploaded_file = st.sidebar.file_uploader("Upload PDB/mmCIF File", type=get_file_extensions(input_format))
        if uploaded_file:
            try:
                sequences, error = parse_sequences_from_structure(uploaded_file, input_format)
                if error:
                    st.sidebar.error(error)
                else:
                    st.sidebar.success(f"Successfully extracted {len(sequences)} sequences from the uploaded file.")
            except Exception as e:
                st.sidebar.error(f"Error parsing {input_format} file: {e}")
    elif input_format == "Newick":
        # Newick is for trees, handle separately
        uploaded_file = st.sidebar.file_uploader("Upload Newick Tree File", type=["nwk", "newick"])
        if uploaded_file:
            try:
                tree = Phylo.read(uploaded_file, "newick")
                st.sidebar.success("Successfully loaded Newick tree.")
            except Exception as e:
                st.sidebar.error(f"Error reading Newick file: {e}")
    else:
        # Other sequence file formats
        uploaded_file = st.sidebar.file_uploader("Upload Sequence File", type=get_file_extensions(input_format))
        if uploaded_file:
            try:
                sequences, error = parse_sequences_from_file(uploaded_file, input_format)
                if error:
                    st.sidebar.error(error)
                else:
                    st.sidebar.success(f"Successfully loaded {len(sequences)} sequences.")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")

    # Proceed only if sequences are loaded or a tree is uploaded
    if (sequences and len(sequences) > 0) or input_format == "Newick":
        # Sequence Type Selection
        if sequences:
            seq_type = st.sidebar.selectbox("🔬 Select Sequence Type", ("DNA", "Protein"), help="Choose whether your sequences are DNA or Protein.", index=1)
        else:
            seq_type = None  # Not applicable for tree

        # Alignment Mode Selection (Only for non-PDB/mmCIF formats)
        if sequences and input_format not in ["PDB", "mmCIF"]:
            alignment_mode = st.sidebar.selectbox("🛠️ Select Alignment Mode", ("Pairwise", "MSA", "Convert Formats"), help="Choose the type of analysis you want to perform.")
        elif sequences and input_format in ["PDB", "mmCIF"]:
            alignment_mode = None  # No alignment modes for PDB/mmCIF
        else:
            alignment_mode = "Phylogenetic Tree"  # For tree uploads

        # Implement functionalities based on alignment mode
        if sequences:
            if input_format in ["PDB", "mmCIF"]:
                # **New Functionality: Display Extracted FASTA Sequences and Provide Download Option**
                st.header("📄 Extracted FASTA Sequences")

                # Convert sequences to FASTA-formatted string
                fasta_io = StringIO()
                SeqIO.write(sequences, fasta_io, "fasta")
                fasta_str = fasta_io.getvalue()

                # Display the FASTA sequences in a read-only text area
                st.text_area("FASTA Sequences", value=fasta_str, height=500)

                # Provide a download button for the FASTA sequences
                file_basename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                st.download_button(
                    label="📥 Download FASTA",
                    data=fasta_str,
                    file_name=f"{file_basename}_sequences.fasta",
                    mime="text/plain"
                )
            else:
                if alignment_mode == "Pairwise":
                    pairwise_alignment_section(sequences, seq_type)
                elif alignment_mode == "MSA":
                    msa_section(sequences, seq_type)
                elif alignment_mode == "Convert Formats":
                    format_conversion_section(sequences, input_format)
        else:
            # Handle Phylogenetic Tree display if a tree was uploaded
            if alignment_mode == "Phylogenetic Tree":
                phylogenetic_tree_section(tree)
    else:
        st.info("Please upload sequences or a Newick tree in the sidebar to begin.")


def get_file_extensions(format_name):
    """Return appropriate file extensions based on the selected format."""
    format_extensions = {
        "FASTA": ["fasta", "fa", "fna", "ffn", "faa", "frn"],
        "Clustal": ["clustal", "aln"],
        "Phylip": ["phy"],
        "EMBL": ["embl"],
        "GenBank": ["gb", "genbank"],
        "Newick": ["nwk", "newick"],
        "PDB": ["pdb"],
        "mmCIF": ["cif", "mmcif"]
    }
    return format_extensions.get(format_name, [])


def parse_sequences_from_text(text):
    """
    Parse sequences from pasted text in FASTA format, handling multiple newlines and blank lines.

    Args:
        text (str): The raw text input containing FASTA-formatted sequences.

    Returns:
        tuple:
            sequences (list of SeqRecord): Parsed sequences or None if parsing fails.
            error (str): Error message if parsing fails, otherwise None.
    """
    try:
        # Normalize line endings to '\n'
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Split the text into lines and remove any lines that are only whitespace
        lines = [line.strip() for line in normalized_text.split('\n') if line.strip() != '']

        # Reconstruct the text with single newlines between lines
        cleaned_text = '\n'.join(lines)

        # Ensure the cleaned text starts with '>'
        if not cleaned_text.startswith('>'):
            return None, "FASTA format should start with '>'. Please check your input."

        # Use StringIO to emulate a file-like object for SeqIO.parse
        seq_io = StringIO(cleaned_text)
        sequences = list(SeqIO.parse(seq_io, "fasta"))

        if not sequences:
            return None, "No valid FASTA sequences found. Please check your input."

        # Additional validation: Ensure each sequence has a unique ID and non-empty sequence
        sequence_ids = set()
        for seq in sequences:
            if seq.id in sequence_ids:
                return None, f"Duplicate sequence ID found: '{seq.id}'. Each sequence must have a unique ID."
            sequence_ids.add(seq.id)
            if not str(seq.seq).strip():
                return None, f"Sequence '{seq.id}' is empty. Please provide valid sequences."

        return sequences, None

    except Exception as e:
        return None, f"An error occurred while parsing the text input: {e}"


def parse_sequences_from_file(file, format_name):
    """
    Parse sequences from an uploaded file based on the selected format.

    Returns:
        tuple:
            sequences (list of SeqRecord) or None
            error (str) or None
    """
    try:
        if format_name.lower() == "newick":
            return None, "Newick format is for phylogenetic trees, not sequences."

        file.seek(0)  # Ensure the file pointer is at the start
        file_content = file.read().decode("utf-8")
        seq_io = StringIO(file_content)
        sequences = list(SeqIO.parse(seq_io, format_name.lower()))
        if not sequences:
            return None, f"No sequences found in the uploaded {format_name} file."

        return sequences, None
    except Exception as e:
        return None, f"Error parsing {format_name} file: {e}"


def parse_sequences_from_structure(file, format_name):
    """
    Parse sequences from PDB or mmCIF structure files.

    Args:
        file: Uploaded file object.
        format_name (str): "PDB" or "mmCIF".

    Returns:
        tuple:
            sequences (list of SeqRecord) or None
            error (str) or None
    """
    try:
        file.seek(0)
        filename = file.name
        file_basename = os.path.splitext(os.path.basename(filename))[0]

        if format_name == "PDB":
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(file_basename, file)
        elif format_name == "mmCIF":
            parser = MMCIFParser(QUIET=True)
            # Read and decode the file content to a string
            content = file.read().decode('utf-8')
            # Use StringIO to create a text stream for MMCIFParser
            structure = parser.get_structure(file_basename, StringIO(content))
        else:
            return None, f"Unsupported format: {format_name}"

        ppb = PPBuilder()
        sequences = []

        for model in structure:
            for chain in model:
                seq_id = f"{file_basename}_{chain.id}"
                sequence = ""
                for pp in ppb.build_peptides(chain):
                    sequence += str(pp.get_sequence())
                if sequence:
                    seq_record = SeqRecord(Seq(sequence), id=seq_id, description="")
                    sequences.append(seq_record)

        if not sequences:
            return None, "No protein sequences found in the structure file."

        return sequences, None

    except Exception as e:
        return None, f"An error occurred while parsing the {format_name} file: {e}"


def save_msa_to_fasta(msa: List[PyFAMSASequence], output_path: str) -> bool:
    """
    Converts pyFAMSA MSA output to a FASTA file compatible with Biopython.
    
    Args:
        msa (List[pyfamsa.Sequence]): The MSA result from pyFAMSA.
        output_path (str): The file path where the FASTA file will be saved.
    
    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    try:
        # Create a list to hold Biopython SeqRecord objects
        seq_records = []
        
        for seq in msa:
            # Decode bytes to strings
            seq_id = seq.id.decode('utf-8')
            sequence = seq.sequence.decode('utf-8')
            
            # Create a SeqRecord object
            record = SeqRecord(Seq(sequence), id=seq_id, description="")
            seq_records.append(record)
        
        # Write the SeqRecord list to a FASTA file
        SeqIO.write(seq_records, output_path, "fasta")
        
        print(f"MSA successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"An error occurred while saving MSA to FASTA: {e}")
        return False


def get_msa_as_fasta(msa: List[PyFAMSASequence]) -> str:
    """
    Converts pyFAMSA MSA output to a FASTA-formatted string compatible with Biopython.
    
    Args:
        msa (List[pyfamsa.Sequence]): The MSA result from pyFAMSA.
    
    Returns:
        str: FASTA-formatted string of the aligned sequences.
    """
    try:
        # Create a list to hold Biopython SeqRecord objects
        seq_records = []
        
        for seq in msa:
            # Decode bytes to strings
            seq_id = seq.id.decode('utf-8')
            sequence = seq.sequence.decode('utf-8')
            
            # Create a SeqRecord object
            record = SeqRecord(Seq(sequence), id=seq_id, description="")
            seq_records.append(record)
        
        # Write the SeqRecord list to a FASTA-formatted string
        fasta_io = StringIO()
        SeqIO.write(seq_records, fasta_io, "fasta")
        fasta_content = fasta_io.getvalue()
        
        return fasta_content
    except Exception as e:
        st.error(f"An error occurred while converting MSA to FASTA: {e}")
        return ""


def pairwise_alignment_section(sequences, seq_type):
    st.header("🔀 Pairwise Alignment")

    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for pairwise alignment.")
        return

    # Select alignment mode
    align_mode = st.selectbox("Select Alignment Mode", ("Global", "Local", "Overlap"), help="Choose the type of pairwise alignment.")

    # Select sequences for alignment
    seq_names = [seq.id for seq in sequences]
    selected_seqs = st.multiselect("Select two sequences for alignment", seq_names, default=seq_names[:2])
    c1, c2 = st.columns(2)
    open_gap_score = c1.number_input("Open Gap Score", value=-0.5, step=0.1, help="Score for opening a gap in the alignment.")
    extend_gap_score = c2.number_input("Extend Gap Score", value=-0.1, step=0.1, help="Score for extending a gap in the alignment.")

    if len(selected_seqs) != 2:
        st.warning("Please select exactly two sequences for pairwise alignment.")
        return

    # Define Reference and Target
    reference_id = selected_seqs[0]
    target_id = selected_seqs[1]

    seq1 = next(seq for seq in sequences if seq.id == reference_id)
    seq2 = next(seq for seq in sequences if seq.id == target_id)

    if st.button("Run Pairwise Alignment"):
        with st.spinner("Aligning..."):
            alignment_text, mutations = perform_pairwise_alignment(seq1, seq2, seq_type, align_mode.lower(), open_gap_score, extend_gap_score)
        st.subheader("🧬 Alignment Result")
        st.code(alignment_text)

        st.subheader("🔍 Point Mutations Relative to Reference")

        if mutations:
            # Display Reference and Target Information
            st.write(f"**Reference Sequence:** `{reference_id}`")
            st.write(f"**Target Sequence:** `{target_id}`")

            # Concatenate mutations into a single string
            mutations_str = ', '.join(mutations)
            st.write(f"**Mutations:** {mutations_str}")

            # Prepare data for Plotly scatter plot
            mutation_positions = []
            mutation_labels = []
            for mut in mutations:
                # Extract the position number from the mutation string
                pos = ''.join(filter(str.isdigit, mut))
                mutation_positions.append(int(pos))
                mutation_labels.append(mut)

            # Create Plotly scatter plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=mutation_positions,
                y=[1]*len(mutation_positions),  # Y-axis can be arbitrary since it's pairwise
                mode='markers',
                marker=dict(size=10, color='red'),
                text=mutation_labels,
                hoverinfo='text'
            ))

            fig.update_layout(
                title="Point Mutations Plot",
                xaxis_title="Position in Reference Sequence",
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.write("No point mutations detected.")

        # Download Alignment
        st.download_button(
            label="📥 Download Alignment",
            data=alignment_text,
            file_name=f"pairwise_alignment_{reference_id}_vs_{target_id}.txt",
            mime="text/plain"
        )


def msa_section(sequences, seq_type):
    st.header("📏 Multiple Sequence Alignment (MSA)")

    if len(sequences) < 2:
        st.warning("Please upload at least two sequences for MSA.")
        return

    # Select reference sequence
    ref_seq_id = st.selectbox("Select Reference Sequence for Mutation Reporting", [seq.id for seq in sequences])

    # Select output format
    msa_output_format = st.selectbox(
        "Select MSA Output Format",
        ("fasta", "clustal", "phylip", "stockholm"),
        help="Choose the desired format for the MSA output."
    )
    # User selects plotting method
    plot_method = st.selectbox(
        "Select Plotting Method",
        ("Plotly (Interactive)", "Matplotlib (Static)"),
        help="Choose 'Plotly' for interactive heatmaps (slower with large MSAs) or 'Matplotlib' for faster, static heatmaps."
    )

    if st.button("Run MSA"):
        with st.spinner("Performing MSA..."):
            msa_result, mutations = perform_msa(sequences, ref_seq_id, seq_type, msa_output_format)
        
        if msa_result:
            st.subheader("📄 MSA Result")
            seqs_total = parse_sequences_from_text(msa_result)[0]
            st.write(f'Total sequences: {len(seqs_total)}. MSA length: {len(seqs_total[0].seq)}')
            #st.text(msa_result)

            # Convert MSA text to image data
            try:
                msa_image, msa_letters = msa_to_image(msa_result, msa_output_format)
                plot_msa_image(msa_image, msa_letters, plot_method)
            except Exception as e:
                st.error(f"Failed to generate MSA heatmap: {e}")

            st.subheader("🔍 Point Mutations Relative to Reference")
            if mutations:
                for seq_id, muts in mutations.items():
                    #st.markdown(f"**{seq_id}:**")
                    #st.write(muts)
                    pass
            else:
                st.write("No point mutations detected relative to the reference sequence.")

            # Download MSA
            st.download_button(
                label="📥 Download MSA",
                data=msa_result,
                file_name=f"msa_alignment.{msa_output_format}",
                mime=f"text/{msa_output_format}"
            )


def format_alignment_display(id1, aligned1, match_line, id2, aligned2, score, interval=10):
    """
    Create a formatted string for alignment display, preserving spacing.
    """

    def generate_location_line(aligned_seq, interval=10):
        location = [' '] * len(aligned_seq)
        residue_count = 0
        for i, char in enumerate(aligned_seq):
            if char!= '-':
                residue_count += 1
                if residue_count % interval == 0:  # Mark at multiples of interval
                    location[i] = str(residue_count)  # Directly place number
        return "".join(location)


    max_len = max(len(id1), len(id2), len("Match"))
    id1_padded = id1.ljust(max_len)
    id2_padded = id2.ljust(max_len)
    match_padded = "Match".ljust(max_len)

    loc1 = generate_location_line(aligned1, interval)
    loc2 = generate_location_line(aligned2, interval)

    alignment_text = (
        "Pairwise Alignment:\n"
        f"{loc1}\n"  # Location line above sequence
        f"{id1_padded}: {aligned1}\n"
        f"{match_padded}: {match_line}\n"
        f"{id2_padded}: {aligned2}\n"
        f"{loc2}\n"  # Location line below sequence
        f"{'Score'.ljust(max_len)}: {score:.2f}\n"
    )
    return alignment_text


def format_conversion_section(sequences, input_format):
    st.header("🔄 Sequence Format Conversion")

    # Output format selection for conversion
    conversion_output_format = st.selectbox(
        "Select Output Format",
        ("fasta", "clustal", "phylip", "embl", "genbank"),
        help="Choose the desired format for the converted sequences."
    )

    if st.button("Convert Format"):
        if input_format.lower() == "newick" and conversion_output_format.lower() != "newick":
            st.warning("Newick format is a tree format and cannot be converted to sequence formats directly.")
        else:
            with st.spinner("Converting format..."):
                converted_data, error = convert_format(sequences, conversion_output_format)
            if converted_data:
                st.success("Format conversion successful.")
                st.text(converted_data)

                # Download Converted File
                st.download_button(
                    label="📥 Download Converted File",
                    data=converted_data,
                    file_name=f"converted_sequences.{conversion_output_format}",
                    mime=f"text/{conversion_output_format}"
                )
            else:
                st.error(f"Format conversion failed: {error}")


def phylogenetic_tree_section(tree):
    st.header("🌳 Phylogenetic Tree")

    if not tree:
        st.warning("No phylogenetic tree to display.")
        return

    # Display the tree
    st.subheader("📄 Phylogenetic Tree")
    tree_io = StringIO()
    Phylo.draw_ascii(tree, out=tree_io)
    tree_content = tree_io.getvalue()
    st.text(tree_content)

    # Download Tree in Newick Format
    tree_newick_io = StringIO()
    Phylo.write(tree, tree_newick_io, "newick")
    tree_newick = tree_newick_io.getvalue()

    st.download_button(
        label="📥 Download Phylogenetic Tree (Newick)",
        data=tree_newick,
        file_name="phylogenetic_tree.newick",
        mime="text/plain"
    )


def perform_pairwise_alignment(seq1, seq2, seq_type, mode="global", open_gap_score = -0.5, extend_gap_score = -0.1):
    """
    Perform pairwise alignment using Biopython's Align.PairwiseAligner.

    Parameters:
        seq1 (SeqRecord): First sequence.
        seq2 (SeqRecord): Second sequence.
        seq_type (str): Type of sequences ('DNA' or 'Protein').
        mode (str, optional): Alignment mode ('global', 'local', 'overlap'). Defaults to 'global'.

    Returns:
        tuple:
            alignment_text (str): Formatted alignment string.
            mutations (list): List of point mutations.
    """
    try:
        aligner = Align.PairwiseAligner()

        # Set alignment mode
        aligner.mode = mode  # Options: 'global', 'local', 'overlap'

        # Set substitution matrix and gap scores based on sequence type
        if seq_type == "DNA":
            aligner.substitution_matrix = substitution_matrices.load("NUC.4.4")
        else:  # Protein
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

        aligner.open_gap_score = open_gap_score
        aligner.extend_gap_score = extend_gap_score

        # Perform alignment
        alignments = aligner.align(seq1.seq, seq2.seq)

        # Take the best alignment
        alignment = next(alignments)

        # Extract aligned sequences with gaps
        aligned_seq1, match_line, aligned_seq2 = reconstruct_aligned_sequences(
            alignment[0], 
            alignment[1], 
            alignment.aligned, 
            aligner.substitution_matrix
        )

        # Generate mutation report
        mutations = report_mutations(aligned_seq1, aligned_seq2)

        # Format alignment for display
        alignment_text = format_alignment_display(seq1.id, aligned_seq1, match_line, seq2.id, aligned_seq2, alignment.score)

        return alignment_text, mutations

    except Exception as e:
        st.error(f"An error occurred during pairwise alignment: {e}")
        print(traceback.format_exc())
        return "Error during alignment.", []


def reconstruct_aligned_sequences(seq1, seq2, alignment_blocks, substitution_matrix, threshold=1):
    """
    Reconstruct the aligned sequences with gaps and generate a match line based on the alignment blocks.

    Parameters:
        seq1 (str): Original sequence 1.
        seq2 (str): Original sequence 2.
        alignment_blocks (list): List of alignment blocks, each block is [[s1_start, s1_end], [s2_start, s2_end]].
        substitution_matrix (SubstitutionMatrix): Biopython substitution matrix used for alignment.
        threshold (int, optional): Score threshold to consider residues as similar. Defaults to 1.

    Returns:
        tuple:
            aligned_seq1_str (str): Aligned sequence 1 with gaps.
            match_line_str (str): Match line indicating matches/mismatches.
            aligned_seq2_str (str): Aligned sequence 2 with gaps.
    """
    
    matched_positions = ''
    
    for n, position in enumerate(str(seq1)):
        if position == str(seq2)[n]:
            matched_positions += "|"
        elif are_similar(position, str(seq2)[n], substitution_matrix, threshold):
            matched_positions += ":"
        else:
            matched_positions += "#"
    
    return seq1, matched_positions, seq2


def are_similar(a, b, substitution_matrix, threshold=1):
    """
    Determine if two residues are similar based on the substitution matrix.

    Parameters:
        a (str): Residue from the first sequence.
        b (str): Residue from the second sequence.
        substitution_matrix (SubstitutionMatrix): Biopython substitution matrix.
        threshold (int, optional): Score threshold for similarity. Defaults to 1.

    Returns:
        bool: True if residues are similar, False otherwise.
    """
    try:
        # Exclude gaps from similarity comparison
        if a == '-' or b == '-':
            return False

        # Fetch the substitution score
        score = substitution_matrix[a.upper()][b.upper()]

        # Determine similarity based on the threshold
        return score >= threshold
    except KeyError:
        # If residues are not found in the substitution matrix, consider them not similar
        return False


def format_alignment_display(id1, aligned1, match_line, id2, aligned2, score, interval=10):
    """
    Create a formatted string for alignment display, including match lines and location markers.
    Ensures that id1, Match, and id2 lines are aligned properly with location markers.

    Parameters:
        id1 (str): Identifier of the first sequence.
        aligned1 (str): Aligned sequence 1 with gaps.
        match_line (str): Match line indicating matches/mismatches.
        id2 (str): Identifier of the second sequence.
        aligned2 (str): Aligned sequence 2 with gaps.
        score (float): Alignment score.
        interval (int): Interval at which to place location numbers (default is 10).

    Returns:
        str: Formatted alignment string ready for display.
    """
    def generate_location_line(aligned_seq, interval=10):
        """
        Generate a location marker line for an aligned sequence.

        Parameters:
            aligned_seq (str): The aligned sequence with gaps.
            interval (int): Interval at which to place location numbers.

        Returns:
            str: A string with location numbers placed at the correct positions.
        """
        location = [' ']*len(aligned_seq)
        residue_count = 0
        next_mark = interval

        for i, char in enumerate(aligned_seq):
            if char != '-':
                residue_count += 1
                if residue_count == next_mark:
                    mark_str = str(next_mark)
                    mark_len = len(mark_str)
                    start_pos = i - mark_len + 1
                    if start_pos < 0:
                        start_pos = 0
                    # Place each digit of the number
                    for j, digit in enumerate(mark_str):
                        pos = start_pos + j
                        if pos < len(aligned_seq):
                            # Avoid overwriting existing digits
                            if location[pos] == ' ':
                                location[pos] = digit
                    next_mark += interval

        location_str = ''.join(location)
        return location_str

    # Define the labels
    labels = [id1, "Match", id2]

    # Determine the maximum label length for padding
    max_label_length = max(len(id1), len("Match"), len(id2))

    # Pad the labels to ensure uniform alignment
    id1_padded = id1.ljust(max_label_length)
    match_padded = "Match".ljust(max_label_length)
    id2_padded = id2.ljust(max_label_length)

    # Generate location lines for both sequences
    location1 = generate_location_line(aligned1, interval)
    location2 = generate_location_line(aligned2, interval)

    # Create padding spaces for numbering lines based on label length and ": "
    padding = ' ' * (max_label_length + 2)  # +2 for ": "

    # Prefix the numbering lines with padding to align with sequences
    location1_padded = padding + location1
    location2_padded = padding + location2

    # Construct the formatted alignment text with numbering lines in the desired order
    alignment_text = (
        'Pairwise Alignment:\n'
        f"{location1_padded}\n"
        f"{id1_padded}: {aligned1}\n"
        f"{match_padded}: {match_line}\n"
        f"{id2_padded}: {aligned2}\n"
        f"{location2_padded}\n"
        f"{'Score'.ljust(max_label_length)}: {score:.2f}\n"
    )

    return alignment_text


def report_mutations(aligned1, aligned2):
    """
    Report point mutations between two aligned sequences in the format S18I.

    Parameters:
        aligned1 (str): Aligned reference sequence with gaps.
        aligned2 (str): Aligned target sequence with gaps.

    Returns:
        list: List of formatted point mutations (e.g., ['S18I', 'A45T']).
    """
    mutations = []
    position = 0  # Position in the reference sequence without gaps

    for i, (a, b) in enumerate(zip(aligned1, aligned2), start=1):
        if a != '-':
            position += 1
        # Only consider mutations where both positions are amino acids (no gaps)
        if a != b and a != '-' and b != '-':
            mutations.append(f"{a}{position}{b}")
    return mutations


def perform_msa(sequences, reference_id, seq_type, output_format):
    """
    Perform Multiple Sequence Alignment using pyFAMSA and report mutations relative to the reference sequence.

    Parameters:
        sequences (list of SeqRecord): List of sequences to align.
        reference_id (str): ID of the reference sequence.
        seq_type (str): Type of sequences ('DNA' or 'Protein').
        output_format (str): Desired output format for the MSA.

    Returns:
        tuple:
            msa_text (str): Aligned sequences in the chosen format.
            mutations (dict): Dictionary of point mutations relative to the reference sequence.
    """
    try:
        # Convert Biopython SeqRecord to pyFAMSA Sequence
        pyfamsa_sequences = [
            PyFAMSASequence(seq.id.encode(), str(seq.seq).encode()) for seq in sequences
        ]

        # Initialize pyFAMSA Aligner
        aligner = PyFAMSAAligner(guide_tree="upgma")

        # Perform MSA
        msa = aligner.align(pyfamsa_sequences)

        # Convert aligned pyFAMSA Sequences back to Biopython SeqRecord
        aligned_seq_records = [
            SeqRecord(
                Seq(seq.sequence.decode()),
                id=seq.id.decode(),
                description=""
            )
            for seq in msa
        ]

        # Convert aligned sequences to desired output format using Biopython
        msa_io = StringIO()
        SeqIO.write(aligned_seq_records, msa_io, output_format)
        msa_text = msa_io.getvalue()
        
        # Convert MSA text to image data
        msa_image, msa_letters = msa_to_image(msa_text, output_format)

        # Get reference sequence
        ref_seq = next((seq for seq in aligned_seq_records if seq.id == reference_id), None)
        if not ref_seq:
            st.error("Reference sequence not found in MSA results.")
            return msa_text, {}

        # Compare each sequence to reference
        mutations = {}
        for seq in aligned_seq_records:
            if seq.id == reference_id:
                continue
            seq_mutations = []
            for i, (ref_base, seq_base) in enumerate(zip(ref_seq.seq, seq.seq), start=1):
                if ref_base != seq_base and ref_base != '-' and seq_base != '-':
                    seq_mutations.append(f"**Position {i}:** {ref_base} → {seq_base}")
            if seq_mutations:
                mutations[seq.id] = seq_mutations

        return msa_text, mutations

    except Exception as e:
        st.error(f"An error occurred during MSA: {e}")
        print(traceback.format_exc())
        return "", {}


def convert_format(sequences, output_format):
    """
    Convert sequences from the current format to the desired output format.

    Parameters:
        sequences (list of SeqRecord): List of sequences to convert.
        output_format (str): Desired output format.

    Returns:
        tuple:
            converted_data (str): Converted sequences as a string.
            error (str): Error message if any.
    """
    try:
        output = StringIO()
        SeqIO.write(sequences, output, output_format)
        converted_data = output.getvalue()
        return converted_data, None
    except Exception as e:
        return None, f"Error during format conversion: {e}"


def build_phylogenetic_tree(sequences, seq_type):
    """
    Build a simple phylogenetic tree from sequences using a distance matrix.

    Parameters:
        sequences (list of SeqRecord): List of sequences.
        seq_type (str): Type of sequences ('DNA' or 'Protein').

    Returns:
        tree (Phylo.Tree): Phylogenetic tree object.
    """
    try:
        # Compute pairwise distances
        ids = [seq.id for seq in sequences]
        matrix = []
        for i, seq1 in enumerate(sequences):
            row = []
            for j, seq2 in enumerate(sequences[:i]):
                distance = compute_distance(seq1.seq, seq2.seq, seq_type)
                row.append(distance)
            matrix.append(row)

        distance_matrix = DistanceMatrix(names=ids, matrix=matrix)

        # Construct the tree using UPGMA
        constructor = DistanceTreeConstructor()
        tree = constructor.upgma(distance_matrix)

        return tree
    except Exception as e:
        st.error(f"Tree construction failed: {e}")
        return None


def compute_distance(seq1, seq2, seq_type):
    """
    Compute a simple distance between two sequences.

    Parameters:
        seq1 (Seq): First sequence.
        seq2 (Seq): Second sequence.
        seq_type (str): Type of sequences ('DNA' or 'Protein').

    Returns:
        float: Calculated distance.
    """
    # Align sequences first if lengths differ
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    if seq_type == "DNA":
        aligner.substitution_matrix = substitution_matrices.load("NUC.4.4")
    else:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    alignments = aligner.align(seq1, seq2)
    
    if not alignments:
        return 1.0  # Maximum distance if no alignment found
    
    alignment = next(alignments)

    # Reconstruct aligned sequences and generate match line
    aligned_seq1, match_line, aligned_seq2 = reconstruct_aligned_sequences(
        str(alignment.aligned[0]),
        str(alignment.aligned[1]),
        alignment.aligned,
        aligner.substitution_matrix
    )

    # Calculate differences excluding gaps
    differences = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != b and a != '-' and b != '-')
    distance = differences / max(len(aligned_seq1), len(aligned_seq2))
    return distance


if __name__ == "__main__":
    main()
