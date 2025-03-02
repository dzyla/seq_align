# Advanced Sequence Alignment and Format Conversion Tool

**Reason:**

This application facilitates bioinformatics analyses by providing tools for both pairwise and multiple-sequence alignment (MSA) of DNA and protein sequences. It allows users to perform Global, Local, and overlap alignments, which are crucial for identifying regions of similarity that may indicate functional, structural, or evolutionary relationships. The tool also supports sequence format conversion between common bioinformatics file types and can generate phylogenetic trees. A key feature is the reporting of point mutations relative to a user-specified reference sequence in MSA, aiding in variant analysis.

**Usage:**

The application provides a web interface accessible through a web browser. It is divided into several functional sections:

1.  **Sequence Input:**
    *   Users can input biological sequences in several ways:
        *   **Text (FASTA):** Paste sequences directly into a text area in FASTA format.
        *   **File Upload:** Upload sequence files in FASTA, Clustal, Phylip, EMBL, GenBank, Newick, PDB, or mmCIF formats.  The application automatically detects the file type (except when "Text (FASTA)" is selected, where it expects FASTA format).
    *   For PDB and mmCIF files, the application extracts protein sequences from the structural data.
    *   For Newick files, the application displays and allows download of a phylogenetic tree.
2.  **Analysis Selection (Sidebar):**
    *   **Input Format:** Select the format of your input.  This determines how the uploaded file or pasted text is interpreted.
    *   **Sequence Type:** Specify whether the sequences are DNA or Protein. This selection affects the substitution matrices used in alignment.
    *   **Alignment Mode:** Choose the analysis to perform:
        *   **Pairwise Alignment:** Aligns two selected sequences. Options include Global, Local, and Overlap alignment modes. Users can also specify gap open and extend penalties.
        *   **MSA (Multiple Sequence Alignment):** Aligns multiple sequences using pyFAMSA. Users select a reference sequence for mutation reporting and the desired output format (fasta, clustal, phylip, stockholm). Also includes heatmap visualization with a choice of Plotly (interactive) or Matplotlib (static).
        *   **Convert Formats:** Converts between sequence file formats (fasta, clustal, phylip, embl, genbank).
        * **Phylogenetic Tree:** build from Newick file
3.  **Output:**
    *   **Pairwise Alignment:** Displays the alignment, alignment score, and a list/plot of point mutations relative to the first selected (reference) sequence.  The alignment can be downloaded as a text file.
    *   **MSA:** Displays the multiple sequence alignment, overall length and a heatmap visualizing the alignment. Point mutations relative to the selected reference sequence are reported in tabular format for each sequence. The MSA can be downloaded in the chosen format.
    *   **Format Conversion:** Displays the converted sequences and provides a download button.
    * **Phylogenetic Tree:** Displays phylogenetic tree and provide download button.

**Installation:**

This application is built using Python and requires several libraries. To install and run the application, you can use either `uv` (recommended for speed) or `conda`.

**Option 1: Using `uv` (Recommended)**

1.  **Prerequisites:** Ensure you have Python 3.7 or later and `uv` installed.  If you don't have `uv`, install it with:
    ```bash
    pip install uv
    ```

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/dzyla/seq_align.git](https://github.com/dzyla/seq_align.git)
    cd seq_align
    ```

3.  **Create and Activate a Virtual Environment (Optional but Recommended):**
     Although `uv` can manage global environments, creating a virtual environment is still good practice.
    ```bash
     uv venv
     source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

**Option 2: Using `conda`**

1.  **Prerequisites:** Ensure you have Anaconda or Miniconda installed.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/dzyla/seq_align.git
    cd seq_align
    ```

3.  **Create and Activate a Conda Environment:**
    ```bash
    conda create -n seq_align_env python=3.9  # Or your preferred Python version >= 3.7
    conda activate seq_align_env
    ```

4.  **Install Dependencies:**
    ```bash
    conda install --file requirements.txt

4.  **Run the Application:**

    ```bash
    streamlit run app.py
    ```
     (Assuming your main application file is named `app.py`.)

    This command will start the Streamlit server, and the application will be accessible in your web browser (usually at `http://localhost:8501`).

**License:**

MIT License

Copyright (c) 2025 Dawid Zyla
