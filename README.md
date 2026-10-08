# Advanced Sequence Alignment Tool

A Streamlit app for aligning, comparing and annotating DNA and protein sequences and structures.

## Inputs

- Paste FASTA text, or upload **FASTA**, **Clustal**, **GenBank**, **PDB** or **mmCIF** files (several PDB/mmCIF files at once).
- **Newick** trees for visualization.
- **Fetch by ID**: UniProt, NCBI, RCSB PDB and AlphaFold DB.
- Built-in examples: human/mouse globins (protein), the pUC19 plasmid (DNA), a germline-like antibody VH/VL pair.

The sequence type (DNA/protein) is detected automatically and can be overridden.

## Analyses

| Analysis | What it does |
|---|---|
| Pairwise | Global, Local or Overlap (free end gaps) alignment; identity, similarity, gaps; substitutions in reference coordinates; circular DNA support |
| MSA | FAMSA alignment, or MAFFT when installed (recommended for DNA); heatmap; column trimming and redundancy removal; substitutions vs a reference; variant table with indels; conservation and sequence logo; percent identity matrix; consensus and most representative sequence; NJ/UPGMA tree with optional bootstrap supports |
| Best Match Finder | Rank sequences against a reference by relative score, identity or coverage |
| Sequence Clustering | MSA- or pairwise-based identity; average-linkage, single-linkage or K-Means (PCoA); cluster FASTA export |
| Phylogenetic Tree | Neighbor-Joining or UPGMA on Jukes–Cantor (DNA) / Kimura (protein) corrected distances |
| Antibody Prediction | ANARCI numbering (IMGT, Kabat, Chothia, Martin, AHo, Wolfguy); FR/CDR regions for IMGT, Kabat, Chothia, Martin; germline V/J assignment; sequence liabilities; H/L pairing; CDRs shown on a structure |
| Translate DNA | Six-frame ORF finder with nucleotide coordinates and selectable codon table |
| Structure Viewer | Mol* viewer for loaded or fetched structures; color by pLDDT or B-factor; "Show on structure" panels for mutations and antibody CDRs; conservation exported as B-factors |
| Sequence Properties | Molecular weight, pI, extinction coefficients, A280 and GRAVY for protein sequences |
| DNA Tools | Reverse complement, codon usage and restriction map for DNA sequences |
| Extracted Sequences | FASTA of the chains extracted from uploaded PDB/mmCIF structures |
| Convert Formats | FASTA, Clustal, PHYLIP (relaxed), Stockholm, NEXUS, EMBL, GenBank |
| PDB Residue Renumbering | Renumber chains to match target sequences, with insertion codes and per-chain review |
| Session export | One ZIP with input sequences, the main computed results and the parameters used |
| Look and feel | "Lab notebook" day theme and "Notebook night" dark theme (switch in ⋮ → Settings, or follow your OS). The **Appearance** popover in the sidebar sets the figure palette, density, and whether interactive charts match the theme or use publication white. |
| Phylogenetic trees | Rectangular, circular and unrooted layouts (plus cladogram), Classic ink / Support dots / Clade colour styles, midpoint or outgroup rooting, ladderize, named clades. Labels never overlap. Export SVG (editable text), PDF or 600-dpi PNG; an Explore view adds hover and zoom. |
| Similarity matrix | Colour range from the data (off-diagonal min–max), fixed 0–1 or custom; colormap, reverse and per-cell values. |

Translate DNA is offered for DNA input and Antibody Prediction for protein input; the menu adapts to the detected sequence type.

## Install

Requires Python ≥ 3.10. Antibody numbering also requires HMMER (`hmmscan`). MAFFT is optional (enables the MAFFT aligner).

```bash
git clone https://github.com/dzyla/seq_align.git
cd seq_align
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# HMMER: conda install -c bioconda hmmer   or   sudo apt-get install hmmer
# optional MAFFT: conda install -c bioconda mafft   or   sudo apt-get install mafft
streamlit run app.py
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python precommit_check.py
```

Default gap scores are −10 (open) and −0.5 (extend), the EMBOSS needle convention.

## License

MIT License

Copyright (c) 2025 Dawid Zyla
