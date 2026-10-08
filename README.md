# Advanced Sequence Alignment Tool

A Streamlit app for aligning, comparing and annotating DNA and protein sequences and structures.

## Inputs

- Paste FASTA text, or upload **FASTA**, **Clustal**, **GenBank**, **PDB** or **mmCIF** files (several PDB/mmCIF files at once).
- **Newick** trees for visualization.
- Built-in examples: human/mouse globins (protein), the pUC19 plasmid (DNA), a germline-like antibody VH/VL pair.

The sequence type (DNA/protein) is detected automatically and can be overridden.

## Analyses

| Analysis | What it does |
|---|---|
| Pairwise | Global, Local or Overlap (free end gaps) alignment; identity, similarity, gaps; substitutions in reference coordinates; circular DNA support |
| MSA | FAMSA alignment; heatmap; substitutions vs a reference; consensus and most representative sequence; NJ/UPGMA tree |
| Best Match Finder | Rank sequences against a reference by relative score, identity or coverage |
| Sequence Clustering | MSA- or pairwise-based identity; average-linkage, single-linkage or K-Means (PCoA); cluster FASTA export |
| Phylogenetic Tree | Neighbor-Joining or UPGMA on Jukes–Cantor (DNA) / Kimura (protein) corrected distances |
| Antibody Prediction | ANARCI numbering (IMGT, Kabat, Chothia, Martin, AHo, Wolfguy); FR/CDR regions for IMGT, Kabat, Chothia, Martin |
| Translate DNA | Six-frame ORF finder with nucleotide coordinates and selectable codon table |
| Convert Formats | FASTA, Clustal, PHYLIP (relaxed), Stockholm, NEXUS, EMBL, GenBank |
| PDB Residue Renumbering | Renumber chains to match target sequences, with insertion codes and per-chain review |

## Install

Requires Python ≥ 3.10. Antibody numbering also requires HMMER (`hmmscan`).

```bash
git clone https://github.com/dzyla/seq_align.git
cd seq_align
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# HMMER: conda install -c bioconda hmmer   or   sudo apt-get install hmmer
streamlit run app.py
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python precommit_check.py
```

Default gap scores are −10 (open) and −0.5 (extend), the EMBOSS needle convention.
