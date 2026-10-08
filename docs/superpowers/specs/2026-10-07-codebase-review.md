# seq_align: Codebase Review and Improvement Spec

**Date:** 2026-10-07
**Status:** Approved by owner ("3D viewer use molstar. the rest sounds great")
**Scope:** All first-party code (`app.py`, `modules/*.py`, `tests/*`). The vendored `anarci/` package is used as-is.

Findings marked **[verified]** were reproduced with scripts during the review. Line numbers refer to commit `a2aebe3`.

---

## 1. Science fixes (results are wrong today)

| ID | Finding | Location |
|----|---------|----------|
| S1 | **Overlap mode always fails.** It sets `open_end_gap_score`, which does not exist in Biopython ≥1.80; the correct names are `end_open_gap_score`/`end_extend_gap_score`. Breaks Pairwise and Best Match; one unit test fails. **[verified]** | `pairwise.py:331-333` |
| S2 | **Local alignments report wrong mutation positions.** Counting starts at the local match, not reference residue 1 (true `I36W` reported as `I16W`). The location ruler has the same offset. **[verified]** | `pairwise.py:473-491`, `432-450` |
| S3 | **Circular mode renumbers the reference.** After rotating the reference, positions are reported in rotated coordinates. | `pairwise.py:261-266` |
| S4 | **PDB renumbering silently writes duplicate residue IDs** (`[4,5,6,4,5,6]`); keeps stale insertion codes; includes ligands/HETATM as `X` in the chain sequence; multi-model mappings overwrite each other. **[verified]** | `pdb_renumber.py:164-252` |
| S5 | **Gap defaults −0.5/−0.1 are far too weak** for BLOSUM62/NUC.4.4, producing gap-riddled alignments **[verified]**. Use −10/−0.5 (EMBOSS needle convention) everywhere. | `pairwise.py:100-112`, `best_match.py:54-65`, `phylogeny.py:150` |
| S6 | **`U`/`O`/`J` pass cleaning but crash BLOSUM62**; lowercase crashes "Accurate" clustering and tree building. **[verified]** | `pairwise.py:226`, `clustering.py:411`, `phylogeny.py:152` |
| S7 | **Identity metrics biased.** The MSA-based score counts double-gap columns as mismatches (identical sequences score 0.67) **[verified]**; the local-mode score divides by the local alignment length (an 8-residue shared motif scores 1.0) **[verified]**. | `clustering.py:386-387`, `417-420` |
| S8 | Clustering: single linkage only (chaining); PCA applied to similarity rows; the auto threshold takes the highest non-empty bin, giving all singletons. | `clustering.py:147-170`, `433-468` |
| S9 | Trees: UPGMA only (assumes a molecular clock), uncorrected p-distance, no rooting choice. | `phylogeny.py:89-171` |
| S10 | Consensus: gaps count as residues; `X` used for DNA; the gapped consensus is exported. | `msa.py:268-324` |
| S11 | MSA: case-sensitive mutation calls; `seq_type` unused; no input cleaning. | `msa.py:214-266` |
| S12 | ORF finder: every internal Met is reported as a separate ORF **[verified]**; coordinates are amino-acid positions in the frame and are wrong for minus frames; RNA `U` is stripped; only codon table 1. | `dna.py:71-137` |
| S13 | Antibody: the default species filter (human, mouse) drops camelid VHH; regions are only extracted for IMGT. | `antibody.py:19`, `60-62` |
| S14 | Best Match "Normalized Score" is score/query length, which is not comparable across queries. | `best_match.py:238-239` |

## 2. Code and flow fixes

| ID | Finding |
|----|---------|
| C1 | FASTA → GenBank/EMBL fails (`missing molecule_type`); unaligned → Clustal/PHYLIP fails. **[verified]** (`conversion.py:84`) |
| C2 | PHYLIP MSA output with IDs sharing a 10-character prefix fails the whole MSA. **[verified]** (`msa.py:242`) |
| C3 | NMR/multi-model PDB files fail with "Duplicate sequence ID". **[verified]** Same-basename uploads collide. (`parsers.py:136`) |
| C4 | PDB Renumbering is offered for FASTA input and parses the FASTA file as a PDB; only the first of several PDB files is renumbered. (`app.py:212-213`) |
| C5 | Stale state: uploads cached by filename only; results keyed on IDs, not content; clustering/tree/renumbering state never cleared on a new dataset; a built tree is reused for a new dataset; `tree_newick` is shared between the MSA tree and the main tree. (`app.py:99,144`, `phylogeny.py:76`) |
| C6 | Errors returned as alignment text; the UI then claims "No point mutations detected". (`pairwise.py:302-305`) |
| C7 | Button-gated results (ORF, ANARCI) vanish on the next interaction. |
| C8 | MSA count re-parses non-FASTA output with the FASTA parser; the MSA reruns when only the output format changes. |
| C9 | Housekeeping: one failing test; `precommit_check.py` exits 0 on failure; committed `*.log`; unpinned requirements; `use_container_width` deprecated; the README lists formats the UI lacks. |

## 3. Memory and performance

| ID | Finding |
|----|---------|
| M1 | `phylogeny.py:39` never closes its matplotlib figure (leak per rerun). `plt.rcParams.update` mutates global state shared across sessions. |
| M2 | The Phylogenetic Tree mode reruns N² alignments on every widget change; clustering recomputes the MSA image every rerun; Kaleido PNG export runs every rerun. |
| M3 | The Plotly heatmap ships an N×L hover-string array; the MSA image is int64 plus an object array of letters. |
| M4 | The renumbering state keeps a full `Structure` plus a deepcopy indefinitely. |
| M5 | `substitution_matrices.load` runs on every aligner creation; `compute_distance` builds a new aligner per pair; the MSA identity is a pure-Python O(N²L) loop; temp files are used where file handles suffice. |

## 4. GUI and usability

G1 "…Ability" wording (accidental tool→ability rename) · G2 sequence type defaults to Protein; auto-detect it · G3 dataset summary panel and example data · G4 analyses for structure input are needlessly restricted · G5 pairwise output is one unwrapped line without identity/similarity/gap statistics · G6 results disappear when parameters change; show a "parameters changed — rerun" notice instead · G7 MSA heatmap uses a continuous colormap over residue codes; gaps are red; the matplotlib legend colors don't match the image **[verified]** · G8 clustering heatmap is unlabeled and unordered · G9 mutation plot puts every point at y=1 · G10 renumbering review has no per-chain target override.

## 5. New features (approved)

F1 Fetch by ID (UniProt, NCBI, RCSB PDB, AlphaFold DB) and SEQRES vs ATOM extraction.
F2 **3D viewer using Mol\*** (owner decision), embedded via MolViewSpec (`molviewspec` Python package → `molstar_streamlit`), coloring mutations, CDRs, conservation and pLDDT; conservation exported into the B-factor column for ChimeraX.
F3 Antibody: germline assignment with % identity, liability flags, Kabat/Chothia/Martin CDR definitions, heavy/light pairing.
F4 Variant table: sequences × positions, with indels in HGVS-like notation.
F5 Conservation: entropy track, sequence logo, percent-identity matrix download.
F6 MSA tools: trim gappy columns, redundancy filter, MAFFT option, NJ with bootstrap, interactive tree.
F7 ProtParam panel (MW, pI, extinction coefficient, A280 0.1 %, GRAVY, instability, aromaticity).
F8 DNA tools: reverse complement, codon usage, restriction map.
F9 Session export ZIP of all results plus parameters.

## Decisions

- Mol\* is embedded through `molviewspec==1.8.1` (`molstar_streamlit(data={filename: bytes}, molstar_version="5.10.1")`). Local structures travel inside an MVSX archive, so no file hosting is needed. Pin the Mol\* JS version; never use `latest`.
- Pure computation lives in functions without `st.*` calls so it can be unit-tested; Streamlit code lives in `*_section` functions.
- Default gap scores are −10 (open) and −0.5 (extend) for both DNA and protein.
