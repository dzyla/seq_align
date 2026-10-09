# seq_align Sub-project C: Coevolution Contacts and Ancestral Sequence Reconstruction

**Date:** 2026-10-09
**Status:** Design approved in conversation (owner: "sounds good, go"; "just make sure it works on streamlit cloud")
**Branch:** `feat/09-coevolution-asr` (from `main` at `4e4f2d8`)
**Scope note:** Sub-project B (family search, taxonomy, annotated trees) was dropped by the owner. Nothing here may duplicate pdb_align, fastPISA or Bio-Bench.

## Goals

1. **Coevolution contacts.** Predict residue–residue contacts from a deep MSA with plmDCA (accurate) or mean-field DCA (fast). Show a contact map, a top-pairs table, the precision against a structure, and links drawn on the structure in Mol\*.
2. **Ancestral sequence reconstruction (ASR) done properly.** Use an ML tree and an empirical model in IQ-TREE 2/3, with marginal ancestral states and a posterior for every residue. The app adds parsimony reconstruction of indels, ancestor picking on the tree, uncertainty reporting, AltAll variants, comparison with extant sequences, and structure mapping.
3. **Both must work on Streamlit Cloud.** Memory and runtime are capped for at most 2.7 GB RAM and about 2 shared cores. The system binaries come from Debian packages in `packages.txt`, and the GitHub tests install the same packages.

**Success:**
- On the owner's typical ColabFold A3M, the top L/5 plmDCA pairs are mostly true contacts. The prototype on HBB (3,826 sequences, Neff/L 7.9) reached P@L/5 = 0.66 against 4HHB chain B, versus 0.41 for mean-field.
- An ancestor can be picked and downloaded with every uncertain site shown and its runner-up available.
- Everything works on the cloud.

## Prototype findings (2026-10-09; inform the limits below)

- **Test data:** a ColabFold API A3M for human HBB (3,973 sequences), compared with 4HHB chain B Cβ contacts (< 8 Å, |i−j| ≥ 6).
- **Mean-field DCA** (pseudocount 0.5, θ = 0.8 reweighting, APC): 3 s, 527 MB peak, P@L/5 0.41 (long-range 0.48).
- **plmDCA:**
  - Symmetric pseudo-likelihood with CCMpred-style regularisation (λ_h = 0.01, λ_J = 0.2·(L−1)).
  - A custom float32 L-BFGS over the upper-triangle couplings, with m = 3–5.
  - 30 iterations: 9 s, 0.86 GB peak, P@L/5 0.66 (long-range 0.66). 50 iterations give no gain.
  - The SciPy float64 version needs 2.3 GB, which is too much for the cloud. The lean implementation is required.
- **IQ-TREE:**
  - 2.0.3 (conda binary `iqtree`) and 3.1.4 (`iqtree3`) give the same `.state` format and node names on the globin alignment: `Node\tSite\tState\tp_A..p_V`, with `#` comment lines first. Each run takes under 1 s.
  - With `-B 1000`, the internal labels in `.treefile` become `NodeN/support`.
  - Debian and Ubuntu ship `iqtree` 2.0.7.

## Global constraints

- Streamlit is pinned at 1.60.0.
- **New system dependency:** `iqtree` in `packages.txt` and in the GitHub `apt-get install` line. No new Python packages; numpy and scipy are already pinned.
- **Binary detection:** the first of `iqtree2`, `iqtree3`, `iqtree` on PATH whose `--version` output contains "IQ-TREE".
- **Cloud resource caps** (constants in code, stated in the UI):
  - mfDCA: ≤ 400 query columns, ≤ 10,000 sequences.
  - plmDCA: ≤ 200 query columns, ≤ 5,000 sequences, 30 L-BFGS iterations by default (10–100 adjustable), history m = 3, int32 indices.
  - Larger alignments are subsampled with a fixed seed (the query always kept), or the user is asked to choose a region of columns.
  - IQ-TREE: 4–300 sequences, `-T 1`, a 15-minute subprocess timeout.
- Heavy computations run once per input and setting combination: `st.cache_data`, max_entries 4 for DCA and 4 for ASR.
- Pure computation lives in functions without `st.*` and is unit-tested. Streamlit code lives in `*_section` functions.
- New figures use plotly (interactive maps) or matplotlib `Figure` + Agg (exports), never pyplot. Exports go through `modules.figures`.
- Tests never need network access. Tests that need IQ-TREE or mkdssp skip when the binary is absent; the GitHub CI installs both, so they run there.
- No emoji characters in source (`tests/test_ui.py` scan).

---

## Part 1: Coevolution (new menu page "Coevolution")

### Alignment source (radio)
- **Current dataset:** the MSA from the existing cached alignment (`cached_run_msa(pairs, seq_type, method)`). The method can be FAMSA or MAFFT, as on the MSA page. Proteins only; a DNA dataset shows an info message.
- **Upload an alignment:** `.a3m`, `.sto`/`.stockholm`, or `.fasta`/`.fa`/`.aln`.
  - The file stays on this page, in `st.session_state.coevo_upload` as raw bytes plus a name. It never becomes the main dataset.
  - Upload size follows Streamlit's default 200 MB limit.

### Preparation (`modules/coevolution.py`, pure)
- `parse_alignment(text) -> [(id, aligned_seq)]`:
  - **A3M:** lowercase letters and "." are removed (insertions).
  - **Stockholm:** sequence lines are concatenated per ID, and `#`/`//` lines are ignored.
  - **Aligned FASTA:** read as is.
  - Sequences whose length differs from the first are dropped, with the count reported.
- `prepare_msa(rows, query=0, max_seq_gap=0.5, max_seqs, seed=0)`:
  - Encodes the 21 states `ACDEFGHIKLMNPQRSTVWY-` as int8 (unknown letters become gaps).
  - Keeps only columns where the query has a residue.
  - Drops sequences with more than half gaps (the query is always kept), then subsamples to `max_seqs` with a fixed seed.
  - Returns `PreparedMSA(msa, query_seq, n_input, n_dropped_length, n_dropped_gappy, n_subsampled)`.
- `sequence_weights(msa, theta=0.8, chunk=512)`:
  - Each sequence's weight is 1 over the number of sequences at least θ identical to it (identity counted over all L columns, gaps included).
  - It is computed in chunks to bound memory. `Neff = Σw`.

### Methods
- `mean_field_dca(msa, w, pseudo=0.5) -> scores`:
  - Weighted frequencies with pseudocount and a (20L)×(20L) covariance (gap state removed).
  - Couplings are −C⁻¹, inverted in float32.
  - Zero-sum gauge, then the Frobenius norm over the 20×20 blocks, then APC.
- `plm_dca(msa, w, iters=30, history=3) -> scores`:
  - Symmetric pseudo-likelihood with the CCMpred regularisation above.
  - Float32 parameters: fields L×21 plus the upper-triangle couplings only (flat int32 indices).
  - The custom L-BFGS uses Armijo backtracking.
  - Same gauge, Frobenius and APC output as mfDCA.
- **Limits:** above a method's column cap, the page asks for a column range in query numbering (a slider). Above its sequence cap, the alignment is subsampled with a fixed seed, and the counts are shown.

### Structure comparison (optional)
- The structure comes from the existing `_add_structure_controls` + chain selection (best chain by `rank_chains` against the query).
- `structure_distances(sf, chain, query_seq) -> L×L` holds Cβ distances (Cα for Gly, NaN when missing). Query positions map to chain residues by `align_target_to_chain`.
- `contact_precision(scores, distances, min_sep, top_k, cutoff=8.0)` returns the fraction of the top-k pairs (|i−j| ≥ min_sep, both positions modelled) closer than the cutoff.

### Output
- **Summary:**
  - Metrics: N input, N used, Neff, Neff/L, L, method, runtime.
  - Depth warning: Neff/L < 1 → st.error ("too few diverse sequences for reliable contacts"); 1 ≤ Neff/L < 5 → st.warning ("contacts may be noisy").
- **Contact map (plotly):**
  - Upper triangle: the top N predicted pairs (N = L by default, slider L/5..2L), marker size and colour by score. With |i−j| ≥ 6, short- and long-range pairs are coloured differently.
  - Lower triangle: structure contacts (< 8 Å) as grey dots when a structure is loaded.
  - Axes are in query numbering, with the query residue letters in the hover text.
  - The figure goes through `show_plotly`.
- **Precision table** (with a structure): P@L/5, P@L/2 and P@L, for |i−j| ≥ 6 and for |i−j| ≥ 24.
- **Top pairs table:** i, j (query numbering), residues, score, rank, and distance (Å) when a structure is loaded. CSV download.
- **Mol\* view:**
  - The top K pairs (slider 5–100, default L/5) are drawn as lines between Cβ atoms (Cα for Gly), using MolViewSpec primitives.
  - Green when the pair is within 8 Å in this structure, red otherwise, grey without distances.
  - Pairs whose residues are not modelled are skipped and counted in a caption.

### Tests
- **A3M parsing:** insertions removed, length mismatches dropped and counted.
- **Stockholm** multi-block parsing.
- **prepare_msa:** query-gap columns removed, gappy sequences dropped while the query is kept, deterministic subsampling.
- **sequence_weights** against a hand-computed example.
- **Synthetic coupling recovery.** Generate 2,000 sequences of L = 30 in which columns 5 and 20 co-vary perfectly (random residue pairs from a fixed 4-pair set) and the other columns are independent. Both methods must rank (5, 20) first.
- **contact_precision** on toy matrices.
- **structure_distances** on `tests/data/1crn.pdb` (an α helix i,i+4 distance ~ 6 Å, NaN handling).
- **Cap behaviour:** a region is required above the column cap, and subsampling counts are reported.
- **Memory guard:** plm_dca on L = 200, N = 1,000 random sequences finishes with peak RSS ≤ 1.5 GB, measured through `resource.getrusage` in a subprocess. Marked slow; it runs in CI.
- **AppTest:** the Coevolution page with an uploaded small A3M fixture, both methods, and no exception.

---

## Part 2: Ancestral reconstruction (new menu page "Ancestral Reconstruction")

### Input and run (`modules/ancestral.py`)
- **Input:** the current dataset's MSA (cached FAMSA/MAFFT). 4–300 sequences; protein (LG) or DNA (GTR).
- `find_iqtree() -> path | None` detects the binary as in the Global constraints. If it is missing, the page shows the install instructions: "add `iqtree` to packages.txt (cloud) / `conda install -c bioconda iqtree` (local)".
- `run_iqtree(aligned_pairs, seq_type, model="LG+G4"|"GTR+G4"|"MFP", bootstrap=0|1000, timeout=900) -> IQTreeResult`:
  - Sequence IDs are written as safe placeholders (`s0001` …) and mapped back afterwards, because IQ-TREE rewrites some characters.
  - Runs `-s aln.fasta -m MODEL -asr -T 1 --prefix run -quiet [-B 1000]` in a temporary directory with a timeout.
  - Parses `run.treefile` (Bio.Phylo; internal labels `NodeN` or `NodeN/support` are split into a name and a confidence) and `run.state`.
  - Reads the model and log-likelihood from `run.iqtree` (`Best-fit model` / `Model of substitution`, `Log-likelihood of the tree`).
  - **Errors:** a timeout → `ASRError("IQ-TREE did not finish within 15 min; use fewer sequences or LG+G4 instead of model selection.")`. A non-zero exit → `ASRError` with the last line of `run.log`.
- `IQTreeResult(tree, states: {node: (states str, posteriors np.ndarray[n_sites, n_alphabet])}, alphabet, model, loglik, files: {name: bytes})`.

### Indels
- `reconstruct_gaps(tree, aligned_pairs) -> {node: bool[n_sites]}` uses Fitch parsimony on presence/absence per column over the tree, as rooted for display.
  - Ties at the root resolve to "present" if the majority of leaves have a residue there.
  - Each internal node takes the Fitch top-down assignment.
- The ancestral sequence excludes absent columns.

### Rooting and node picking
- **Rooting:** midpoint (default) or outgroup, with the leaves chosen in a multiselect. It reuses `modules.tree_layout.prepare_tree` with `ladderize=True`. Node names survive re-rooting, and the new root has no states.
- **Picking an ancestor:**
  - (a) "Ancestor of these sequences": a multiselect of leaves resolves to their MRCA node with `tree.common_ancestor`. A single leaf is not an ancestor and is rejected.
  - (b) A node list with descendant counts and their first three leaf names, for example "Node3: 3 sequences (HBB1_MOUSE, HBB_HUMAN, HBG1_HUMAN)".
- **Tree display:** the existing `phylogenetic_tree_section` renderer, with internal nodes labelled by their IQ-TREE names. The selected node is highlighted with a marker, and bootstrap supports are shown when requested.

### Per-ancestor output
- **The ancestral sequence:** monospace, with each residue's background shaded by posterior (white 1.0 to orange < 0.8).
  - Mean posterior and the fraction of sites ≥ 0.9.
  - A plotly line of the posterior along the ancestor sequence, through `show_plotly`.
- **Uncertain sites:** posterior < 0.8 (slider 0.5–0.95). Each row gives the position in the ancestor, the alignment column, the ML residue and its posterior, and alternatives with posterior ≥ 0.2 (slider). CSV download.
- **Downloads:**
  - FASTA of the ML ancestor (header `>{node} ML ancestor; model; lnL`).
  - FASTA of the AltAll variant: every uncertain site set to its best alternative ≥ the alternative threshold.
  - The full IQ-TREE files (treefile, state, iqtree report) as a ZIP.
- **Comparison with an extant sequence:** pick a leaf to get the substitutions (ancestor → extant, in that leaf's numbering), the identity, and insertions and deletions.
- **On a structure:** the existing `structure_highlight_panel`, with the ancestor sequence as the reference. Uncertain sites are coloured by posterior bin (≥ 0.9 none, 0.8–0.9 light orange, < 0.8 orange), and positions differ from the chosen extant in a second colour.

### Tests
- **.state parsing** of a committed fixture: a real IQ-TREE 2.0.3 `.state` + `.treefile` + `.iqtree` from the globin alignment.
- **Node label splitting** (`Node4/50`).
- **reconstruct_gaps:** a gap shared by a clade gives an absent ancestor; a gap in one leaf gives a present ancestor; the root tie rule.
- MRCA picking, and rejection of a single leaf.
- **AltAll construction**, and the uncertain-sites thresholds.
- **ID round-trip** through the placeholders.
- **Error paths:** a missing binary gives an informative message; a timeout gives `ASRError` (monkeypatched subprocess).
- **End to end:** a real IQ-TREE run on the 6-globin alignment, skipped if the binary is absent; CI installs `iqtree`. It must give 4 internal nodes with posteriors in [0, 1] summing to ~1 per site.
- **AppTest:** the page with a monkeypatched `run_iqtree` returning the fixture, then picking an ancestor, with no exception.

---

## Part 3: Cloud wiring

- `packages.txt` gains `iqtree`. `.github/workflows/tests.yml` installs it with hmmer, mafft and dssp.
- The README rows explain both pages and their limits.
- The Guide popover gets one line about Coevolution and Ancestral Reconstruction.
- **Navigation:**
  - "Coevolution" appears in the analysis menu for any loaded dataset (its upload source does not depend on the dataset).
  - "Ancestral Reconstruction" appears for datasets with at least 4 sequences.
  - With no dataset loaded, the landing page also offers a button, "Coevolution from an uploaded alignment (A3M / Stockholm / FASTA)". It sets `st.session_state.coevo_standalone = True` and renders the Coevolution page with only the upload source, so a ColabFold A3M can be analysed without loading anything else.
- After merge, verify on the cloud:
  - the Coevolution page on an uploaded A3M (plmDCA);
  - Ancestral Reconstruction on the globins: the caption shows "IQ-TREE 2.0.7", the Debian version.

## Out of scope

- GPU or C-accelerated DCA, and coevolution between two proteins (interface coupling).
- Joint (as opposed to marginal) ASR, and codon models.
- Family search, taxonomy and annotated trees (sub-project B, dropped).
