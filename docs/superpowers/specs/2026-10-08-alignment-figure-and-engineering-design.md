# seq_align Sub-project A: Alignment Figure, Engineering Aids, Batch UniProt

**Date:** 2026-10-08
**Status:** Design approved in conversation (owner, parts 1–4, with the change "keep the Sequence Properties page; don't remove much")
**Branch:** `feat/08-figure-engineering` (from `main` at `7770f25`)
**Context:**
- seq_align is the heavy server-side sequence-analysis app.
- The owner's other apps cover structure comparison (pdb_align), interfaces (fastPISA), and bench calculators/cloning (Bio-Bench, https://dzyla.github.io/toolbox/). Nothing here may duplicate them.
- Sub-projects B (profile-HMM family search + taxonomy-coloured/annotated trees) and C (coevolution or ancestral reconstruction) get their own specs later.

## Goals

1. A publication alignment figure in the ESPript style, with secondary structure from a structure, exported as SVG/PDF/PNG.
2. Back-to-consensus mutation suggestions from the MSA.
3. A motif/site map across the family.
4. Batch UniProt download from many accessions, a search query, or gene names + organism.
5. Remove the DNA Tools page, which duplicates Bio-Bench, and point to Bio-Bench. **Sequence Properties stays.**

**Success:**
- A figure that can go straight into a paper. It must look like ESPript output, with DSSP secondary structure correctly placed over the reference.
- A ranked, exportable list of consensus mutations.
- A per-site conservation answer ("this sequon is in 5/6 sequences").
- Hundreds of UniProt IDs load in seconds, with the missing ones listed.

## Global constraints

- Streamlit is pinned at 1.60.0. The only new runtime dependency is the system package `dssp` (provides `mkdssp`) in `packages.txt`; the local environment already has `mkdssp` 3.1.2 in `/home/dzyla/miniconda3/bin`. No new Python packages.
- Pure computation lives in functions without `st.*` and is unit-tested. Streamlit code lives in `*_section` functions or tabs.
- New figure code uses `matplotlib.figure.Figure` + Agg, never pyplot. No global rcParams mutation; use `rc_context` only. Figures are drawn on white.
- Exports follow the tree convention: SVG with editable text (`svg.fonttype="none"`), PDF with TrueType fonts (`pdf.fonttype=42`), and a 600-dpi PNG generated on click.
- Network code is tested with mocked HTTP; no test may depend on UniProt or RCSB being reachable.
- Requirements markers for Python 3.13/3.14 stay unchanged.

---

## Part 1: Publication alignment figure ("Figure" tab on the MSA page)

### Inputs (tab controls)
- **Sequences:** a multiselect of the rows to show and their order. The default is all rows, reference first. It uses the untrimmed `rows` (coordinate-bearing), not the trimmed view.
- **Region:** a start..end residue range in reference numbering, defaulting to the full reference. It maps to alignment columns.
- **Secondary structure source**, a radio:
  - (a) **Loaded structure**: choose a structure + chain from `st.session_state.structures`. The best chain is preselected with the existing `rank_chains`.
  - (b) **PDB ID** + chain, fetched with `fetch_pdb_structure`.
  - (c) **AlphaFold** model for a UniProt accession (`fetch_alphafold_structure`).
  - (d) **None**.
- **Style:** a segmented control with Classic ESPript (default), Lab notebook, and Greyscale.
- **Layout:** block width 40–120 (default 60); font size 6–12 pt (default 8); similarity threshold 0.5–1.0 (default 0.7).
- **Tracks:** toggles for the consensus line (off), conservation bar (on), and markers (on, when Part 2 supplies any).
- **Names:** truncate at N characters (0 = off), and underscores to spaces (on).

### Secondary structure
- `secondary_structure(sf, chain) -> (chain_sequence, ss_string, source)`:
  - Runs DSSP through `Bio.PDB.DSSP` with `dssp="mkdssp"` and returns 8-state codes (H G I E B T S -).
  - If `mkdssp` is missing or fails, it parses the file's own records instead: PDB `HELIX`/`SHEET`, or mmCIF `_struct_conf` / `_struct_sheet_range`, giving H/E only.
  - If neither yields anything, it returns an empty SS and a reason string. The tab shows that reason and draws no track.
  - `source` is `"DSSP"`, `"file records"` or `"none"`, and is stated in the caption.
- **Mapping to the reference:** align the chain sequence to the ungapped reference with the existing `align_target_to_chain`. Transfer SS per aligned residue; reference residues absent from the chain get no SS.
- **Elements** (ESPript convention):
  - H runs of 4 or more → α helix, labelled α1, α2…
  - G runs of 3 or more → η (3-10), labelled η1…
  - E runs of 2 or more → β strand drawn as an arrow, labelled β1…
  - T runs of 2 or more → "TT".
  - I (π) draws like α. B and S are not drawn.
  - Numbering is by order along the reference.

### Column classes (stated in the caption)
- **Identical:** every shown sequence has the same residue at the column, with no gaps.
- **Similar:** at least `threshold` of the shown sequences carry residues from one group, and the residue at that cell belongs to that group. The groups are FYW, ILVM, KRH, DE, STA, NQ, G, P, C.
- Both the threshold and the groups are written in the caption. The caption must not claim equivalence with ESPript's Risler scoring.

### Rendering (`modules/espript.py`)
- **Pure functions:**
  - `column_classes(seqs, threshold)`
  - `ss_elements(ss)`
  - `map_ss_to_alignment(ref_aligned, chain_seq, chain_ss)`
  - `layout_blocks(n_cols, block)`
  - `render_alignment_figure(rows, ref_id, ss_by_column, style, block, font, threshold, tracks, markers, name_opts) -> Figure`
- **Inch-based layout** (as in `tree_render`): name column, then columns of width `font·0.62/72·1.25`, row height `font·1.45/72`.
- **Each block has:**
  - an SS track with labels above;
  - a ruler in reference numbering every 10;
  - rows with gap "." (ESPript convention);
  - optional consensus line and conservation bar;
  - marker row(s).
- **Styles:**
  - Classic: identical = white on `#d6312b`; similar = bold `#d6312b`; frames `#2a5db0` around runs of non-"none" columns.
  - Lab: identical white on `#0f766e`; similar bold `#0f766e` on `#d7efe9`; frames `#0f766e`.
  - Greyscale: identical white on `#1c1c1c`; similar on `#c4c4c4`; no frames.
- **Markers** are drawn under the last row of each block: ★ for consensus suggestions, ▲ for motif hits, each with a small legend.
- **Size guard:** if rows × columns shown exceeds 30,000, the tab asks the user to narrow the region or sequences before rendering, showing the count and the limit.
- **Caching:** the rendered outputs (preview PNG 150 dpi, SVG, PDF) are cached with `st.cache_data(max_entries=8)` keyed on all inputs. The 600-dpi PNG is a callable download.

### Shared export module
- Move `export_bytes`, `EXPORT_FORMATS`, `resolve_family`/`font_families`/`quiet_fonts` (and the measurer if needed) from `tree_render.py` into a new `modules/figures.py`.
- `tree_render` re-imports them, so existing tests and imports keep working.

### Tests
- Column classes:
  - an all-identical column;
  - a gap → not identical;
  - group similarity at and below the threshold;
  - only in-group residues are highlighted.
- `ss_elements`: labels/counters, minimum lengths, and that π counts as α.
- `map_ss_to_alignment`: gaps in the reference, a chain shorter than the reference, and a point mismatch.
- DSSP on `tests/data/1crn.pdb` returns the expected helix/strand counts (crambin: 2 helices, 1 sheet).
  - The test is skipped if `mkdssp` is absent.
  - A file-records fallback test uses a PDB with HELIX/SHEET lines.
- Block wrapping: the number of blocks and the last block's width.
- Exports: SVG starts with `<?xml` and contains residue letters as `<text>`; PDF `%PDF`; PNG at 600 dpi.
- The size guard triggers above 30,000.
- AppTest: the globin MSA → Figure tab, switching all three styles with SS source None, raises no exception, and the downloads are present.

---

## Part 2: Engineering aids (two MSA tabs)

### "Consensus" tab: back-to-consensus suggestions
- **Target:** a selectbox of any row (default: the reference).
- **Rule:** a suggestion is made at column j when:
  - the column occupancy (non-gap share) is at least 0.5;
  - the most frequent residue c* among non-gap residues has frequency f ≥ `min_freq` (default 0.6, slider 0.4–0.95);
  - the target residue at j is a residue (not a gap) and differs from c*.
- **Table:** target position (target numbering), the suggestion label (`A23S`), consensus frequency, conservation score (existing `column_conservation`), BLOSUM62(target, c*), and the number of non-gap sequences.
- **Ranking:** sorted by `f × conservation`, descending. Rows that are gaps in the target are skipped.
- **Warnings:** below 20 sequences, show "Consensus from fewer than 20 sequences is unreliable." No suggestions → an info message.
- **Outputs:** CSV download.
- **Linking to the figure and structure:**
  - The suggestions feed the figure's ★ markers, by alignment column.
  - They also feed the existing `structure_highlight_panel` (colour by rank), so they can be seen on a structure.
- **Pure function:** `consensus_suggestions(rows, target_id, min_freq=0.6, min_occupancy=0.5) -> DataFrame`.

### "Motifs & sites" tab
- **Built-in library** in PROSITE syntax, converted to regex by `prosite_to_regex`:

| Motif | Pattern |
|---|---|
| N-glycosylation sequon | `N-{P}-[ST]-{P}` |
| Furin | `R-x-[KR]-R` |
| TEV | `E-N-L-Y-F-Q-[GS]` |
| Thrombin | `L-V-P-R-G-S` |
| PreScission | `L-E-V-L-F-Q-G-P` |
| Factor Xa | `I-[ED]-G-R` |
| Enterokinase | `D-D-D-D-K` |
| Deamidation | `N-G` |
| Isomerisation | `D-G` |
| Cysteine | `C` |

- **Custom motifs:** the user adds a PROSITE pattern or a raw regex (a radio picks the syntax). An invalid pattern shows a clear error.
- **Search:** each motif is searched in every ungapped sequence with overlapping matches (lookahead). Hits are mapped back to alignment columns via a residue → column index.
- **Outputs:**
  - A **site summary** with one row per motif and alignment start column: reference position, number of sequences carrying it ("5/6"), and the sequence IDs.
  - A **per-sequence hit table** with sequence, motif, start–end in sequence numbering, and the matched text.
  - CSV downloads for both.
  - Hits feed the figure's ▲ markers (motifs chosen in a multiselect) and the structure panel.
- **Pure functions:**
  - `prosite_to_regex(pattern)`
  - `find_motifs(rows, motifs) -> hits DataFrame`
  - `site_summary(hits, rows, ref_id) -> DataFrame`

### Tests
- `consensus_suggestions`:
  - a clear consensus mismatch is suggested;
  - no suggestion below `min_freq` or `min_occupancy`;
  - a target gap is skipped;
  - the label uses target numbering;
  - the ordering is by score.
- `prosite_to_regex`:
  - `x`, `[..]`, `{..}`, `x(2)`, `x(2,4)`, `<`/`>` anchors;
  - invalid input raises `ValueError`.
- `find_motifs`:
  - overlapping hits are all reported;
  - the hit → column mapping is correct across gaps;
  - case-insensitive matching.
- `site_summary`: the counts.
- AppTest: both tabs on the globin MSA raise no exceptions.

---

## Part 3: Batch UniProt download (Fetch by ID → source "UniProt batch")

### Modes (radio)
1. **IDs:** a text area, or a .txt/.csv/.tsv upload.
   - Tokens matching the UniProt accession regex or the entry-name pattern (`[A-Z0-9]{1,10}_[A-Z0-9]{1,5}`) are extracted and de-duplicated in order.
   - Accessions use `https://rest.uniprot.org/uniprotkb/accessions?accessions=…` in chunks of up to 500.
   - Entry names use `https://rest.uniprot.org/uniprotkb/stream?query=(id:A OR id:B …)` in chunks of 100.
   - Missing IDs (requested minus returned) are listed.
2. **Search query:** a free text UniProt query.
   - The count comes first from `/uniprotkb/search?query=…&size=1` (header `X-Total-Results`).
   - The cap defaults to 2,000 and can be raised to 10,000 in a number input. Above the cap, the app shows the count and asks the user to narrow the query (nothing is downloaded).
   - Otherwise it streams `/uniprotkb/stream?query=…`.
3. **Genes + organism:**
   - Gene names come from a text area or upload.
   - The organism is a selectbox: Human 9606, Mouse 10090, Rat 10116, Yeast 559292, E. coli K-12 83333, Zebrafish 7955, Fruit fly 7227, C. elegans 6239, Arabidopsis 3702, or "Other taxon ID" (a number input).
   - Toggles: "Reviewed only" (on) and "Include isoforms" (off → `includeIsoform=false`).
   - The query is `(gene_exact:G1 OR gene_exact:G2 …) AND organism_id:T [AND reviewed:true]`, chunked at 100 genes.
   - The result table labels each entry with its matched gene. Genes with no entry are listed.

### Preview and load
- Every mode first fetches TSV with the fields `accession,id,protein_name,gene_primary,organism_name,length` and shows a preview `st.data_editor` with a "Load" checkbox column (all ticked).
- The **Load selected** button fetches FASTA for the ticked accessions with the batched accessions endpoint and feeds the normal sequence pipeline. It goes through `_load_once` with the digest of the accession list.
- **Download FASTA** and **Download TSV** are also offered.
- FASTA headers stay UniProt's (`>sp|P69905|HBA_HUMAN …`).
- **Errors:** 429 → "UniProt rate limit reached; wait a few seconds and try again". A network error → `FetchError` with the URL kind. A partial result loads what was found and lists the rest.
- **Pure, mockable functions in `modules/fetch.py`:**
  - `extract_uniprot_ids(text) -> (accessions, entry_names)`
  - `uniprot_batch_tsv(accessions, entry_names)`
  - `uniprot_query_count(query)`
  - `uniprot_query_tsv(query)`
  - `gene_query(genes, taxon, reviewed, isoforms) -> str`
  - `uniprot_fasta_for(accessions)`
  - All go through one `_get` with the existing hints.
- **Existing behaviour:** `fetch_uniprot_fasta` keeps its signature but uses the batched endpoint.

### Tests (HTTP mocked by monkeypatching `requests.get` / `_get`)
- ID extraction from mixed text and CSV.
- Chunking: 1,200 accessions → 3 requests.
- Missing IDs are detected.
- `gene_query` strings, including the reviewed and isoform options.
- The count-over-cap path downloads nothing.
- TSV parsing into the preview frame.
- AppTest of the Fetch-by-ID source with the network functions monkeypatched: the preview appears and Load selected loads the sequences.

---

## Part 4: Remove DNA Tools; link Bio-Bench

- **Remove:** the "DNA Tools" page from the menu (`navigation.available_analyses`), `dna_tools_section`, and the functions only it uses (`reverse_complement`, `codon_usage`, `restriction_sites`, `restriction_map_figure` and their constants) together with their tests.
- **Keep:**
  - **Translate DNA & ORF finder** (`dna_translation_section`, `find_orfs`).
  - **Sequence Properties.**
  - The clustering/identity matrix and trees.
- **Add:** a line in the Guide popover and in the sidebar footer: "Quick calculators (restriction maps, codon usage, primers, cloning) → [Bio-Bench](https://dzyla.github.io/toolbox/)".
- **README:** remove DNA Tools from the analyses table, add the new features, and mention Bio-Bench.
- **Tests:** the navigation tests are updated so DNA no longer lists "DNA Tools"; no other test may reference the removed functions.

---

## Delivery

- One plan on `feat/08-figure-engineering`, executed with subagent-driven development, in this task order:
  1. Shared figures module and the DNA Tools removal with the Bio-Bench link.
  2. Batch UniProt.
  3. ESPript core module.
  4. Figure tab and secondary-structure sources.
  5. Consensus tab.
  6. Motifs & sites tab, with markers wired into the figure.
- The plan carries prototype-verified code for the figure renderer: the prototype at commit time rendered the globin MSA with 4HHB chain B DSSP in all three styles, checked visually by the owner.
- A final whole-branch review on the most capable model, with one fix wave.
- The cloud check is re-run: a fresh Python 3.12 environment with `packages.txt` including `dssp`, a real `streamlit run`, and the suite on 3.13 and 3.14.

## Out of scope

- Profile-HMM search, taxonomy and annotated trees (sub-project B).
- Coevolution and ancestral reconstruction (sub-project C).
- ESPript's Risler-matrix similarity scoring and its secondary-structure comparison of multiple structures.
