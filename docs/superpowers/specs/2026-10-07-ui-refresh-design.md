# seq_align UI Refresh: Similarity Scale, Publication Trees, Modern Theme

**Date:** 2026-10-07
**Status:** Design approved in conversation (owner: "yes, go" on all three parts)
**Branch:** `feat/07-ui-refresh` (from `main` at `f203b23`)

## Goal

1. Make small differences visible in the clustering similarity matrix.
2. Make phylogenetic trees publication-ready, in four layouts and three styles, with no overlapping labels.
3. Give the whole app a modern, minimal look ("Lab notebook" by day, "Notebook night" by night) without fragile CSS.

**Success:**
- A 0.95 vs 0.98 similarity difference is clearly visible.
- A tree figure can go into a paper without touching it in Illustrator.
- The app looks like a polished tool, not a default Streamlit page.
- All existing tests stay green.

## Global constraints

- Streamlit stays pinned at 1.60.0. No new runtime dependencies; everything uses matplotlib, plotly, biopython and numpy, which are already pinned.
- Pure computation lives in functions without `st.*` calls (unit-tested). Streamlit code lives in `*_section` functions (existing convention).
- Never mutate process-global plotting state (`plt.rcParams`, `pio.templates.default`). Streamlit sessions share the process. Use `plt.rc_context` and per-figure templates.
- Static publication figures (tree Figure view, MSA image, sequence logo) always render on white, whatever the theme.
- Requirements markers for Python 3.13 and 3.14 stay as they are. The Streamlit Cloud check is re-run at the end.

---

## Part 1: Similarity matrix color scale

**Where:** Sequence Clustering → "Matrix Heatmap" tab (`modules/clustering.py`).

**Controls,** in one row above the heatmap:

- **Color range** (segmented control, key `sim_range`):
  - `Data range` (default) spans the **off-diagonal** min..max. The diagonal is always 1.0 and would otherwise flatten the scale; it displays in the top color.
  - `Fixed 0–1` is the current behavior.
  - `Custom` shows two number inputs, `sim_min` and `sim_max`, each limited to [0, 1]. If min ≥ max, show an error and fall back to Data range.
- **Colormap** (selectbox, key `sim_cmap`): Viridis, Cividis, Teal, Magma, Greys. A `Reverse` checkbox sits next to it. The default comes from the Appearance palette (Part 3).
- **Show values** checkbox: on by default when n ≤ 20 and hidden when n > 20. It prints each value in its cell with 3 decimals.

**Caption** under the plot gives the active scale, e.g. "Colors span 0.912–0.987 (off-diagonal range)". The color bar always shows real values.

**Pure functions:**

- `color_limits(matrix, mode, custom=None) -> tuple[float, float]`.
  - `mode` is one of `"data"`, `"fixed"`, `"custom"`.
  - Data mode ignores the diagonal and NaNs.
  - If off-diagonal values are constant or absent (a single sequence, or all identical), return `(v - 0.01, v + 0.01)` clipped to [0, 1]. Make sure zmin < zmax.
  - Custom mode raises `ValueError` if min ≥ max.
- `similarity_heatmap_figure(matrix, ids, order, zmin=0.0, zmax=1.0, colorscale="Viridis", show_values=False)`. The existing signature keeps working; the new arguments are optional.

**Tests:**
- Data mode on a matrix with a 1.0 diagonal returns the off-diagonal range.
- A constant matrix gives a valid non-empty range.
- NaNs are ignored.
- Custom mode validates min < max.
- The figure carries `zmin`/`zmax`/colorscale.
- `text` is present only when `show_values` is set.

---

## Part 2: Publication trees

> **Updated during planning (2026-10-08)** after a tested prototype. Where this section and the plan
> (`docs/superpowers/plans/2026-10-08-07-ui-refresh.md`) differ, the plan's code is authoritative.
> - `TreeGeometry` stores `edges` (each drawn as an L, an arc plus radial line, or a straight line) instead of separate segments and arcs.
> - Label placement happens in `tree_render.place` (in inches, with measured text sizes).
> - Circular labels always sit on an outer ring, with dotted leaders where tips fall short.
> - Unrooted labels are horizontal and placed greedily outward from their tips, trying ±60° in steps. Branches and leaders act as obstacles. Labels never overlap; this replaced the repulsion pass, which could oscillate.
> - Circular and unrooted clade names go in a legend; rectangular clade names sit beside the clade bar.
> - The Figure view shows a 150-dpi preview of the exact figure. The 600-dpi PNG is generated on click (callable download data), because it takes 2–6 s.

### Architecture

Shared geometry with two renderers:

- **`modules/tree_layout.py`** (pure, no plotting): `compute_geometry(tree, layout, cladogram=False) -> TreeGeometry`.
  - `TreeGeometry` (dataclass) holds:
    - `segments` (straight lines as `(x0, y0, x1, y1)`)
    - `arcs` (`(cx, cy, r, theta0, theta1)` for circular)
    - `nodes` (`NodeInfo`: x, y, support or None, clade index or None, is_leaf)
    - `leaves` (`LeafLabel`: x, y, angle_deg, ha, name)
    - `scale` (branch-length units per x unit, or None for a cladogram)
    - `extent`
  - `layout` is one of `"rectangular"`, `"circular"`, `"unrooted"`.
  - `cladogram=True` ignores branch lengths and aligns tips. It is allowed for rectangular and circular; unrooted ignores it and the UI disables it.
  - Rectangular uses the same coordinate convention as the current `tree_layout` (x = depth, y = leaf index). Circular maps the leaf index to an angle over 360° minus a small gap so the first and last tips don't touch. Unrooted uses the equal-angle algorithm (each subtree gets an angle wedge proportional to its leaf count).
- **`prepare_tree(tree, rooting="midpoint", outgroup=(), ladderize=True) -> Tree`**:
  - Works on a deep copy.
  - `rooting` is one of `"midpoint"`, `"outgroup"`, `"as_built"`.
  - If the outgroup is invalid (unknown names, or all leaves), raise `TreeError`. The UI catches it and falls back to midpoint with a warning.
  - Ladderize puts larger clades at the bottom.
- **`assign_clades(tree, k) -> dict[clade, int]`**:
  - Cuts the rooted tree into k groups by taking the k deepest-splitting subtrees: repeatedly split the group whose root has the greatest height.
  - k = 1 means no colouring. k is limited to the number of leaves.
  - Leaves and the internal nodes inside a group get its index.
- **`modules/tree_render.py`**:
  - `TreeStyle` (dataclass) fields:
    - `style` ∈ {classic, dots, clades}
    - `support_min` = 50
    - `font_family` = "Arial"
    - `font_size` = 9
    - `italic` = True
    - `underscores_to_spaces` = True
    - `truncate` = 0 (0 = off)
    - `line_width` = 1.0
    - `width_cm` = 16.0
    - `height_cm` = None (auto)
    - `clade_names`: list[str]
    - `palette` = Okabe–Ito
  - `render_matplotlib(geometry, style) -> Figure`. Uses `plt.rc_context` only and a white background. Rectangular corners are rounded with a small radius.
  - `render_plotly(geometry, style, dark=False) -> go.Figure`. Same marks and colours, with hover on leaves (full name) and nodes (support, branch length).
  - `export_bytes(fig, fmt) -> bytes` for `"svg"`, `"pdf"` and `"png"`. Settings:
    - SVG: `svg.fonttype = "none"`, so text stays editable.
    - PDF: `pdf.fonttype = 42`.
    - PNG: 600 dpi.
    - All via `rc_context`, `bbox_inches="tight"`.

### Styles

| Style | Branches | Support | Labels | Extras |
|---|---|---|---|---|
| Classic ink | black, `line_width` | grey numbers at the node, upper left, white halo (`patheffects`), shown when ≥ `support_min` | italic, next to the tip | scale bar |
| Support dots | black | circles: filled ≥ 95, grey 70–94, open < 70; nodes below `support_min` get no dot | aligned at max x with dotted leaders (rectangular/circular, non-cladogram); next to the tip otherwise | scale bar plus a three-item dot legend |
| Clade colour | coloured per clade (stem branches into a clade take its colour; branches above the clades stay black) | dots as in Support dots | coloured per clade | soft background band per clade (alpha 0.08); clade bar with the name on the right (rectangular) or an outer arc (circular); legend (unrooted) |

If the tree has no support values, all styles skip the support marks and the legend. Cladograms have no scale bar.

### Labels: no overlap

**Rectangular and cladogram:**
- Auto height = `n_leaves × font_size_pt × 1.4` plus margins.
- If the user sets `height_cm` and it is too small, reduce the font until labels fit. Below 5 pt, keep 5 pt and show a warning ("labels will overlap; increase height").

**Circular:**
- Radius ≥ `n_leaves × label_height / (2π × (1 − gap))`.
- Labels are radial; on the left half (90° < angle < 270°) they are rotated 180° with `ha="right"` so nothing is upside down.

**Unrooted:**
- After the equal-angle layout, each label is anchored at its tip and oriented along the incoming branch direction (flipped like circular).
- Then run a repulsion pass of up to 200 iterations. Measure label boxes in data units from a matplotlib renderer, then push each overlapping pair apart along the line between their centres.
- Draw a thin grey leader from the tip to any label moved by more than half a label height.

**Support labels** get a white halo. If the median vertical node spacing is under 1.2 × the support font height, Classic ink switches to dots for that figure, with a caption.

**Long names:** `truncate` > 0 cuts names to N characters plus "…". Hover and the Newick file keep full names.

**Big trees:** above 500 leaves, the Figure view hides labels and shows a notice; Explore keeps hover.

### UI (`phylogenetic_tree_section(tree, key_prefix)`)

**Settings,** in a bordered "Figure settings" container with two columns:
- Layout (segmented): Rectangular / Circular / Unrooted. Plus a Cladogram toggle, disabled for unrooted.
- Style (segmented): Classic ink / Support dots / Clade colour.
- Rooting: Midpoint / Outgroup / As built. Outgroup shows a multiselect of leaves. Plus a Ladderize toggle.
- Clades k (slider 1..min(12, n), shown for Clade colour). A small `st.data_editor` holds clade names (default "Clade 1..k").
- Support ≥ (slider 0–100, default 50).
- Font (Arial / Inter / Times New Roman), size (5–16, default 9), italic, underscores → spaces, truncate (0–60).
- Width (cm, 8–40, default 16), height (Auto or manual cm), line width (0.5–3).

**Views** (segmented): **Figure** (default) and **Explore**.
- Figure: `st.pyplot` of the exact export figure.
- Explore: `render_plotly`, which follows the theme via `show_plotly` from Part 3.

**Downloads,** in a row: SVG, PDF, PNG (600 dpi), Newick.

**Widget keys** are prefixed with `key_prefix` (`"tree"`, `"msa_tree"`, `"newick"`) so the Phylogenetic Tree page, the MSA tree tab and an uploaded Newick file keep independent settings.

**Caching:** the rendered figure is cached with `st.cache_data(max_entries=16)`, keyed on (Newick text, prepared settings). The figure bytes are cached rather than Figure objects, and every created figure is closed.

**Replaced:** the old `tree_layout` and `tree_figure` in `phylogeny.py` and the `Phylo.draw` path. Their tests are rewritten for the new modules. `tree_to_newick` and the bootstrap code are unchanged; the Newick export uses the tree as rooted and ladderized in the UI.

### Tests

- **Geometry:**
  - Rectangular leaf y are 0..n−1 and parent y is the mean of its children.
  - Circular leaves lie on a circle of the expected radius in cladogram mode.
  - Unrooted branch lengths are preserved (segment length = branch length × scale).
- **No overlap:** for random trees (seeded) with 10, 50 and 150 leaves, every layout (rectangular, cladogram, circular, unrooted) at default settings produces leaf label bounding boxes (via `fig.canvas.get_renderer()` and `Text.get_window_extent`) with no pairwise intersection, allowing 0.5 px tolerance.
- **Font shrink:** a forced small height lowers the font and reports the warning flag below 5 pt.
- **Rooting:** an outgroup becomes the root's child; an invalid outgroup raises `TreeError`; ladderize ordering holds.
- **Clades:** `assign_clades` with k groups gives exactly k distinct indices covering all leaves; k = 1 gives none.
- **Exports:** SVG starts with `<?xml` or `<svg` and contains the leaf names as `<text>`; PDF starts with `%PDF`; PNG starts with `\x89PNG` and has 600 dpi metadata.
- **AppTest:** build a tree on the globin example and switch through layouts and styles with no exception; the download buttons are present.

---

## Part 3: GUI and theme

### Native themes (`.streamlit/config.toml`)

**`[theme]` (shared):**
- `font = "Inter:https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"`
- `headingFont = "Source Serif 4:https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&display=swap"`
- `codeFont = "IBM Plex Mono:https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap"`
- `baseRadius = "0.5rem"`
- `showWidgetBorder = true`
- `showSidebarBorder = true`
- `chartCategoricalColors = Okabe–Ito with teal first: #0f766e, #E69F00, #56B4E9, #009E73, #F0E442, #0072B2, #D55E00, #CC79A7`
- `chartSequentialColors = ["#f0fdfa", "#ccfbf1", "#99f6e4", "#5eead4", "#2dd4bf", "#14b8a6", "#0d9488", "#0f766e", "#115e59", "#134e4a"]`

**`[theme.light]`, Lab notebook:**
- `primaryColor` #0f766e
- `backgroundColor` #fbfaf7
- `secondaryBackgroundColor` #f3f0e9
- `textColor` #1c1b19
- `borderColor` #e7e3da
- `linkColor` #0f766e
- `codeBackgroundColor` #f3f0e9
- `[theme.light.sidebar]`: `backgroundColor` #f3f0e9

**`[theme.dark]`, Notebook night:**
- `primaryColor` #2dd4bf
- `backgroundColor` #1a1916
- `secondaryBackgroundColor` #211f1b
- `textColor` #ece8df
- `borderColor` #2e2c27
- `linkColor` #5eead4
- `codeBackgroundColor` #211f1b
- `[theme.dark.sidebar]`: `backgroundColor` #211f1b

Day/night switching happens under ⋮ → Settings and follows the OS by default. Fonts load from Google Fonts in the browser; offline, the browser falls back to system fonts.

All option names were checked against Streamlit 1.60.0's config template. A test loads the file through `streamlit.config` and asserts both themes are set.

### Layout

- **Header:** replace the emoji `st.title` with a compact wordmark row: "seq_align" in the heading font plus a one-line subtitle ("Align, compare and annotate sequences and structures"). Put a **Guide** popover at the right, holding the current "How to use" text.
- **`page_header(title, subtitle)`** in a new `modules/ui.py`: a serif title plus a muted one-line description. Every analysis section uses it instead of `st.header` and the emoji text.
- **Emoji removal:** every emoji in headers, subheaders, buttons and download labels across `app.py` and `modules/` is removed (about 30 sites). Download, run, export and fetch buttons get Material icons through the `icon=` parameter (`:material/download:`, `:material/play_arrow:`, `:material/ios_share:`, `:material/cloud_download:`). Status messages keep Streamlit's own icons.
- **Sidebar:** labelled groups, each label a small uppercase muted caption written by `ui.sidebar_group(label)`.
  - DATA: example and Load, input format, input widgets.
  - ANALYSIS: sequence type, then the analysis chooser as `st.radio(..., key="analysis_select", label_visibility="collapsed")`. It replaces the selectbox; the key and option list stay the same.
  - Footer: Export session, then the Appearance popover.
- **Dataset overview** (`modules/overview.py`): a row of `st.metric(..., border=True)` cards: Sequences, Type, Length (min–max, median), Structures (when any). The current details stay in an expander.

### Appearance popover (`modules/ui.py`, session-scoped)

- **Figure palette** (`appearance_palette`): Lab teal (default), Viridis, Cividis, Greys. This sets the default colormap for the similarity heatmap, the identity heatmap and other continuous plotly scales. "Lab teal" is a plotly scale from #f3efe6 to #0f766e by day, or from #26241f to #2dd4bf by night.
- **Density** (`appearance_density`): Comfortable (default) or Compact. Compact injects one `<style>` rule that tightens vertical gaps in `.block-container` (`gap`, padding). This is the only custom CSS in the app, and it is documented in `ui.py`.
- **Interactive charts** (`appearance_charts`): Match theme (default) or Publication white. `show_plotly(fig, **kw)` in `ui.py` replaces all 11 `st.plotly_chart` calls.
  - Match theme uses `theme="streamlit"` (the native theme adapts plotly).
  - Publication white uses `theme=None` plus `template="simple_white"`, with Inter or Arial fonts.
- **Mol\* canvas:** `build_view_state` passes `background_color` = white by day and #1a1916 by night, chosen from `st.context.theme.type` (default light when unknown).

### Tests

- The config file parses, and both themes define the primary, background and text colours.
- `page_header` renders title and subtitle (AppTest markdown).
- `show_plotly` honours `appearance_charts`.
- The palette default reaches `similarity_heatmap_figure`.
- **AppTest flows:** existing tests switch from `at.selectbox(key="analysis_select")` to `at.radio(key="analysis_select")` (13 sites).
- **Header text:** no emoji code points in `page_header` and sidebar labels. A test scans `app.py` and `modules/*.py` source for emoji ranges inside `st.header`, `st.subheader`, `st.title`, `page_header` and `button` label strings.

---

## Part 4: Delivery

- One implementation plan on `feat/07-ui-refresh`, executed with subagent-driven development: a review per task, a final whole-branch review on the most capable model, one fix wave.
- The Streamlit Cloud reproduction from 2026-10-07 is re-run on the result: a fresh Python 3.12 env from `requirements.txt`, real `streamlit run` plus the websocket probe, and the suite on 3.13 and 3.14.
- Visual verification needs a browser. If none is available, the owner does the final visual pass. The tree no-overlap guarantee is covered by the measured-bounding-box test, not by eye.

## Out of scope

- Runtime switching between more than two colour themes (see the CSS fragility trade-off).
- Multipage navigation (`st.navigation`).
- Changing analysis features.
