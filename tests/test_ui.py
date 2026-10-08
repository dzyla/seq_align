import textwrap
import tomllib
from pathlib import Path

from streamlit import config
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def run_page(tmp_path, body):
    """Run a tiny Streamlit script that can import the app's modules."""
    script = tmp_path / "page.py"
    script.write_text(f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n" + textwrap.dedent(body))
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _theme():
    return tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]


def test_theme_config_defines_day_and_night():
    theme = _theme()
    light, dark = theme["light"], theme["dark"]
    assert (light["primaryColor"], light["backgroundColor"], light["textColor"]) == ("#0f766e", "#fbfaf7", "#1c1b19")
    assert (dark["primaryColor"], dark["backgroundColor"], dark["textColor"]) == ("#2dd4bf", "#1a1916", "#ece8df")
    assert light["sidebar"]["backgroundColor"] == "#f3f0e9"
    assert dark["sidebar"]["backgroundColor"] == "#211f1b"


def test_theme_options_are_known_to_streamlit():
    def keys(prefix, table):
        for k, v in table.items():
            if isinstance(v, dict):
                yield from keys(f"{prefix}.{k}", v)
            else:
                yield f"{prefix}.{k}"
    known = config._config_options_template
    assert [k for k in keys("theme", _theme()) if k not in known] == []


def test_page_header_renders_title_and_subtitle(tmp_path):
    at = run_page(tmp_path, """
        from modules.ui import page_header
        page_header("Pairwise alignment", "Two sequences, one reference")
    """)
    assert at.header[0].value == "Pairwise alignment"
    assert at.caption[0].value == "Two sequences, one reference"


def test_appearance_defaults_and_palette(tmp_path):
    at = run_page(tmp_path, """
        import streamlit as st
        from modules.ui import appearance_popover, palette_colorscale
        appearance_popover()
        st.session_state.out = palette_colorscale()
    """)
    assert at.session_state["appearance_palette"] == "Lab teal"
    assert at.session_state["appearance_density"] == "Comfortable"
    assert at.session_state["appearance_charts"] == "Match theme"
    assert at.session_state["out"] == [[0.0, "#f3efe6"], [1.0, "#0f766e"]]


def test_named_palette_and_publication_white_charts(tmp_path):
    at = run_page(tmp_path, """
        import plotly.graph_objects as go
        import streamlit as st
        from modules.ui import apply_density, palette_colorscale, show_plotly
        st.session_state.appearance_palette = "Cividis"
        st.session_state.appearance_charts = "Publication white"
        st.session_state.appearance_density = "Compact"
        apply_density()
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
        show_plotly(fig)
        st.session_state.out = (palette_colorscale(), fig.layout.paper_bgcolor)
    """)
    assert at.session_state["out"] == ("Cividis", "white")
    assert len(at.get("plotly_chart")) == 1
