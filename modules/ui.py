"""Shared UI pieces: page headers, sidebar groups, the Appearance popover and theme-aware plotly display.

Colours and fonts come from the native themes in .streamlit/config.toml (Lab notebook by day, Notebook
night by night; switched under the app menu > Settings). The only custom CSS in the app is the Compact
density rule below, which changes spacing only.
"""
import streamlit as st

PALETTES = ("Lab teal", "Viridis", "Cividis", "Greys")
DENSITIES = ("Comfortable", "Compact")
CHART_MODES = ("Match theme", "Publication white")
LAB_TEAL = {"light": [[0.0, "#f3efe6"], [1.0, "#0f766e"]], "dark": [[0.0, "#26241f"], [1.0, "#2dd4bf"]]}
COMPACT_CSS = ('<style>[data-testid="stMainBlockContainer"]{padding-top:2.5rem;padding-bottom:2rem}'
               '[data-testid="stVerticalBlock"]{gap:0.55rem}</style>')
MOLSTAR_BACKGROUND = {"light": "#ffffff", "dark": "#1a1916"}


def theme_type():
    """'light' or 'dark' for the viewer's active theme (light when unknown, e.g. in tests)."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def page_header(title, subtitle=None):
    """Page title in the heading font plus a muted one-line description."""
    st.header(title, anchor=False)
    if subtitle:
        st.caption(subtitle)


def sidebar_group(label):
    """Small uppercase label that starts a group of sidebar controls."""
    st.sidebar.caption(f"**{label.upper()}**")


def appearance_popover():
    """Session-scoped appearance options, shown at the bottom of the sidebar."""
    with st.sidebar.popover("Appearance", icon=":material/palette:", width="stretch"):
        st.segmented_control("Figure palette", PALETTES, default=PALETTES[0], required=True,
                             key="appearance_palette")
        st.segmented_control("Density", DENSITIES, default=DENSITIES[0], required=True, key="appearance_density")
        st.segmented_control("Interactive charts", CHART_MODES, default=CHART_MODES[0], required=True,
                             key="appearance_charts")
        st.caption("Day or night theme: app menu (⋮) → Settings.")


def apply_density():
    if st.session_state.get("appearance_density") == "Compact":
        st.html(COMPACT_CSS)


def palette_colorscale():
    """Plotly colorscale for continuous data under the current Figure palette and theme."""
    palette = st.session_state.get("appearance_palette", PALETTES[0])
    return LAB_TEAL[theme_type()] if palette == "Lab teal" else palette


# plotly config for charts that should not zoom or pan (pass as config= to show_plotly)
STATIC_PLOT = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


def show_plotly(fig, **kwargs):
    """st.plotly_chart honouring the Interactive charts appearance setting."""
    if st.session_state.get("appearance_charts") == "Publication white":
        fig.update_layout(template="simple_white", font_family="Inter, Arial, sans-serif",
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, theme=None, **kwargs)
    else:
        st.plotly_chart(fig, theme="streamlit", **kwargs)
