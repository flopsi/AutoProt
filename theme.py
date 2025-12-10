"""
theme.py - ThermoFisher Scientific Brand Theme Configuration
============================================================

Provides unified theme management for Streamlit app across all pages.
Supports both Light and Dark modes with ThermoFisher brand colors.

Colors:
  Primary: Red (#E71316) - Brand color for highlights, buttons, titles
  Dark Gray (#54585A) - Text and secondary elements
  White (#FFFFFF) - Backgrounds
  
  Chart Sequence: Navy → Sky → Green → Yellow → Orange → Dark Red
  (Used for consistent visualization across the app)

Usage:
  from theme import get_theme_colors, apply_theme_css, get_chart_colors
  
  colors = get_theme_colors()
  apply_theme_css()
  chart_colors = get_chart_colors()
"""

import streamlit as st
from typing import Dict, List, Tuple

# ============================================================================
# PRIMARY COLORS (ThermoFisher Brand)
# ============================================================================

PRIMARY_RED = "#E71316"
BRAND_RED = "#E71316"
DARK_GRAY = "#54585A"
LIGHT_GRAY = "#E2E3E4"
WHITE = "#FFFFFF"
BLACK = "#000000"

# DARK MODE BACKGROUNDS
DARK_BG = "#1F2121"
DARK_SURFACE = "#2A2D2F"
DARK_TEXT = "#E2E3E4"

# LIGHT MODE BACKGROUNDS
LIGHT_BG = "#FFFFFF"
LIGHT_SURFACE = "#F5F5F5"
LIGHT_TEXT = "#54585A"

# ACCENT COLORS (Chart Sequence)
ACCENT_NAVY = "#262262"
ACCENT_SKY = "#9BD3DD"
ACCENT_GREEN = "#B5BD00"
ACCENT_YELLOW = "#F1B434"
ACCENT_ORANGE = "#EA7600"
ACCENT_DARKRED = "#A6192E"

# Chart color palette in recommended order
CHART_COLORS = [
    ACCENT_NAVY,
    ACCENT_SKY,
    ACCENT_GREEN,
    ACCENT_YELLOW,
    ACCENT_ORANGE,
    ACCENT_DARKRED,
]

# ============================================================================
# THEME CONFIGURATION FUNCTIONS
# ============================================================================

def get_theme_colors() -> Dict[str, str]:
    """
    Get color palette based on current theme preference.
    
    Returns:
      dict with color tokens for current theme
    """
    # Detect system theme preference
    is_dark = _is_dark_theme()
    
    if is_dark:
        return {
            "bg_primary": DARK_BG,
            "bg_secondary": DARK_SURFACE,
            "text_primary": DARK_TEXT,
            "text_secondary": "#B0B0B0",
            "brand_red": PRIMARY_RED,
            "gray_dark": DARK_GRAY,
            "gray_light": LIGHT_GRAY,
            "border": "#404040",
            "chart_bg": DARK_BG,
        }
    else:
        return {
            "bg_primary": LIGHT_BG,
            "bg_secondary": LIGHT_SURFACE,
            "text_primary": LIGHT_TEXT,
            "text_secondary": "#808080",
            "brand_red": PRIMARY_RED,
            "gray_dark": DARK_GRAY,
            "gray_light": LIGHT_GRAY,
            "border": "#E0E0E0",
            "chart_bg": LIGHT_BG,
        }

def get_chart_colors(n_colors: int = 6) -> List[str]:
    """
    Get chart color palette (ThermoFisher sequence).
    
    Args:
      n_colors: number of colors needed (max 6)
      
    Returns:
      list of hex colors in recommended order
    """
    return CHART_COLORS[:min(n_colors, len(CHART_COLORS))]

def _is_dark_theme() -> bool:
    """
    Detect if dark theme is active.
    Checks Streamlit's config or user preference.
    
    Returns:
      True if dark theme, False if light theme
    """
    try:
        # Check if user has set theme preference
        theme = st.session_state.get("theme_mode", "light")
        return theme == "dark"
    except:
        return False

def get_status_color(status: str, dark_mode: bool = False) -> str:
    """
    Get semantic color for status badges.
    
    Args:
      status: one of 'success', 'error', 'warning', 'info'
      dark_mode: whether to use dark theme colors
      
    Returns:
      hex color code
    """
    status_colors = {
        "success": "#B5BD00",  # Green
        "error": "#A6192E",    # Dark Red
        "warning": "#EA7600",  # Orange
        "info": "#9BD3DD",     # Sky Blue
    }
    return status_colors.get(status, PRIMARY_RED)

# ============================================================================
# CSS STYLING FUNCTIONS
# ============================================================================

def apply_theme_css() -> None:
    """
    Apply custom CSS styling for ThermoFisher theme.
    Call this once per page after st.set_page_config().
    """
    colors = get_theme_colors()
    
    css = f"""
    <style>
    
    /* Root theme variables */
    :root {{
        --color-primary: {PRIMARY_RED};
        --color-dark-gray: {DARK_GRAY};
        --color-light-gray: {LIGHT_GRAY};
        --color-white: {WHITE};
        --color-black: {BLACK};
    }}
    
    /* Main app styling */
    [data-testid="stAppViewContainer"] {{
        background-color: {colors['bg_primary']};
        color: {colors['text_primary']};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {colors['bg_secondary']};
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {colors['brand_red']};
        font-weight: 600;
    }}
    
    h1 {{
        font-size: 28pt;
        color: {colors['brand_red']};
    }}
    
    h2 {{
        font-size: 24pt;
        color: {colors['text_primary']};
    }}
    
    h3 {{
        font-size: 20pt;
        color: {colors['text_primary']};
    }}
    
    /* Body text */
    body {{
        color: {colors['text_primary']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    
    p {{
        color: {colors['text_secondary']};
    }}
    
    /* Buttons */
    .streamlit-button {{
        background-color: {colors['brand_red']};
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 8px 16px;
    }}
    
    .streamlit-button:hover {{
        background-color: {ACCENT_DARKRED};
        opacity: 0.9;
    }}
    
    /* Input fields */
    input, textarea, select {{
        border: 1px solid {colors['border']};
        border-radius: 4px;
        padding: 8px 12px;
        color: {colors['text_primary']};
        background-color: {colors['bg_secondary']};
    }}
    
    input:focus, textarea:focus, select:focus {{
        border-color: {colors['brand_red']};
        outline: none;
        box-shadow: 0 0 0 3px rgba(231, 19, 22, 0.1);
    }}
    
    /* Cards / Containers */
    [data-testid="stContainer"] {{
        background-color: {colors['bg_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }}
    
    /* Tabs */
    [data-testid="stTabs"] {{
        border-bottom: 2px solid {colors['border']};
    }}
    
    [data-testid="stTabs"] button {{
        color: {colors['text_secondary']};
        font-weight: 500;
    }}
    
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {colors['brand_red']};
        border-bottom: 3px solid {colors['brand_red']};
    }}
    
    /* Metrics */
    [data-testid="metric"] {{
        background-color: {colors['bg_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 16px;
    }}
    
    /* Data frames */
    [data-testid="stDataFrame"] {{
        border: 1px solid {colors['border']};
        border-radius: 8px;
    }}
    
    /* Messages */
    .stSuccess {{
        background-color: rgba(181, 189, 0, 0.1);
        color: {colors['text_primary']};
    }}
    
    .stError {{
        background-color: rgba(166, 25, 46, 0.1);
        color: {colors['text_primary']};
    }}
    
    .stWarning {{
        background-color: rgba(234, 118, 0, 0.1);
        color: {colors['text_primary']};
    }}
    
    .stInfo {{
        background-color: rgba(155, 211, 221, 0.1);
        color: {colors['text_primary']};
    }}
    
    /* Links */
    a {{
        color: {ACCENT_SKY};
        text-decoration: none;
    }}
    
    a:hover {{
        text-decoration: underline;
    }}
    
    /* Dividers */
    hr {{
        border: none;
        border-top: 2px solid {colors['border']};
        margin: 16px 0;
    }}
    
    /* Spinners & Loading */
    [data-testid="stSpinner"] {{
        color: {colors['brand_red']};
    }}
    
    /* Checkboxes and Radios */
    input[type="checkbox"]:checked {{
        background-color: {colors['brand_red']};
        border-color: {colors['brand_red']};
    }}
    
    input[type="radio"]:checked {{
        border-color: {colors['brand_red']};
    }}
    
    /* Slider */
    .stSlider > div > div > div {{
        background-color: {colors['brand_red']};
    }}
    
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def apply_metric_style(label: str, value: str, delta: str = None, color: str = None) -> None:
    """
    Render a styled metric with ThermoFisher theme.
    
    Args:
      label: metric label
      value: metric value
      delta: optional delta/change value
      color: optional accent color (default: primary red)
    """
    accent = color or PRIMARY_RED
    
    metric_css = f"""
    <div style="
        background-color: rgba(231, 19, 22, 0.05);
        border-left: 4px solid {accent};
        padding: 16px;
        border-radius: 4px;
        margin: 8px 0;
    ">
        <div style="color: #54585A; font-size: 12px; font-weight: 600; text-transform: uppercase;">
            {label}
        </div>
        <div style="color: {accent}; font-size: 28px; font-weight: 700; margin: 8px 0;">
            {value}
        </div>
        {f'<div style="color: #54585A; font-size: 12px;">{delta}</div>' if delta else ''}
    </div>
    """
    
    st.markdown(metric_css, unsafe_allow_html=True)

# ============================================================================
# CHART STYLING
# ============================================================================

def get_plotly_theme() -> Dict:
    """
    Get Plotly theme configuration for ThermoFisher branding.
    
    Returns:
      dict with plotly template configuration
    """
    colors = get_theme_colors()
    
    return {
        "layout": {
            "paper_bgcolor": colors["chart_bg"],
            "plot_bgcolor": colors["chart_bg"],
            "font": {
                "family": "Arial, sans-serif",
                "size": 12,
                "color": colors["text_primary"],
            },
            "title": {
                "font": {
                    "size": 18,
                    "color": PRIMARY_RED,
                    "family": "Arial, sans-serif",
                }
            },
            "xaxis": {
                "gridcolor": colors["border"],
                "linecolor": colors["border"],
                "showgrid": True,
                "zeroline": False,
            },
            "yaxis": {
                "gridcolor": colors["border"],
                "linecolor": colors["border"],
                "showgrid": True,
                "zeroline": False,
            },
        },
        "data": {
            "scatter": [
                {
                    "marker": {"color": CHART_COLORS[0]},
                }
            ]
        },
    }

def get_matplotlib_colors() -> Dict[str, str]:
    """
    Get matplotlib color configuration.
    
    Returns:
      dict with matplotlib rcParams
    """
    is_dark = _is_dark_theme()
    
    if is_dark:
        return {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_SURFACE,
            "axes.edgecolor": LIGHT_GRAY,
            "text.color": DARK_TEXT,
            "xtick.color": DARK_TEXT,
            "ytick.color": DARK_TEXT,
            "grid.color": "#404040",
            "axes.labelcolor": DARK_TEXT,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlecolor": PRIMARY_RED,
        }
    else:
        return {
            "figure.facecolor": LIGHT_BG,
            "axes.facecolor": LIGHT_SURFACE,
            "axes.edgecolor": DARK_GRAY,
            "text.color": LIGHT_TEXT,
            "xtick.color": LIGHT_TEXT,
            "ytick.color": LIGHT_TEXT,
            "grid.color": "#E0E0E0",
            "axes.labelcolor": LIGHT_TEXT,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlecolor": PRIMARY_RED,
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_theme_mode(mode: str) -> None:
    """
    Set theme mode (light or dark).
    
    Args:
      mode: 'light' or 'dark'
    """
    if mode in ["light", "dark"]:
        st.session_state.theme_mode = mode
    else:
        raise ValueError("Theme mode must be 'light' or 'dark'")

def theme_toggle() -> None:
    """
    Display a theme toggle button in the sidebar.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("☀️ Light Mode", use_container_width=True):
            set_theme_mode("light")
            st.rerun()
    
    with col2:
        if st.button("🌙 Dark Mode", use_container_width=True):
            set_theme_mode("dark")
            st.rerun()

