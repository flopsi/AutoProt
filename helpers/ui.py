"""
helpers/ui.py

UI components and caching utilities for Streamlit interface
Reusable widgets and session state management
"""

import streamlit as st
import pandas as pd
from typing import Any, Optional, List
import hashlib

# ============================================================================
# SESSION STATE UTILITIES
# Helpers for managing Streamlit session state
# ============================================================================

def init_session_state(key: str, default_value: Any):
    """
    Initialize session state variable if it doesn't exist.
    
    Args:
        key: Session state key
        default_value: Initial value to set
    """
    if key not in st.session_state:
        st.session_state[key] = default_value

def get_session_state(key: str, default: Any = None) -> Any:
    """
    Safely get value from session state with fallback.
    
    Args:
        key: Session state key
        default: Default value if key not found
    
    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)

def set_session_state(key: str, value: Any):
    """
    Set value in session state.
    
    Args:
        key: Session state key
        value: Value to store
    """
    st.session_state[key] = value

def clear_session_state(*keys: str):
    """
    Clear specified keys from session state.
    
    Args:
        *keys: Variable number of keys to delete
    """
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

# ============================================================================
# DATA CACHING
# Cache expensive computations across page reruns
# ============================================================================

def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Compute stable hash for DataFrame to use as cache key.
    
    Args:
        df: DataFrame to hash
    
    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

def cache_in_session(key: str, compute_fn, *args, **kwargs):
    """
    Cache result of expensive computation in session state.
    Only recomputes if not in cache.
    
    Args:
        key: Cache key
        compute_fn: Function to call if not cached
        *args: Positional arguments for compute_fn
        **kwargs: Keyword arguments for compute_fn
    
    Returns:
        Cached or newly computed result
    """
    if key not in st.session_state:
        st.session_state[key] = compute_fn(*args, **kwargs)
    return st.session_state[key]

# ============================================================================
# UI COMPONENTS
# Reusable Streamlit widgets
# ============================================================================

def metric_card(label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """
    Display metric in a styled card.
    
    Args:
        label: Metric name
        value: Primary value to display
        delta: Optional change indicator
        help_text: Optional tooltip text
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label=label, value=value, delta=delta, help=help_text)

def download_button_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """
    Create download button for DataFrame as CSV.
    
    Args:
        df: DataFrame to download
        filename: Output filename
        label: Button label
    """
    csv = df.to_csv(index=True).encode('utf-8')
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

def download_button_excel(df: pd.DataFrame, filename: str, label: str = "Download Excel"):
    """
    Create download button for DataFrame as Excel.
    
    Args:
        df: DataFrame to download
        filename: Output filename (should end with .xlsx)
        label: Button label
    """
    from io import BytesIO
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True)
    
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )

def create_expander_with_help(title: str, help_text: str, expanded: bool = False):
    """
    Create expander with help icon.
    
    Args:
        title: Expander title
        help_text: Tooltip text
        expanded: Whether to start expanded
    
    Returns:
        Streamlit expander context manager
    """
    return st.expander(f"{title} ℹ️", expanded=expanded, help=help_text)

def multiselect_with_all(
    label: str,
    options: List[str],
    default: Optional[List[str]] = None,
    key: Optional[str] = None
) -> List[str]:
    """
    Multiselect with "Select All" checkbox.
    
    Args:
        label: Widget label
        options: List of options
        default: Default selected options
        key: Optional widget key
    
    Returns:
        List of selected options
    """
    col1, col2 = st.columns([4, 1])
    
    with col2:
        select_all = st.checkbox("Select All", key=f"{key}_all" if key else None)
    
    with col1:
        if select_all:
            selected = st.multiselect(label, options, default=options, key=key)
        else:
            selected = st.multiselect(label, options, default=default, key=key)
    
    return selected

def show_data_summary(df: pd.DataFrame, numeric_cols: List[str]):
    """
    Display data summary statistics in columns.
    
    Args:
        df: DataFrame to summarize
        numeric_cols: Numeric columns to analyze
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Proteins", len(df))
    
    with col2:
        st.metric("Samples", len(numeric_cols))
    
    with col3:
        missing_rate = df[numeric_cols].isna().sum().sum() / df[numeric_cols].size * 100
        st.metric("Missing", f"{missing_rate:.1f}%")
    
    with col4:
        conditions = len(set(col[0] for col in numeric_cols if col))
        st.metric("Conditions", conditions)

# ============================================================================
# THEME SELECTOR
# UI component for theme switching
# ============================================================================

def theme_selector(key: str = "theme_select") -> str:
    """
    Create theme selector widget.
    
    Args:
        key: Widget key
    
    Returns:
        Selected theme name
    """
    from helpers.core import get_theme_names
    
    themes = get_theme_names()
    theme_labels = {
        "light": "☀️ Light",
        "dark": "🌙 Dark",
        "colorblind": "🎨 Colorblind-Friendly",
        "journal": "📄 Journal (B&W)"
    }
    
    selected = st.selectbox(
        "Color Theme",
        options=themes,
        format_func=lambda x: theme_labels.get(x, x.title()),
        key=key
    )
    
    return selected

# ============================================================================
# PROGRESS INDICATORS
# Loading and progress widgets
# ============================================================================

def show_progress(message: str, progress: float):
    """
    Display progress bar with message.
    
    Args:
        message: Status message
        progress: Progress value (0.0 to 1.0)
    """
    st.progress(progress, text=message)

def show_spinner(message: str = "Loading..."):
    """
    Context manager for spinner.
    
    Args:
        message: Loading message
    
    Returns:
        Streamlit spinner context
    """
    return st.spinner(message)
