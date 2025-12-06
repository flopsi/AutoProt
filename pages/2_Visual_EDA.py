"""
pages/2_Visual_EDA.py

Visual Exploratory Data Analysis page
Modular design with efficient caching and navigation controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from helpers.core import ProteinData, get_theme
from helpers.audit import log_event
from helpers.analysis import detect_conditions_from_columns, group_columns_by_condition

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Visual EDA", layout="wide", page_icon="📊")

# ============================================================================
# CACHE MANAGEMENT FUNCTIONS
# ============================================================================

def clear_page_cache():
    """Clear only Visual EDA page-specific session state."""
    keys_to_clear = [
        'eda_transformation',
        'eda_log_base',
        'eda_normalization',
        'eda_
