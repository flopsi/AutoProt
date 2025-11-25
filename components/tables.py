"""
Table rendering components
"""

import streamlit as st
import pandas as pd


def render_data_table(df: pd.DataFrame, title: str = "Data Table"):
    """
    Render an interactive data table
    
    Args:
        df: DataFrame to display
        title: Title for the table
    """
    st.subheader(title)
    st.dataframe(df, use_container_width=True, height=400)
