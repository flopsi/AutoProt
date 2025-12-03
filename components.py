import streamlit as st

COLORS = {
    "red": "#E71316",
    "dark_red": "#A6192E", 
    "gray": "#54585A",
    "light_gray": "#E2E3E4",
    "white": "#FFFFFF",
    "navy": "#262262",
    "orange": "#EA7600",
}


def inject_custom_css():
    """Inject global CSS for consistent styling."""
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }

        body, .stMarkdown, .stText {
            font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Header */
        .header-bar {
            background: linear-gradient(90deg, #E71316 0%, #A6192E 100%);
            padding: 15px 20px;
            margin: -1rem -1rem 2rem -1rem;
            width: calc(100% + 2rem);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header-brand {
            color: white;
            font-size: 18pt;
            font-weight: bold;
        }
        .header-status {
            display: flex;
            gap: 15px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-dot.active { background-color: #B5BD00; }
        .status-dot.inactive { background-color: rgba(255,255,255,0.4); }
        .status-text {
            color: white;
            font-size: 10pt;
        }

        /* Buttons */
        .stButton > button {
            background-color: #E71316;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #A6192E;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #54585A;
            font-size: 12px;
            padding: 20px 0;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render header with protein/peptide status indicators."""
    # Consistent with other pages
    protein_loaded = st.session_state.get("protein_model") is not None
    peptide_loaded = st.session_state.get("peptide_model") is not None

    protein_dot = "active" if protein_loaded else "inactive"
    peptide_dot = "active" if peptide_loaded else "inactive"

    st.markdown(f"""
    <div class="header-bar">
        <span class="header-brand">Proteomics Analysis</span>
        <div class="header-status">
            <span class="status-text"><span class="status-dot {protein_dot}"></span>Protein</span>
            <span class="status-text"><span class="status-dot {peptide_dot}"></span>Peptide</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_navigation(back_page=None, next_page=None):
    """Render bottom navigation with Back, Home, Restart, Next buttons."""
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        if back_page:
            if st.button("← Back", use_container_width=True, type="secondary"):
                st.switch_page(back_page)
        else:
            st.button("← Back", use_container_width=True, disabled=True)

    with col2:
        if st.button("🏠 Home", use_container_width=True, type="secondary"):
            st.switch_page("app.py")

    with col3:
        if st.button("🔄 Restart", use_container_width=True, type="secondary"):
            _clear_session_state()
            st.switch_page("app.py")

    with col4:
        if next_page:
            if st.button("Next →", use_container_width=True, type="primary"):
                st.switch_page(next_page)
        else:
            st.button("Next →", use_container_width=True, disabled=True)


def _clear_session_state():
    """Clear all cached data for restart."""
    keys_to_clear = [
        "protein_model", "peptide_model",
        "protein_index_col", "peptide_index_col", 
        "protein_species_col", "peptide_species_col",
        "peptide_seq_col",
        "protein_missing_mask", "peptide_missing_mask",
        "upload_key", "raw_df", "column_renames", "selected_quant_cols",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.cache_data.clear()


def render_footer():
    """Render standardized page footer."""
    st.markdown("""
    <div class="footer">
        <p><strong>For research use only</strong></p>
        <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
