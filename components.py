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
    """Inject global CSS."""
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }

        body, .stMarkdown, .stText {
            font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Header bar (display only, no nav links) */
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

        /* Button styling */
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

        /* Module cards */
        .module-card {
            background-color: #E2E3E4;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #E71316;
            margin-bottom: 15px;
            min-height: 120px;
        }
        .module-card h3 { margin: 0 0 10px 0; color: #54585A; }
        .module-card p { margin: 0; color: #54585A; }

        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .badge-protein { background-color: #262262; color: white; }
        .badge-peptide { background-color: #EA7600; color: white; }

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
    """Render header bar with status indicators."""
    protein_loaded = st.session_state.get("protein_data") is not None
    peptide_loaded = st.session_state.get("peptide_data") is not None

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
    """Render bottom navigation buttons."""
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if back_page:
            if st.button("← Back", use_container_width=True):
                st.switch_page(back_page)
        else:
            st.button("← Back", use_container_width=True, disabled=True)

    with col2:
        if st.button("Home", use_container_width=True):
            st.switch_page("app.py")

    with col3:
        if st.button("Restart", use_container_width=True):
            # Clear all cached data
            keys_to_clear = [
                "protein_data", "peptide_data",
                "protein_index_col", "peptide_index_col",
                "protein_missing_mask", "peptide_missing_mask",
                "upload_key"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.switch_page("app.py")

    with col4:
        if next_page:
            if st.button("Next →", use_container_width=True):
                st.switch_page(next_page)
        else:
            st.button("Next →", use_container_width=True, disabled=True)


def render_footer():
    """Render page footer."""
    st.markdown("""
    <div class="footer">
        <p><strong>For research use only</strong></p>
        <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
