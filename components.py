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
    """Inject global CSS including navigation styles."""
    st.markdown("""
    <style>
        /* Hide default Streamlit navigation */
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }

        /* Global font */
        body, .stMarkdown, .stText {
            font-family: Arial, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Navigation bar */
        .nav-container {
            background: linear-gradient(90deg, #E71316 0%, #A6192E 100%);
            padding: 0;
            margin: -1rem -1rem 2rem -1rem;
            width: calc(100% + 2rem);
        }
        .nav-inner {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
        }
        .nav-brand {
            color: white;
            font-size: 18pt;
            font-weight: bold;
            padding: 15px 0;
            text-decoration: none;
        }
        .nav-links {
            display: flex;
            gap: 0;
        }
        .nav-link {
            color: white;
            text-decoration: none;
            padding: 20px 25px;
            font-size: 11pt;
            font-weight: 500;
            transition: background-color 0.2s;
            border-bottom: 3px solid transparent;
        }
        .nav-link:hover {
            background-color: rgba(255,255,255,0.1);
        }
        .nav-link.active {
            background-color: rgba(255,255,255,0.15);
            border-bottom: 3px solid white;
        }

        /* Status indicator in nav */
        .nav-status {
            display: flex;
            gap: 15px;
            align-items: center;
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
            opacity: 0.9;
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

        /* Header banner */
        .header-banner {
            background: linear-gradient(90deg, #E71316 0%, #A6192E 100%);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .header-banner h1 {
            color: white;
            margin: 0;
            font-size: 28pt;
        }
        .header-banner p {
            color: white;
            margin: 5px 0 0 0;
            opacity: 0.9;
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
            margin-top: 40px;
        }
    </style>
    """, unsafe_allow_html=True)


def render_navbar(active_page: str = "home"):
    """Render horizontal navigation bar."""

    # Get data status for indicators
    protein_loaded = st.session_state.get("protein_data") is not None
    peptide_loaded = st.session_state.get("peptide_data") is not None

    protein_dot = "active" if protein_loaded else "inactive"
    peptide_dot = "active" if peptide_loaded else "inactive"

    # Navigation HTML
    nav_html = f"""
    <div class="nav-container">
        <div class="nav-inner">
            <span class="nav-brand">Proteomics Analysis</span>
            <div class="nav-links">
                <a href="/" target="_self" class="nav-link {'active' if active_page == 'home' else ''}">Home</a>
                <a href="/1_Data_Upload" target="_self" class="nav-link {'active' if active_page == 'upload' else ''}">Data Upload</a>
                <a href="/2_EDA" target="_self" class="nav-link {'active' if active_page == 'eda' else ''}">EDA</a>
            </div>
            <div class="nav-status">
                <span class="status-text"><span class="status-dot {protein_dot}"></span>Protein</span>
                <span class="status-text"><span class="status-dot {peptide_dot}"></span>Peptide</span>
            </div>
        </div>
    </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)


def render_footer():
    """Render page footer."""
    st.markdown("""
    <div class="footer">
        <p><strong>For research use only</strong></p>
        <p>© 2024 Thermo Fisher Scientific Inc. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
