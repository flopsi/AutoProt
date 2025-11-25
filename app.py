import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
# Set page configst.set_page_config(
    page_title="ProteoFlow - Intelligent Proteomics",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded")
# Import custom modules (these will be in separate files)from utils.data_generator import generate_mock_data
from utils.analysis import process_data, calculate_stats
from components.plots import create_volcano_plot
from components.tables import render_data_table
from components.stats import render_stats_cards
from services.gemini_service import analyze_proteins, chat_with_data
# Initialize session statedef init_session_state():
    if 'view' not in st.session_state:
        st.session_state.view = 'upload'    if 'data' not in st.session_state:
        st.session_state.data = None    if 'p_val_cutoff' not in st.session_state:
        st.session_state.p_val_cutoff = 1.3    if 'fc_cutoff' not in st.session_state:
        st.session_state.fc_cutoff = 1.0    if 'selected_protein' not in st.session_state:
        st.session_state.selected_protein = None    if 'experiment_context' not in st.session_state:
        st.session_state.experiment_context = ""    if 'ai_report' not in st.session_state:
        st.session_state.ai_report = ""    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_input' not in st.session_state:
        st.session_state.chat_input = ""init_session_state()
# Custom CSS for stylingdef load_custom_css():
    st.markdown("""    <style>    /* Main container */    .main {        background-color: #f8fafc;    }    /* Sidebar styling */    [data-testid="stSidebar"] {        background-color: #0f172a;    }    [data-testid="stSidebar"] .element-container {        color: #cbd5e1;    }    /* Headers */    .big-header {        font-size: 2rem;        font-weight: bold;        color: #1e293b;        margin-bottom: 0.5rem;    }    .sub-header {        font-size: 0.875rem;        color: #64748b;        margin-bottom: 2rem;    }    /* Cards */    .stat-card {        background: white;        padding: 1.5rem;        border-radius: 1rem;        box-shadow: 0 1px 3px rgba(0,0,0,0.1);        border: 1px solid #e2e8f0;    }    .protein-detail-card {        background: #eef2ff;        border: 1px solid #c7d2fe;        border-radius: 0.75rem;        padding: 1.5rem;        margin: 1rem 0;    }    /* Buttons */    .stButton>button {        border-radius: 0.5rem;        font-weight: 500;        transition: all 0.2s;    }    .stButton>button:hover {        transform: translateY(-1px);        box-shadow: 0 4px 6px rgba(0,0,0,0.1);    }    /* Chat messages */    .chat-message {        padding: 0.75rem 1rem;        border-radius: 1rem;        margin: 0.5rem 0;        max-width: 80%;    }    .user-message {        background-color: #4f46e5;        color: white;        margin-left: auto;        border-bottom-right-radius: 0.25rem;    }    .assistant-message {        background-color: white;        color: #1e293b;        border: 1px solid #e2e8f0;        border-bottom-left-radius: 0.25rem;    }    /* Upload area */    .upload-container {        background: white;        padding: 3rem;        border-radius: 1rem;        box-shadow: 0 4px 6px rgba(0,0,0,0.1);        border: 1px solid #e2e8f0;        text-align: center;        margin: 2rem auto;        max-width: 800px;    }    /* Hide Streamlit branding */    #MainMenu {visibility: hidden;}    footer {visibility: hidden;}    </style>    """, unsafe_allow_html=True)
load_custom_css()
# Sidebarwith st.sidebar:
    st.markdown("### 🧪 ProteoFlow")
    st.markdown("<p style='font-size: 0.75rem; color: #64748b;'>Intelligent Proteomics</p>", unsafe_allow_html=True)
    st.divider()
    # Navigation    st.markdown("#### Navigation")
    if st.button("📤 Data Input", use_container_width=True,
                 type="primary" if st.session_state.view == 'upload' else "secondary"):
        st.session_state.view = 'upload'        st.rerun()
    if st.button("📊 Dashboard", use_container_width=True,
                 disabled=st.session_state.data is None,
                 type="primary" if st.session_state.view == 'dashboard' else "secondary"):
        st.session_state.view = 'dashboard'        st.rerun()
    if st.button("📝 AI Report", use_container_width=True,
                 disabled=st.session_state.data is None,
                 type="primary" if st.session_state.view == 'report' else "secondary"):
        st.session_state.view = 'report'        st.rerun()
    # Parameters (only show on dashboard)    if st.session_state.view == 'dashboard' and st.session_state.data is not None:
        st.divider()
        st.markdown("#### ⚙️ Parameters")
        st.session_state.p_val_cutoff = st.slider(
            "P-value Cutoff (-log10)",
            min_value=0.0,
            max_value=5.0,
            value=st.session_state.p_val_cutoff,
            step=0.1,
            help="Threshold for statistical significance"        )
        st.session_state.fc_cutoff = st.slider(
            "Log2 FC Cutoff",
            min_value=0.0,
            max_value=3.0,
            value=st.session_state.fc_cutoff,
            step=0.1,
            help="Threshold for fold change significance"        )
    st.divider()
    st.markdown("<p style='text-align: center; font-size: 0.75rem; color: #64748b;'>v1.0.0 • Streamlit • GenAI</p>",
                unsafe_allow_html=True)
# Main content areadef render_upload_view():
    st.markdown("<div class='big-header'>Upload Data</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>{st.session_state.experiment_context or 'No experiment context set.'}</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📤 Import Proteomics Data")
        st.markdown("Upload a CSV or TSV file containing your MaxQuant or FragPipe output.")
        st.markdown("**Required columns:** Gene, Fold Change, P-value")
        st.markdown("")
        # File upload        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'tsv'], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                # Try to read the file                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, sep='\t')
                # For demo, we'll use mock data but could validate uploaded data here                st.session_state.data = generate_mock_data(500)
                st.session_state.view = 'dashboard'                st.success("Data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        st.markdown("**OR**")
        if st.button("Load Demo Dataset", use_container_width=True, type="secondary"):
            st.session_state.data = generate_mock_data(500)
            st.session_state.experiment_context = "Comparison of drug treated (compound X) vs DMSO control in HeLa cells, 24h exposure."            st.session_state.view = 'dashboard'            st.rerun()
        st.markdown("---")
        # Experiment context        st.markdown("##### Experimental Context (Optional)")
        context = st.text_area(
            "Context",
            value=st.session_state.experiment_context,
            placeholder="e.g. Comparison of wild-type vs knockout mice liver tissue...",
            height=80,
            label_visibility="collapsed",
            help="This context helps the AI generate more relevant biological insights."        )
        st.session_state.experiment_context = context
    st.markdown("</div>", unsafe_allow_html=True)
def render_dashboard_view():
    st.markdown("<div class='big-header'>Exploratory Analysis</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>{st.session_state.experiment_context or 'No experiment context set.'}</div>",
                unsafe_allow_html=True)
    # Process data with current thresholds    processed_data = process_data(
        st.session_state.data,
        st.session_state.p_val_cutoff,
        st.session_state.fc_cutoff
    )
    # Stats cards    render_stats_cards(processed_data)
    # Main content: Volcano plot and table    col1, col2 = st.columns([2, 1])
    with col1:
        # Volcano plot        fig = create_volcano_plot(
            processed_data,
            st.session_state.p_val_cutoff,
            st.session_state.fc_cutoff
        )
        # Handle click events        selected_points = st.plotly_chart(fig, use_container_width=True,
                                         on_select="rerun", key="volcano_plot")
        # Selected protein details        if st.session_state.selected_protein is not None:
            protein = st.session_state.selected_protein
            st.markdown(f"""            <div class='protein-detail-card'>                <h3 style='color: #4338ca; margin-bottom: 0.5rem;'>⚡ Protein Details: {protein['gene']}</h3>                <p style='color: #3730a3; font-size: 0.875rem; line-height: 1.5;'>{protein['description']}</p>                <div style='margin-top: 1rem; display: flex; gap: 1rem;'>                    <div style='background: white; padding: 0.5rem 1rem; border-radius: 0.375rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>                        <span style='color: #64748b;'>Log2FC: </span>                        <span style='font-family: monospace; font-weight: bold;'>{protein['log2FoldChange']:.3f}</span>                    </div>                    <div style='background: white; padding: 0.5rem 1rem; border-radius: 0.375rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>                        <span style='color: #64748b;'>-Log10P: </span>                        <span style='font-family: monospace; font-weight: bold;'>{protein['negLog10PValue']:.3f}</span>                    </div>                </div>            </div>            """, unsafe_allow_html=True)
        else:
            st.info("👆 Select a data point on the plot to view details")
    with col2:
        # Data table        render_data_table(processed_data)
        # AI Analysis button        st.markdown("---")
        with st.container():
            st.markdown("##### ⚡ AI Analysis")
            st.markdown("Ready to interpret these findings? Send the significant proteins to Gemini for biological context.")
            if st.button("Generate Full Report", use_container_width=True, type="primary"):
                st.session_state.view = 'report'                st.rerun()
def render_report_view():
    st.markdown("<div class='big-header'>Insights & Report</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>{st.session_state.experiment_context or 'No experiment context set.'}</div>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📝 Automated Analysis Report")
        # Generate report button        if not st.session_state.ai_report:
            if st.button("🔄 Run Analysis", type="primary"):
                with st.spinner("Querying Gemini knowledge base..."):
                    # Get significant proteins                    processed_data = process_data(
                        st.session_state.data,
                        st.session_state.p_val_cutoff,
                        st.session_state.fc_cutoff
                    )
                    significant = processed_data[processed_data['significance'] != 'NS']
                    top_proteins = significant.nlargest(15, 'negLog10PValue')
                    # Generate report                    report = analyze_proteins(
                        top_proteins.to_dict('records'),
                        st.session_state.experiment_context
                    )
                    st.session_state.ai_report = report
                    st.rerun()
        # Display report        if st.session_state.ai_report:
            st.markdown(st.session_state.ai_report)
        else:
            st.info("No report generated yet. Click 'Run Analysis' to generate.")
    with col2:
        st.markdown("#### 💬 Chat with Data")
        # Chat history        chat_container = st.container(height=400)
        with chat_container:
            if len(st.session_state.chat_history) == 0:
                st.markdown("""                <div style='text-align: center; color: #94a3b8; padding: 2rem; font-size: 0.875rem;'>                    <p>Ask questions like:</p>                    <ul style='list-style: none; padding: 0; margin-top: 0.5rem;'>                        <li>"What is the function of the top upregulated protein?"</li>                        <li>"Are there any mitochondrial proteins changed?"</li>                    </ul>                </div>                """, unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""                    <div style='display: flex; justify-content: flex-end;'>                        <div class='chat-message user-message'>{msg['text']}</div>                    </div>                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""                    <div style='display: flex; justify-content: flex-start;'>                        <div class='chat-message assistant-message'>{msg['text']}</div>                    </div>                    """, unsafe_allow_html=True)
        # Chat input        chat_input = st.text_input("Ask a follow-up question...", key="chat_text_input")
        if st.button("Send", type="primary") or (chat_input and st.session_state.get('enter_pressed')):
            if chat_input.strip():
                # Add user message                st.session_state.chat_history.append({
                    'role': 'user',
                    'text': chat_input,
                    'timestamp': datetime.now().isoformat()
                })
                # Generate response                with st.spinner("Thinking..."):
                    processed_data = process_data(
                        st.session_state.data,
                        st.session_state.p_val_cutoff,
                        st.session_state.fc_cutoff
                    )
                    stats = calculate_stats(processed_data)
                    context = f"Experiment: {st.session_state.experiment_context}. "                    context += f"Total: {stats['total']}, Up: {stats['up']}, Down: {stats['down']}. "                    if st.session_state.selected_protein:
                        context += f"Selected Protein: {st.session_state.selected_protein['gene']}."                    else:
                        context += "Selected Protein: None."                    response = chat_with_data(
                        st.session_state.chat_history[:-1],
                        chat_input,
                        context
                    )
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'text': response,
                        'timestamp': datetime.now().isoformat()
                    })
                st.rerun()
# Main routingif st.session_state.view == 'upload':
    render_upload_view()
elif st.session_state.view == 'dashboard':
    render_dashboard_view()
elif st.session_state.view == 'report':
    render_report_view()
