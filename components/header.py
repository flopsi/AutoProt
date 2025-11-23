import streamlit as st

def render_header():
    st.markdown('''
    <div style="
        background-color: #E71316;
        padding: 20px 40px;
        color: white;
        margin-bottom: 30px;
    ">
        <h1 style="margin: 0; font-size: 28px; font-weight: 600;">
            Thermo Fisher Scientific
        </h1>
        <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.95;">
            Proteomics Data Analysis Platform
        </p>
    </div>
    ''', unsafe_allow_html=True)