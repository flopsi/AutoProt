# pages/4_Peptide_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
from shared import restart_button

def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.set_page_config(page_title="Peptide Analysis", layout="wide")

st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>Peptide-Level Exploratory Analysis</h1></div>', unsafe_allow_html=True)

if ss("pept_df") is None:
    st.error("No peptide data! Please upload in Peptide Import first.")
    st.stop()

df = ss("pept_df")
c1 = ss("pept_c1")
c2 = ss("pept_c2")

st.success(f"Analyzing {len(df):,} peptides")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Peptide Intensities")
    fig = px.violin(df[c1+c2].melt(), x="variable", y="value", color="variable", box=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Peptides per Protein")
    if df.index.name and "Protein" in df.columns:
        counts = df.groupby("Protein").size().sort_values(ascending=False).head(20)
        fig = px.bar(x=counts.index, y=counts.values)
        st.plotly_chart(fig, use_container_width=True)

restart_button()
