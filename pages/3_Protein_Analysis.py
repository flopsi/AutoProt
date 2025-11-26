# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
from shared import restart_button, debug

def ss(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

st.set_page_config(page_title="Protein Analysis", layout="wide")

st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1 {margin:0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>Protein-Level Exploratory Analysis</h1></div>', unsafe_allow_html=True)

# Load data
if ss("prot_df") is None:
    st.error("No protein data found! Please go back and upload a file.")
    st.stop()

df = ss("prot_df")
c1 = ss("prot_c1")
c2 = ss("prot_c2")

st.success(f"Analyzing {len(df):,} proteins • A: {len(c1)} reps • B: {len(c2)} reps")

# Example plots
col1, col2 = st.columns(2)
with col1:
    st.subheader("Intensity Box Plot")
    plot_data = df[c1 + c2].melt()
    fig = px.box(plot_data, x="variable", y="value", color="variable", log_y=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("PCA")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = df[c1 + c2].fillna(0)
    pca = PCA(n_components=2)
    components = pca.fit_transform(StandardScaler().fit_transform(X))
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Condition"] = ["A"] * len(c1) + ["B"] * len(c2)
    fig2 = px.scatter(pca_df, x="PC1", y="PC2", color="Condition")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Volcano Plot, Heatmaps, etc. coming soon!")

restart_button()
