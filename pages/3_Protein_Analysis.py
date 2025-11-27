# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from itertools import combinations

# Load data
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2
all_reps = c1 + c2

st.title("Protein-Level Exploratory Analysis")
st.success(f"Analyzing {len(df):,} proteins • {len(c1)} vs {len(c2)} replicates")

# === ROBUST SPECIES DETECTION (FIXED) ===
def extract_species(text):
    if pd.isna(text):
        return "Other"
    text = str(text).upper()
    species_keywords = {
        "HUMAN": ["HUMAN", "HOMO SAPIENS", "HSA"],
        "MOUSE": ["MOUSE", "MUS MUSCULUS", "MMU"],
        "RAT": ["RAT", "RATTUS NORVEGICUS", "RNO"],
        "ECOLI": ["ECOLI", "ESCHERICHIA COLI"],
        "BOVIN": ["BOVIN", "BOVINE", "BOS TAURUS"],
        "YEAST": ["YEAST", "SACCHAROMYCES", "SCE"],
        "RABIT": ["RABBIT", "RABIT", "OCU"],
        "CANFA": ["DOG", "CANIS", "CANFA"],
        "MACMU": ["MACACA", "RHESUS", "MACMU"],
        "PANTR": ["CHIMP", "PANTR"],
    }
    for species, keywords in species_keywords.items():
        if any(kw in text for kw in keywords):
            return species
    return "Other"

# Find species column
species_col = None
for col in df.columns:
    if col in all_reps: continue
    sample = df[col].dropna().astype(str).str.upper()
    if sample.str.contains("HUMAN|MOUSE|RAT|ECOLI|BOVIN|YEAST|RABIT|CANFA|MACMU|PANTR").any():
        species_col = col
        break

if species_col:
    df["Detected_Species"] = df[species_col].apply(extract_species)
    detected = df["Detected_Species"].value_counts()
    st.write("**Detected species:**")
    st.dataframe(detected.reset_index().rename(columns={"index": "Species", "Detected_Species": "Count"}), use_container_width=True)
else:
    df["Detected_Species"] = "HUMAN"  # default
    st.info("No species column found — assuming HUMAN")

# === USER SELECTS CONSTANT SPECIES ===
unique_species = sorted(df["Detected_Species"].unique())
if len(unique_species) > 1:
    constant_species = st.selectbox(
        "Select constant (reference) species for QC",
        options=unique_species,
        index=0
    )
else:
    constant_species = unique_species[0]
    st.success(f"Only one species: **{constant_species}**")

df_const = df[df["Detected_Species"] == constant_species].copy()
st.write(f"Using **{len(df_const):,}** {constant_species} proteins for distribution testing")

# === 6 INDIVIDUAL DENSITY PLOTS (RAW) ===
st.subheader("Individual Intensity Density Plots (Raw — Constant Species)")

raw_data = df_const[all_reps].replace(0, np.nan)

row1 = st.columns(3)
row2 = st.columns(3)

for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=raw_data[rep].dropna(),
            nbinsx=100,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        median = raw_data[rep].median()
        fig.add_vline(x=median, line_dash="dash", line_color="black")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="Raw Intensity",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === KS TEST ON CONSTANT SPECIES ===
st.subheader("Distribution Similarity Test (KS) — Constant Species Only")

significant_pairs = []
for rep1, rep2 in combinations(all_reps, 2):
    d1 = raw_data[rep1].dropna()
    d2 = raw_data[rep2].dropna()
    if len(d1) > 1 and len(d2) > 1:
        _, p = stats.ks_2samp(d1, d2)
        if p < 0.05:
            significant_pairs.append(f"{rep1} vs {rep2}")

if significant_pairs:
    st.error(f"**Significant differences** in {constant_species} distributions:\n" + " • ".join(significant_pairs))
else:
    st.success(f"**All {constant_species} replicates have similar distributions** — excellent!")

# === LOG₂ PLOTS ===
st.markdown("### log₂ Transformation (Standard)")
log_data = np.log2(df_const[all_reps])

# Overlay histogram
st.subheader("Overlay Histogram — log₂(Intensity)")
fig_overlay = go.Figure()
for rep in all_reps:
    fig_overlay.add_trace(go.Histogram(
        x=log_data[rep].dropna(),
        name=rep,
        opacity=0.6,
        nbinsx=100,
        histnorm="density"
    ))
fig_overlay.update_layout(barmode="overlay", height=500)
st.plotly_chart(fig_overlay, use_container_width=True)

# Violin + box
st.subheader("Violin + Box — log₂(Intensity)")
melted = log_data.melt(var_name="Replicate", value_name="log₂(Intensity)").dropna()
melted["Condition"] = melted["Replicate"].apply(lambda x: "A" if x in c1 else "B")
melted["Replicate"] = pd.Categorical(melted["Replicate"], all_reps, ordered=True)

fig_violin = px.violin(
    melted, x="Replicate", y="log₂(Intensity)", color="Condition",
    color_discrete_map={"A": "#E71316", "B": "#1f77b4"},
    box=True, points=False, violinmode="overlay"
)
fig_violin.update_traces(box_visible=True, meanline_visible=True)
st.plotly_chart(fig_violin, use_container_width=True)

st.success("**log₂ transformation applied — ready for analysis**")

if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Differential_Analysis.py")
