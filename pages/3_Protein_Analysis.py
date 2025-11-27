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

# === DETECT SPECIES ===
species_list = ["HUMAN","MOUSE","RAT","ECOLI","BOVIN","YEAST","RABIT","CANFA","MACMU","PANTR"]
def find_species_col(df):
    pattern = "|".join(species_list)
    for c in df.columns:
        if c in all_reps: continue
        if df[c].astype(str).str.upper().str.contains(pattern).any():
            return c
    return None

sp_col = find_species_col(df)

if sp_col and sp_col != "Not found":
    df["Species"] = df[sp_col].astype(str).str.upper().apply(
        lambda x: next((s for s in species_list if s in x), "Other")
    )
    detected_species = df["Species"].dropna().unique()
else:
    detected_species = ["HUMAN"]  # fallback
    df["Species"] = "HUMAN"

# === USER SELECTS CONSTANT SPECIES ===
st.subheader("1. Select Constant (Reference) Species for QC")
if len(detected_species) > 1:
    constant_species = st.selectbox(
        "Multiple species detected — which is the constant reference?",
        options=sorted(detected_species),
        index=0
    )
    st.info(f"KS test will be performed **only on {constant_species}** proteins")
else:
    constant_species = detected_species[0]
    st.success(f"Only one species detected: **{constant_species}** — using as reference")

# Filter to constant species
df_const = df[df["Species"] == constant_species].copy()
if len(df_const) == 0:
    st.error(f"No proteins found for species {constant_species}")
    st.stop()

st.write(f"Using **{len(df_const):,}** {constant_species} proteins for distribution testing")

# === 6 INDIVIDUAL DENSITY PLOTS (RAW) ===
st.subheader("2. Individual Intensity Density Plots (Raw — Constant Species Only)")

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

# === KS TEST ON CONSTANT SPECIES ONLY ===
st.subheader("3. Distribution Similarity Test (KS) — Constant Species Only")

significant_pairs = []
for rep1, rep2 in combinations(all_reps, 2):
    d1 = raw_data[rep1].dropna()
    d2 = raw_data[rep2].dropna()
    if len(d1) > 1 and len(d2) > 1:
        _, p = stats.ks_2samp(d1, d2)
        if p < 0.05:
            significant_pairs.append(f"{rep1} vs {rep2}")

if significant_pairs:
    st.error(f"**Significant differences detected** in {constant_species} distributions:\n" + " • ".join(significant_pairs))
    st.info("This may indicate technical bias — check injection, digestion, or labeling")
else:
    st.success(f"**All {constant_species} replicates have similar distributions** — excellent technical quality!")

# === LOG₂ TRANSFORMATION & FINAL PLOTS ===
st.markdown("### 4. Standard log₂ Transformation")
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
fig_overlay.update_layout(barmode="overlay", height=500, xaxis_title="log₂(Intensity)", yaxis_title="Density")
st.plotly_chart(fig_overlay, use_container_width=True)

# Violin + box
st.subheader("Violin + Box Plot — log₂(Intensity)")
melted = log_data.melt(var_name="Replicate", value_name="log₂(Intensity)").dropna()
melted["Condition"] = melted["Replicate"].apply(lambda x: "A" if x in c1 else "B")
melted["Replicate"] = pd.Categorical(melted["Replicate"], all_reps, ordered=True)

fig_violin = px.violin(
    melted, x="Replicate", y="log₂(Intensity)", color="Condition",
    color_discrete_map={"A": "#E71316", "B": "#1f77b4"},
    box=True, points=False, violinmode="overlay"
)
fig_violin.update_traces(box_visible=True, meanline_visible=True)
fig_violin.update_layout(height=650)
st.plotly_chart(fig_violin, use_container_width=True)

st.success("**log₂ transformation applied — ready for analysis**")

if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/4_Differential_Analysis.py")
