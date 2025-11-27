# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson

# Load data
if "prot_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_df.copy()
c1 = st.session_state.prot_c1.copy()
c2 = st.session_state.prot_c2.copy()
all_reps = c1 + c2

st.title("Protein-Level QC (Schessner et al., 2022)")

# === 1. NORMALITY TESTING ON RAW DATA ===
st.subheader("1. Normality Testing on Raw Data (Shapiro-Wilk)")

transform_options = {
    "Raw": lambda x: x,
    "log₂": lambda x: np.log2(x + 1),
    "log10": lambda x: np.log10(x + 1),
    "Square root": lambda x: np.sqrt(x + 1),
    "Box-Cox": lambda x: boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: yeojohnson(x + 1)[0],
}

results = []
best_transform = "Raw"
best_w = 0

for rep in all_reps:
    raw_vals = df[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8:
        continue
        
    row = {"Replicate": rep}
    
    w_raw, p_raw = stats.shapiro(raw_vals)
    row["Raw W"] = f"{w_raw:.4f}"
    row["Raw p"] = f"{p_raw:.2e}"
    
    rep_best = "Raw"
    rep_w = w_raw
    
    for name, func in transform_options.items():
        if name == "Raw": continue
        try:
            t_vals = func(raw_vals)
            if t_vals is None or np.any(np.isnan(t_vals)): continue
            w, p = stats.shapiro(t_vals)
            row[f"{name} W"] = f"{w:.4f}"
            if w > rep_w:
                rep_w = w
                rep_best = name
        except:
            row[f"{name} W"] = "—"
    
    row["Best Transform"] = rep_best
    row["Best W"] = f"{rep_w:.4f}"
    
    if rep_w > best_w:
        best_w = rep_w
        best_transform = rep_best
        
    results.append(row)

st.table(pd.DataFrame(results))
st.success(f"Recommended transformation: {best_transform} (highest Shapiro-Wilk W)")
st.info("Schessner et al., 2022 — Figure 4 & Section 2.1.1")

# === 2. DATA VIEW & FILTERING PANEL ===
st.subheader("2. Data View & Filtering")

col1, col2, col3 = st.columns(3)

with col1:
    filter_strategy = st.radio(
        "Filtering",
        ["Raw data", "Low intensity", "±2σ filtered", "Combined"],
        index=0
    )

with col2:
    transformation_choice = st.radio(
        "Transformation",
        ["Raw data", f"Recommended ({best_transform})"],
        index=1
    )

with col3:
    available_species = ["All proteins"]
    if "Species" in df.columns:
        available_species += sorted(df["Species"].dropna().unique().tolist())
    species_choice = st.radio("Species", available_species, index=0)

# === APPLY ALL SELECTIONS ===
df_current = df.copy()

# Filtering
log10_full = np.log10(df_current[all_reps].replace(0, np.nan))

if filter_strategy in ["Low intensity", "Combined"]:
    mask = (log10_full >= 0.5).all(axis=1)
    df_current = df_current[mask]

if filter_strategy in ["±2σ filtered", "Combined"]:
    mask = pd.Series(True, index=df_current.index)
    log10_current = np.log10(df_current[all_reps].replace(0, np.nan))
    for rep in all_reps:
        vals = log10_current[rep].dropna()
        if len(vals) == 0: continue
        mean, std = vals.mean(), vals.std()
        mask &= (log10_current[rep] >= mean - 2*std) & (log10_current[rep] <= mean + 2*std)
    df_current = df_current[mask]

# Transformation
if transformation_choice == f"Recommended ({best_transform})":
    func = transform_options[best_transform]
    df_current[all_reps] = df_current[all_reps].apply(func)

# Species
if species_choice != "All proteins":
    df_current = df_current[df_current["Species"] == species_choice]

# === 3. PROTEIN COUNT TABLE ===
st.subheader("3. Protein Counts")
count_data = []
count_data.append({"Species": "Total", "Proteins": len(df_current)})
if "Species" in df_current.columns:
    for sp in df_current["Species"].value_counts().index:
        count_data.append({"Species": sp, "Proteins": df_current["Species"].value_counts()[sp]})
st.table(pd.DataFrame(count_data))

# === 4. KS TEST ON SELECTED DATA (YES/NO ONLY) ===
st.subheader("4. Replicate Similarity (Kolmogorov–Smirnov Test)")

# Use filtered + transformed data
test_data = df_current

ks_results = []
for i, rep1 in enumerate(all_reps):
    for j in range(i+1, len(all_reps)):
        rep2 = all_reps[j]
        vals1 = test_data[rep1].replace(0, np.nan).dropna()
        vals2 = test_data[rep2].replace(0, np.nan).dropna()
        if len(vals1) < 10 or len(vals2) < 10:
            ks_results.append({"Rep1": rep1, "Rep2": rep2, "Different?": "—"})
            continue
        _, p = stats.ks_2samp(vals1, vals2)
        different = "Yes" if p < 0.05 else "No"
        ks_results.append({"Rep1": rep1, "Rep2": rep2, "Different?": different})

st.table(pd.DataFrame(ks_results))

# === 5. PROCEED ===
st.markdown("### Proceed to Differential Analysis")
if st.button("Accept & Continue", type="primary"):
    st.session_state.intensity_transformed = df_current[all_reps]
    st.session_state.df_filtered = df_current
    st.session_state.transform_applied = best_transform if "Recommended" in transformation_choice else "Raw"
    st.session_state.qc_accepted = True
    st.success("Data ready for differential analysis!")
    st.balloons()
