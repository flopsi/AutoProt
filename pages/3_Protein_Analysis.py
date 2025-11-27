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

# === 1. NORMALITY TESTING ON RAW DATA (Schessner et al., 2022 Figure 4) ===
st.subheader("1. Normality Testing on Raw Data (Shapiro-Wilk)")

transform_options = {
    "Raw": lambda x: x,
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
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
st.success(f"**Recommended transformation: {best_transform}** (highest W)")
st.info("Shapiro-Wilk W statistic — Schessner et al., 2022, Figure 4")

# === 2. DATA VIEW & FILTERING PANEL (Schessner et al., 2022 Figure 4) ===
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
if filter_strategy in ["Low intensity", "Combined"]:
    mask = (np.log10(df_current[all_reps].replace(0, np.nan)) >= 0.5).all(axis=1)
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

# === 3. DENSITY PLOTS (Schessner et al., 2022 Figure 4) ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = df_current[rep].replace(0, np.nan).dropna()
        if len(vals) == 0:
            st.write("No data")
            continue
            
        mean = float(vals.mean())
        std = float(vals.std())
        lower = mean - 2*std
        upper = mean + 2*std

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=80,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        fig.add_vrect(x0=lower, x1=upper, fillcolor="white", opacity=0.35, line_width=2)
        fig.add_vline(x=mean, line_dash="dash", line_color="black")
        
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 4. PROTEIN COUNT TABLE (Schessner et al., 2022 Table 1) ===
st.subheader("Protein Counts After Filtering")
count_data = []
count_data.append({"Species": "Total", "Proteins": len(df_current)})
if "Species" in df_current.columns:
    for sp in df_current["Species"].value_counts().index:
        count_data.append({"Species": sp, "Proteins": df_current["Species"].value_counts()[sp]})
st.table(pd.DataFrame(count_data))

# === 5. KS TEST (Yes/No only) ===
st.subheader("Replicate Similarity (Kolmogorov–Smirnov Test)")
ks_results = []
for i, rep1 in enumerate(all_reps):
    for j in range(i+1, len(all_reps)):
        rep2 = all_reps[j]
        vals1 = df_current[rep1].replace(0, np.nan).dropna()
        vals2 = df_current[rep2].replace(0, np.nan).dropna()
        if len(vals1) < 10 or len(vals2) < 10:
            ks_results.append({"Rep1": rep1, "Rep2": rep2, "Different?": "—"})
            continue
        _, p = stats.ks_2samp(vals1, vals2)
        different = "Yes" if p < 0.05 else "No"
        ks_results.append({"Rep1": rep1, "Rep2": rep2, "Different?": different})

st.table(pd.DataFrame(ks_results))

# === 6. ACCEPT ===
if st.button("Accept & Proceed to Differential Analysis", type="primary"):
    st.session_state.intensity_transformed = df_current[all_reps]
    st.session_state.df_filtered = df_current
    st.session_state.transform_applied = best_transform if "Recommended" in transformation_choice else "Raw"
    st.session_state.qc_accepted = True
    st.success("Ready for differential analysis!")
    st.balloons()
