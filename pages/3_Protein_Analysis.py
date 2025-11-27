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

st.title("Protein-Level QC & Data Transformation")

# === 3. DATA VIEW & FILTERING — EXACTLY LIKE SCHESSNER ET AL., 2022 FIGURE 4 ===
st.subheader("3. Data View & Filtering (Schessner et al., 2022 Figure 4)")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("**Filtering Strategy**")
    filter_strategy = st.radio(
        "Apply filtering:",
        ["Raw data", "Low intensity filtered", "±2σ filtered", "Combined"],
        index=0,
        key="filter_strategy"
    )

with col2:
    st.markdown("**Transformation**")
    # Normality test on raw data (once)
    transform_options = {
        "Raw": lambda x: x,
        "log₂": lambda x: np.log2(x + 1),
        "log₁₀": lambda x: np.log10(x + 1),
        "Square root": lambda x: np.sqrt(x + 1),
        "Box-Cox": lambda x: boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
        "Yeo-Johnson": lambda x: yeojohnson(x + 1)[0],
    }

    best_transform = "Raw"
    best_w = 0
    for rep in all_reps:
        raw_vals = df[rep].replace(0, np.nan).dropna()
        if len(raw_vals) < 8: continue
        w_raw, _ = stats.shapiro(raw_vals)
        for name, func in transform_options.items():
            if name == "Raw": continue
            try:
                t_vals = func(raw_vals)
                if t_vals is None or np.any(np.isnan(t_vals)): continue
                w, _ = stats.shapiro(t_vals)
                if w > best_w:
                    best_w = w
                    best_transform = name
            except: pass

    transformation_choice = st.radio(
        "Apply transformation:",
        ["Raw data", f"Recommended ({best_transform})"],
        index=1
    )

with col3:
    st.markdown("**Species**")
    available_species = ["All proteins"]
    if "Species" in df.columns:
        available_species += df["Species"].dropna().unique().tolist()
    species_choice = st.radio(
        "Show species:",
        available_species,
        index=0
    )

# === APPLY FILTERING + TRANSFORMATION + SPECIES FILTER ===
df_current = df.copy()

# 1. Filtering
log10_full = np.log10(df_current[all_reps].replace(0, np.nan))

if filter_strategy in ["Low intensity filtered", "Combined"]:
    mask = pd.Series(True, index=df_current.index)
    for rep in all_reps:
        mask &= (log10_full[rep] >= 0.5)
    df_current = df_current[mask]

if filter_strategy in ["±2σ filtered", "Combined"]:
    mask = pd.Series(True, index=df_current.index)
    log10_current = np.log10(df_current[all_reps].replace(0, np.nan))
    for rep in all_reps:
        vals = log10_current[rep].dropna()
        if len(vals) == 0: continue
        mean = vals.mean()
        std = vals.std()
        mask &= (log10_current[rep] >= mean - 2*std) & (log10_current[rep] <= mean + 2*std)
    df_current = df_current[mask]

# 2. Transformation
if transformation_choice == f"Recommended ({best_transform})":
    func = transform_options[best_transform]
    df_current[all_reps] = df_current[all_reps].apply(func)

# 3. Species filter
if species_choice != "All proteins":
    df_current = df_current[df_current["Species"] == species_choice]

# === 4. 6 DENSITY PLOTS — EXACTLY LIKE SCHESSNER ET AL., 2022 FIGURE 4 ===
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

# === 5. PROTEIN COUNT TABLE ===
st.subheader("Protein Counts After Filtering")
count_data = [{"Species": "All proteins", "Count": len(df_current)}]
if "Species" in df_current.columns:
    for sp in df_current["Species"].dropna().unique():
        count_data.append({"Species": sp, "Count": len(df_current[df_current["Species"] == sp])})
st.table(pd.DataFrame(count_data))

# === 6. ACCEPT ===
if st.button("Accept & Proceed to Differential Analysis", type="primary"):
    st.session_state.intensity_transformed = df_current[all_reps]
    st.session_state.df_filtered = df_current
    st.session_state.transform_applied = best_transform if "Recommended" in transformation_choice else "Raw"
    st.session_state.qc_accepted = True
    st.success("Ready for differential analysis!")
    st.balloons()
