# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
if "prot_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_df.copy()
c1 = st.session_state.prot_c1.copy()
c2 = st.session_state.prot_c2.copy()
all_reps = c1 + c2

st.title("Protein-Level QC & PCA (Schessner et al., 2022 Figure 4)")

# === 1. NORMALITY TESTING ON RAW DATA ===
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
    if len(raw_vals) < 8: continue
        
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

# === 2. DATA VIEW & FILTERING PANEL ===
st.subheader("2. Data View & Filtering")

col1, col2, col3 = st.columns(3)

with col1:
    filter_strategy = st.radio("Filtering", ["Raw data", "Low intensity", "±2σ filtered", "Combined"], index=0)

with col2:
    transformation_choice = st.radio("Transformation", ["Raw data", f"Recommended ({best_transform})"], index=1)

with col3:
    available_species = ["All proteins"]
    if "Species" in df.columns:
        available_species += sorted(df["Species"].dropna().unique().tolist())
    species_choice = st.radio("Species", available_species, index=0)

# === 3. APPLY TRANSFORMATION (FOR PCA & PLOTS) ===
df_pca = df.copy()
if transformation_choice == f"Recommended ({best_transform})":
    func = transform_options[best_transform]
    df_pca[all_reps] = df_pca[all_reps].apply(func)

# === 4. TWO PCA PLOTS (Schessner et al., 2022 Figure 4) ===
st.subheader("PCA: Transformed Data — Without vs With Filtering")

col_left, col_right = st.columns(2)

# Left: Without filtering
with col_left:
    st.markdown("**Without Filtering**")
    df_left = df_pca.copy()
    if species_choice != "All proteins":
        df_left = df_left[df_left["Species"] == species_choice]
    
    X_left = df_left[all_reps].replace(0, np.nan).fillna(df_left[all_reps].median())
    X_left_scaled = StandardScaler().fit_transform(X_left)
    pca_left = PCA(n_components=2)
    pc_left = pca_left.fit_transform(X_left_scaled)
    
    fig_left = go.Figure()
    for i, rep in enumerate(all_reps):
        color = "#E71316" if rep in c1 else "#1f77b4"
        fig_left.add_trace(go.Scatter(
            x=pc_left[:, 0], y=pc_left[:, 1],
            mode='markers', name=rep,
            marker=dict(color=color, size=8),
            text=[rep] * len(pc_left)
        ))
    fig_left.update_layout(
        title=f"PCA (Variance: {pca_left.explained_variance_ratio_[0]:.1%} + {pca_left.explained_variance_ratio_[1]:.1%})",
        xaxis_title=f"PC1 ({pca_left.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca_left.explained_variance_ratio_[1]:.1%})",
        height=500
    )
    st.plotly_chart(fig_left, use_container_width=True)

# Right: With filtering
with col_right:
    st.markdown("**With Final Filtering**")
    df_right = df_pca.copy()
    
    # Apply filtering
    log10_full = np.log10(df_right[all_reps].replace(0, np.nan))
    if filter_strategy in ["Low intensity", "Combined"]:
        mask = (log10_full >= 0.5).all(axis=1)
        df_right = df_right[mask]
    if filter_strategy in ["±2σ filtered", "Combined"]:
        mask = pd.Series(True, index=df_right.index)
        log10_current = np.log10(df_right[all_reps].replace(0, np.nan))
        for rep in all_reps:
            vals = log10_current[rep].dropna()
            if len(vals) == 0: continue
            mean, std = vals.mean(), vals.std()
            mask &= (log10_current[rep] >= mean - 2*std) & (log10_current[rep] <= mean + 2*std)
        df_right = df_right[mask]
    
    if species_choice != "All proteins":
        df_right = df_right[df_right["Species"] == species_choice]
    
    X_right = df_right[all_reps].replace(0, np.nan).fillna(df_right[all_reps].median())
    X_right_scaled = StandardScaler().fit_transform(X_right)
    pca_right = PCA(n_components=2)
    pc_right = pca_right.fit_transform(X_right_scaled)
    
    fig_right = go.Figure()
    for i, rep in enumerate(all_reps):
        color = "#E71316" if rep in c1 else "#1f77b4"
        fig_right.add_trace(go.Scatter(
            x=pc_right[:, 0], y=pc_right[:, 1],
            mode='markers', name=rep,
            marker=dict(color=color, size=8),
            text=[rep] * len(pc_right)
        ))
    fig_right.update_layout(
        title=f"PCA (Variance: {pca_right.explained_variance_ratio_[0]:.1%} + {pca_right.explained_variance_ratio_[1]:.1%})",
        xaxis_title=f"PC1 ({pca_right.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca_right.explained_variance_ratio_[1]:.1%})",
        height=500
    )
    st.plotly_chart(fig_right, use_container_width=True)

# === 5. DENSITY PLOTS ===
st.subheader("Intensity Density Plots (log₁₀)")
# [Your existing 6 density plots code here — uses df_current from previous version]

# === 6. PROTEIN COUNT TABLE ===
st.subheader("Protein Counts After Filtering")
count_data = [{"Species": "Total", "Count": len(df_right)}]
if "Species" in df_right.columns:
    for sp in df_right["Species"].value_counts().index:
        count_data.append({"Species": sp, "Count": df_right["Species"].value_counts()[sp]})
st.table(pd.DataFrame(count_data))

# === 7. ACCEPT ===
if st.button("Accept & Proceed to Differential Analysis", type="primary"):
    st.session_state.intensity_transformed = df_right[all_reps]
    st.session_state.df_filtered = df_right
    st.session_state.transform_applied = best_transform if "Recommended" in transformation_choice else "Raw"
    st.session_state.qc_accepted = True
    st.success("Ready for differential analysis!")
    st.balloons()
