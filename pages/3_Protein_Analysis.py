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

st.title("Protein-Level QC (Schessner et al., 2022 Figure 4)")

# === 1. NORMALITY TESTING & RECOMMENDED TRANSFORMATION ===
st.subheader("1. Normality Testing & Recommended Transformation")

transform_options = {
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
    "Square root": lambda x: np.sqrt(x + 1),
    "Box-Cox": lambda x: boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: yeojohnson(x + 1)[0],
}

results = []
best_transform = "log₁₀"
best_w = 0

for rep in all_reps:
    raw_vals = df[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8: continue
        
    row = {"Replicate": rep}
    w_raw, p_raw = stats.shapiro(raw_vals)
    row["Raw W"] = f"{w_raw:.4f}"
    
    rep_best = "log₁₀"
    rep_w = 0
    
    for name, func in transform_options.items():
        try:
            t_vals = func(raw_vals)
            if t_vals is None or np.any(np.isnan(t_vals)): continue
            w, _ = stats.shapiro(t_vals)
            row[f"{name} W"] = f"{w:.4f}"
            if w > rep_w:
                rep_w = w
                rep_best = name
        except:
            row[f"{name} W"] = "—"
    
    row["Best"] = rep_best
    if rep_w > best_w:
        best_w = rep_w
        best_transform = rep_best
        
    results.append(row)

st.table(pd.DataFrame(results))
st.success(f"**Recommended transformation: {best_transform}**")

# === 2. DATA PROCESSING PANEL ===
st.subheader("2. Data Processing & Visualization")

col_t, col_s, col_f = st.columns(3)

with col_t:
    transformation = st.radio(
        "Transformation",
        ["Recommended", "Raw"],
        index=0,
        key="trans_opt"
    )

with col_s:
    available_species = ["All proteins"]
    if "Species" in df.columns:
        available_species += sorted(df["Species"].dropna().unique().tolist())
    visual_species = st.radio("Visualize species", available_species, index=0)

with col_f:
    filtering = st.radio(
        "Filtering",
        ["Low intensity", "±2σ filtered", "Combined"],
        index=2  # default Combined
    )

# === APPLY TRANSFORMATION & FILTERING ===
df_processed = df.copy()

# Transformation
if transformation == "Recommended":
    func = transform_options[best_transform]
    df_processed[all_reps] = df_processed[all_reps].apply(func)

# Filtering
if filtering in ["Low intensity", "Combined"]:
    mask = (np.log10(df_processed[all_reps].replace(0, np.nan)) >= 0.5).all(axis=1)
    df_processed = df_processed[mask]

if filtering in ["±2σ filtered", "Combined"]:
    mask = pd.Series(True, index=df_processed.index)
    log10_current = np.log10(df_processed[all_reps].replace(0, np.nan))
    for rep in all_reps:
        vals = log10_current[rep].dropna()
        if len(vals) == 0: continue
        mean, std = vals.mean(), vals.std()
        mask &= (log10_current[rep] >= mean - 2*std) & (log10_current[rep] <= mean + 2*std)
    df_processed = df_processed[mask]

# Visual species filter
df_visual = df_processed.copy()
if visual_species != "All proteins":
    df_visual = df_visual[df_visual["Species"] == visual_species]

# === 3. DENSITY PLOTS ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = df_visual[rep].replace(0, np.nan).dropna()
        if len(vals) == 0:
            st.write("No data")
            continue
            
        mean = vals.mean()
        std = vals.std()
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

# === 4. PROTEIN COUNT TABLE ===
st.subheader("Protein Counts After Processing")
count_data = []
count_data.append({"Species": "All proteins", "Proteins": len(df_processed)})
if "Species" in df_processed.columns:
    for sp in df_processed["Species"].value_counts().index:
        count_data.append({"Species": sp, "Proteins": df_processed["Species"].value_counts()[sp]})
st.table(pd.DataFrame(count_data))

# === 5. PCA ON FINAL PROCESSED DATA (Schessner et al., 2022 Figure 4) ===
st.subheader("PCA on Final Processed Data (Mean Intensity per Replicate)")

# Compute mean intensity per replicate
mean_intensities = []
rep_labels = []
rep_colors = []

for rep in all_reps:
    # Use the final processed data (after transformation + filtering)
    vals = df_processed[rep].replace(0, np.nan).dropna()
    if len(vals) == 0:
        continue
    mean_intensities.append(vals.mean())
    rep_labels.append(rep)
    rep_colors.append("#E71316" if rep in c1 else "#1f77b4")

# Convert to array: shape (n_replicates, 1)
X = np.array(mean_intensities).reshape(-1, 1)

if len(X) < 2:
    st.write("Not enough replicates for PCA")
else:
    # Standardize and perform PCA
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X_scaled)

# === 5. PCA ON FINAL PROCESSED DATA (Schessner et al., 2022 Figure 4) ===
st.subheader("PCA on Final Processed Data (Mean Intensity per Replicate)")

# Compute mean intensity per replicate
mean_intensities = []
rep_labels = []
rep_colors = []

for rep in all_reps:
    vals = df_processed[rep].replace(0, np.nan).dropna()
    if len(vals) == 0:
        continue
    mean_intensities.append(vals.mean())
    rep_labels.append(rep)
    rep_colors.append("#E71316" if rep in c1 else "#1f77b4")

if len(mean_intensities) < 2:
    st.write("Not enough replicates for PCA")
else:
    # Create dummy features to allow PCA with 2 components (trick for 1D data)
    X = np.array(mean_intensities).reshape(-1, 1)
    X_extended = np.hstack([X, np.zeros_like(X)])  # Add dummy column

    # PCA with 2 components
    pca = PCA(n_components=2)
    pc = pca.fit_transform(StandardScaler().fit_transform(X_extended))

    fig = go.Figure()

    for i, label in enumerate(rep_labels):
        fig.add_trace(go.Scatter(
            x=[pc[i, 0]],
            y=[pc[i, 1]],
            mode='markers+text',
            name=label,
            marker=dict(color=rep_colors[i], size=16, line=dict(width=2, color='black')),
            text=label,
            textposition="top center",
            textfont=dict(size=14, color="black")
        ))

    fig.update_layout(
        title="PCA of Replicate Mean Intensities<br>"
              f"PC1: {pca.explained_variance_ratio_[0]:.1%} | PC2: {pca.explained_variance_ratio_[1]:.1%}",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        height=600,
        showlegend=False,
        template="simple_white",
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    st.plotly_chart(fig, use_container_width=True)

# === 6. ACCEPT ===
if st.button("Accept & Proceed to Differential Analysis", type="primary"):
    st.session_state.intensity_transformed = df_processed[all_reps]
    st.session_state.df_filtered = df_processed
    st.session_state.transform_applied = best_transform
    st.session_state.qc_accepted = True
    st.success("Ready for differential analysis!")
    st.balloons()
