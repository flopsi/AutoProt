# pages/3_Peptide_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f

# Load peptide data
if "pep_df" not in st.session_state:
    st.error("No peptide data found! Please go to Peptide Import first.")
    if st.button("Go to Peptide Import"):
        st.switch_page("pages/1_Peptide_Import.py")
    st.stop()

df = st.session_state.pep_df.copy()
c1 = st.session_state.pep_c1.copy()
c2 = st.session_state.pep_c2.copy()
all_reps = c1 + c2

st.title("Peptide-Level QC (Schessner et al., 2022 Figure 4)")

# === 1. NORMALITY TESTING ON RAW DATA ===
st.subheader("1. Normality Testing on Raw Data (Shapiro-Wilk)")

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

# === 2. DATA VIEW & FILTERING PANEL ===
st.subheader("2. Data Processing & Visualization")

col_t, col_s, col_f = st.columns(3)

with col_t:
    transformation = st.radio("Transformation", ["Recommended", "Raw"], index=0)

with col_s:
    available_species = ["All peptides"]
    if "Species" in df.columns:
        available_species += sorted(df["Species"].dropna().unique().tolist())
    visual_species = st.radio("Visualize species", available_species, index=0)

with col_f:
    filtering = st.radio("Filtering", ["No filtering", "Low intensity", "±2σ filtered", "Combined"], index=0)

# === APPLY TRANSFORMATION & FILTERING ===
df_processed = df.copy()

# Transformation
if transformation == "Recommended":
    func = transform_options[best_transform]
    df_processed[all_reps] = df_processed[all_reps].apply(func)

# Filtering
if filtering != "No filtering":
    log10_full = np.log10(df_processed[all_reps].replace(0, np.nan))
    
    if filtering in ["Low intensity", "Combined"]:
        mask = (log10_full >= 0.5).all(axis=1)
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
if visual_species != "All peptides":
    df_visual = df_visual[df_visual["Species"] == visual_species]

# === 3. 6 DENSITY PLOTS ===
st.subheader("Peptide Intensity Density Plots (log₁₀)")

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

# === 4. PEPTIDE COUNT TABLE ===
st.subheader("Peptide Counts After Processing")
count_data = []
count_data.append({"Species": "All peptides", "Count": len(df_processed)})
if "Species" in df_processed.columns:
    for sp in df_processed["Species"].value_counts().index:
        count_data.append({"Species": sp, "Count": df_processed["Species"].value_counts()[sp]})
st.table(pd.DataFrame(count_data))

# === 5. PCA ON FINAL DATA ===
st.subheader("PCA of Replicate Profiles (Schessner et al., 2022 Figure 4)")

df_pca = df_processed[all_reps].copy()
df_pca = df_pca.dropna(how='any')

if len(df_pca) < 10:
    st.warning("Not enough peptides for reliable PCA")
else:
    X = StandardScaler().fit_transform(df_pca.values)
    pca = PCA(n_components=2)
    pc_scores = pca.fit_transform(X.T)

    fig = go.Figure()
    for i, rep in enumerate(all_reps):
        color = "#E71316" if rep in c1 else "#1f77b4"
        fig.add_trace(go.Scatter(
            x=[pc_scores[i, 0]],
            y=[pc_scores[i, 1]],
            mode='markers+text',
            name=rep,
            marker=dict(color=color, size=18, line=dict(width=3, color='black')),
            text=rep,
            textposition="top center",
            textfont=dict(size=14)
        ))

    fig.update_layout(
        title=f"PCA (PC1: {pca.explained_variance_ratio_[0]:.1%} • PC2: {pca.explained_variance_ratio_[1]:.1%})",
        xaxis_title="PC1", yaxis_title="PC2",
        height=600, showlegend=False, template="simple_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# === 6. PERMANOVA TEST ===
st.subheader("Replicate Similarity (PERMANOVA)")
dist = squareform(pdist(pc_scores))
groups = ['A'] * len(c1) + ['B'] * len(c2)
n = len(groups)
a = 2
SST = np.sum(dist**2) / (2*n)
SSW = 0
for g in set(groups):
    idx = [i for i, x in enumerate(groups) if x == g]
    if len(idx) > 1:
        SSW += np.sum(dist[np.ix_(idx,idx)]**2) / (2*len(idx))
SSB = SST - SSW
F_stat = (SSB/(a-1)) / (SSW/(n-a)) if (n-a) > 0 else float('inf')
p_val = 1 - f.cdf(F_stat, a-1, n-a)

col1, col2 = st.columns(2)
with col1:
    st.metric("PERMANOVA F", f"{F_stat:.3f}")
with col2:
    st.metric("p-value", f"{p_val:.2e}")

if p_val < 0.05:
    st.error("Significant difference between conditions")
else:
    st.success("No significant difference — excellent technical reproducibility")

# === 7. ACCEPT ===
if st.button("Accept & Proceed", type="primary"):
    st.session_state.pep_intensity_transformed = df_processed[all_reps]
    st.session_state.pep_df_filtered = df_processed
    st.session_state.pep_qc_accepted = True
    st.success("Peptide data ready!")
    st.balloons()
