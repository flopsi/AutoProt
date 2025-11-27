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

st.title("Protein-Level QC")

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

# === 2. DATA VIEW & FILTERING PANEL ===
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
        ["No filtering", "Low intensity", "±2σ filtered", "Combined"],
        index=0  # default: No filtering (as requested)
    )

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
if visual_species != "All proteins":
    df_visual = df_visual[df_visual["Species"] == visual_species]

# === 3. 6 DENSITY PLOTS (Schessner et al., 2022 Figure 4) ===
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

# === 5. PCA & REPLICATE SIMILARITY (Schessner et al., 2022 Figure 4) ===
st.subheader("PCA & Replicate Similarity (PERMANOVA)")

# Use final processed data
df_pca = df_processed[all_reps].copy()
df_pca = df_pca.dropna(how='any')  # Remove proteins with any missing value

if len(df_pca) < 10:
    st.error("Not enough proteins after filtering for reliable PCA")
else:
    # Standardize across proteins
    X = StandardScaler().fit_transform(df_pca.values)
    
    # PCA: replicates as samples
    pca = PCA(n_components=2)
    pc_scores = pca.fit_transform(X.T)  # shape: (6, 2)

    # Create beautiful, integrated PCA plot
    fig = go.Figure()

    for i, rep in enumerate(all_reps):
        color = "#E71316" if rep in c1 else "#1f77b4"
        fig.add_trace(go.Scatter(
            x=[pc_scores[i, 0]],
            y=[pc_scores[i, 1]],
            mode='markers+text',
            name=rep,
            marker=dict(
                color=color,
                size=18,
                line=dict(width=3, color='black')
            ),
            text=rep,
            textposition="top center",
            textfont=dict(size=14, family="Arial", color="black")
        ))

    fig.update_layout(
        title="PCA of Replicate Profiles",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        height=500,
        showlegend=False,
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='gray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='gray')
    )

    st.plotly_chart(fig, use_container_width=True)

    # === PERMANOVA: Within vs Between Group Variance ===
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import f

    # Distance matrix (Euclidean on PC scores)
    dist_matrix = squareform(pdist(pc_scores, metric='euclidean'))

    # Group labels
    groups = ['A'] * len(c1) + ['B'] * len(c2)

    # PERMANOVA calculation
    n = len(groups)
    a = len(set(groups))  # number of groups
    N = n

    # Total sum of squares
    SST = np.sum(dist_matrix**2) / (2 * N)

    # Within-group sum of squares
    SSW = 0
    for group in set(groups):
        idx = [i for i, g in enumerate(groups) if g == group]
        if len(idx) > 1:
            submatrix = dist_matrix[np.ix_(idx, idx)]
            SSW += np.sum(submatrix**2) / (2 * len(idx))

    # Between-group sum of squares
    SSB = SST - SSW

    # Degrees of freedom
    df_between = a - 1
    df_within = N - a

    # Mean squares
    MSB = SSB / df_between if df_between > 0 else 0
    MSW = SSW / df_within if df_within > 0 else 0

    # F-statistic
    F = MSB / MSW if MSW > 0 else float('inf')
    p_value = 1 - f.cdf(F, df_between, df_within) if MSW > 0 else 0

    # Display result
    st.markdown("#### **PERMANOVA: Replicate Similarity**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("F-statistic", f"{F:.3f}")
    with col2:
        st.metric("p-value", f"{p_value:.3e}")
    with col3:
        if p_value < 0.05:
            st.error("**Significant difference** between groups")
        else:
            st.success("**No significant difference** — excellent reproducibility")

    st.info("**PERMANOVA** tests if variance between biological groups is greater than within-group technical variance (Schessner et al., 2022)")

# === FINAL FILTERING & LOG2 TRANSFORMATION (Schessner et al., 2022) ===
st.subheader("Final Filtering & log₂ Transformation")

# Use best transformation for filtering
best_transform_func = transform_options[best_transform]
df_for_filtering = df.copy()
df_for_filtering[all_reps] = df_for_filtering[all_reps].apply(best_transform_func)

# Apply filtering
mask = pd.Series(True, index=df_for_filtering.index)

if filtering in ["Low intensity", "Combined"]:
    log10_vals = np.log10(df_for_filtering[all_reps].replace(0, np.nan))
    mask &= (log10_vals >= 0.5).all(axis=1)

if filtering in ["±2σ filtered", "Combined"]:
    log10_current = np.log10(df_for_filtering[all_reps].replace(0, np.nan))
    for rep in all_reps:
        vals = log10_current[rep].dropna()
        if len(vals) == 0: continue
        mean, std = vals.mean(), vals.std()
        mask &= (log10_current[rep] >= mean - 2*std) & (log10_current[rep] <= mean + 2*std)

filtered_index = df_for_filtering[mask].index

# After final filtering
st.session_state.raw_intensities_filtered = df.loc[filtered_index, all_reps].copy()

# Final data: log2 of original raw values (GOLD STANDARD)
df_final = df.loc[filtered_index].copy()
df_final[all_reps] = np.log2(df_final[all_reps].replace(0, np.nan))
df_final[all_reps] = df_final[all_reps].fillna(df_final[all_reps].min() - 1)

# Save to session
st.session_state.intensity_final = df_final[all_reps]
st.session_state.df_final = df_final

st.success(f"Final dataset: {len(df_final):,} proteins (log₂ transformed)")
st.info("**Best filtering** using optimal transformation → **log₂** for biology (Schessner et al., 2022 + community standard)")

# === 5-ROW SNAPSHOT (for differential analysis) ===
st.subheader("Final Data Snapshot (5 rows)")
st.dataframe(df_final[all_reps].head(5).round(3), use_container_width=True)
