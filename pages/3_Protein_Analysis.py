# pages/3_Protein_Analysis.py
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

# ----------------- INPUT & BASIC SETUP -----------------
if "prot_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_df.copy()
c1 = st.session_state.prot_c1.copy()
c2 = st.session_state.prot_c2.copy()
all_reps = [c for c in (c1 + c2) if c in df.columns]

if not all_reps:
    st.error("No replicate intensity columns found in imported protein data.")
    st.stop()

st.title("Protein-Level QC")

# ----------------- 1. NORMALITY TESTING -----------------
st.subheader("1. Normality Testing & Recommended Transformation")

# Ensure strictly positive values for transforms that need it
eps = 1e-6

transform_options = {
    "log₂": lambda x: np.log2(x + eps),
    "log₁₀": lambda x: np.log10(x + eps),
    "Square root": lambda x: np.sqrt(x + eps),
    "Box-Cox": lambda x: boxcox(x + eps)[0] if (x + eps > 0).all() else None,
    "Yeo-Johnson": lambda x: yeojohnson(x + eps)[0],
}

results = []
best_transform = "log₁₀"
best_w_global = 0.0

for rep in all_reps:
    raw_vals = df[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8:
        continue

    row = {"Replicate": rep}
    w_raw, p_raw = stats.shapiro(raw_vals)
    row["Raw W"] = f"{w_raw:.4f}"

    rep_best = "log₁₀"
    rep_best_w = 0.0

    for name, func in transform_options.items():
        try:
            t_vals = func(raw_vals)
            if t_vals is None or np.any(~np.isfinite(t_vals)):
                continue
            w, _ = stats.shapiro(t_vals)
            row[f"{name} W"] = f"{w:.4f}"
            if w > rep_best_w:
                rep_best_w = w
                rep_best = name
        except Exception:
            row[f"{name} W"] = "—"

    row["Best"] = rep_best
    if rep_best_w > best_w_global:
        best_w_global = rep_best_w
        best_transform = rep_best

    results.append(row)

if results:
    st.table(pd.DataFrame(results))
    st.success(f"Recommended transformation: {best_transform}")
else:
    st.warning("Too few values per replicate for Shapiro–Wilk; using log₁₀ as default.")
    best_transform = "log₁₀"

# ----------------- 2. DATA VIEW & FILTERING PANEL -----------------
st.subheader("2. Data Processing & Visualization")

col_t, col_s, col_f = st.columns(3)

with col_t:
    transformation = st.radio(
        "Transformation",
        ["Recommended", "Raw"],
        index=0,
        key="trans_opt",
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
        index=0,
    )

# Apply transformation to a working copy
df_processed = df.copy()
if transformation == "Recommended":
    func = transform_options[best_transform]
    df_processed[all_reps] = df_processed[all_reps].apply(func)

# Apply view-level filtering (does not affect raw df)
if filtering != "No filtering":
    log10_full = np.log10(df_processed[all_reps].replace(0, np.nan) + eps)

    if filtering in ["Low intensity", "Combined"]:
        mask_low = (log10_full >= 0.5).all(axis=1)
        df_processed = df_processed[mask_low]

    if filtering in ["±2σ filtered", "Combined"]:
        mask_sigma = pd.Series(True, index=df_processed.index)
        log10_current = np.log10(df_processed[all_reps].replace(0, np.nan) + eps)
        for rep in all_reps:
            vals = log10_current[rep].dropna()
            if len(vals) == 0:
                continue
            mean, std = vals.mean(), vals.std()
            mask_sigma &= (log10_current[rep] >= mean - 2 * std) & (
                log10_current[rep] <= mean + 2 * std
            )
        df_processed = df_processed[mask_sigma]

# Species selection for visualization
df_visual = df_processed.copy()
if visual_species != "All proteins" and "Species" in df_visual.columns:
    df_visual = df_visual[df_visual["Species"] == visual_species]

# ----------------- 3. DENSITY PLOTS -----------------
st.subheader("Intensity Density Plots (log₁₀)")

row1 = st.columns(3)
row2 = st.columns(3)

for i, rep in enumerate(all_reps[:6]):  # assume up to 6 reps
    col = row1[i] if i < 3 else row2[i - 3]
    with col:
        vals = df_visual[rep].replace(0, np.nan).dropna()
        if len(vals) == 0:
            st.write("No data")
            continue

        vals_log10 = np.log10(vals + eps)
        mean = vals_log10.mean()
        std = vals_log10.std()
        lower = mean - 2 * std
        upper = mean + 2 * std

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=vals_log10,
                nbinsx=80,
                histnorm="density",
                name=rep,
                marker_color="#E71316" if rep in c1 else "#1f77b4",
                opacity=0.75,
            )
        )
        fig.add_vrect(
            x0=lower,
            x1=upper,
            fillcolor="white",
            opacity=0.35,
            line_width=2,
        )
        fig.add_vline(x=mean, line_dash="dash", line_color="black")

        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------- 4. PROTEIN COUNT TABLE -----------------
st.subheader("Protein Counts After Processing")

count_data = [{"Species": "All proteins", "Proteins": len(df_processed)}]
if "Species" in df_processed.columns:
    species_counts = df_processed["Species"].value_counts()
    for sp, cnt in species_counts.items():
        count_data.append({"Species": sp, "Proteins": cnt})

st.table(pd.DataFrame(count_data))

# ----------------- 5. PCA & REPLICATE SIMILARITY -----------------
st.subheader("PCA & Replicate Similarity (PERMANOVA)")

df_pca = df_processed[all_reps].copy()
df_pca = df_pca.dropna(how="any")

if len(df_pca) < 10 or df_pca.shape[1] < 2:
    st.error("Not enough proteins or replicates after filtering for reliable PCA.")
else:
    X = StandardScaler().fit_transform(df_pca.values)

    pca = PCA(n_components=2)
    # transpose: proteins x reps -> reps x PCs
    pc_scores = pca.fit_transform(X.T)

    fig = go.Figure()
    for i, rep in enumerate(all_reps):
        color = "#E71316" if rep in c1 else "#1f77b4"
        fig.add_trace(
            go.Scatter(
                x=[pc_scores[i, 0]],
                y=[pc_scores[i, 1]],
                mode="markers+text",
                name=rep,
                marker=dict(
                    color=color,
                    size=18,
                    line=dict(width=3, color="black"),
                ),
                text=rep,
                textposition="top center",
                textfont=dict(size=14, family="Arial", color="black"),
            )
        )

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
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # PERMANOVA-style summary (still assuming two groups A/B)
    if len(c1) > 0 and len(c2) > 0:
        dist_matrix = squareform(pdist(pc_scores, metric="euclidean"))
        groups = ["A"] * len(c1) + ["B"] * len(c2)

        n = len(groups)
        a = len(set(groups))
        N = n

        SST = np.sum(dist_matrix**2) / (2 * N)

        SSW = 0.0
        for group in set(groups):
            idx = [i for i, g in enumerate(groups) if g == group]
            if len(idx) > 1:
                sub = dist_matrix[np.ix_(idx, idx)]
                SSW += np.sum(sub**2) / (2 * len(idx))

        SSB = SST - SSW
        df_between = a - 1
        df_within = N - a
        MSB = SSB / df_between if df_between > 0 else 0.0
        MSW = SSW / df_within if df_within > 0 else 0.0
        F_stat = MSB / MSW if MSW > 0 else float("inf")
        p_value = 1 - f.cdf(F_stat, df_between, df_within) if MSW > 0 else 0.0

        st.markdown("#### PERMANOVA: Replicate Similarity")
        col1m, col2m, col3m = st.columns(3)
        with col1m:
            st.metric("F-statistic", f"{F_stat:.3f}")
        with col2m:
            st.metric("p-value", f"{p_value:.3e}")
        with col3m:
            if p_value < 0.05:
                st.error("Significant difference between groups")
            else:
                st.success("No significant difference — good reproducibility")
    else:
        st.info("PCA shown for replicates; PERMANOVA skipped (needs both A and B groups).")

# ----------------- 6. FINAL FILTERING & LOG2 EXPORT -----------------
st.subheader("Final Filtering & log₂ Transformation")

# Rebuild a clean copy of df for final filtering (using best transform)
best_transform_func = transform_options[best_transform]
df_for_filtering = df.copy()
df_for_filtering[all_reps] = df_for_filtering[all_reps].apply(best_transform_func)

mask_final = pd.Series(True, index=df_for_filtering.index)

if filtering in ["Low intensity", "Combined"]:
    log10_vals = np.log10(df_for_filtering[all_reps].replace(0, np.nan) + eps)
    mask_final &= (log10_vals >= 0.5).all(axis=1)

if filtering in ["±2σ filtered", "Combined"]:
    log10_cur = np.log10(df_for_filtering[all_reps].replace(0, np.nan) + eps)
    for rep in all_reps:
        vals = log10_cur[rep].dropna()
        if len(vals) == 0:
            continue
        mean, std = vals.mean(), vals.std()
        mask_final &= (log10_cur[rep] >= mean - 2 * std) & (
            log10_cur[rep] <= mean + 2 * std
        )

filtered_index = df_for_filtering[mask_final].index

# Save raw intensities after final filtering
st.session_state.raw_intensities_filtered = df.loc[filtered_index, all_reps].copy()

# Final data: log₂ of original raw values (for downstream biology)
df_final = df.loc[filtered_index].copy()
df_final[all_reps] = np.log2(df_final[all_reps].replace(0, np.nan) + eps)
df_final[all_reps] = df_final[all_reps].fillna(df_final[all_reps].min().min() - 1)

st.session_state.intensity_final = df_final[all_reps]
st.session_state.df_final = df_final

st.success(f"Final dataset: {len(df_final):,} proteins (log₂ transformed)")

st.subheader("Final Data Snapshot (5 rows)")
st.dataframe(df_final[all_reps].head(5).round(3), use_container_width=True)
