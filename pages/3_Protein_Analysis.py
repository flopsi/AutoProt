# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
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

st.title("Protein-Level QC & Replicate Difference Testing")

# === 1. MANUAL LOG10 INTENSITY RANGE (ABOVE PLOTS) ===
st.subheader("1. Manual log₁₀ Intensity Range Selection")

raw_data_all = df[all_reps].replace(0, np.nan)
log10_all = np.log10(raw_data_all)

global_min = log10_all.min().min()
global_max = log10_all.max().max()

min_val, max_val = st.slider(
    "Select log₁₀ intensity range to keep",
    min_value=float(global_min),
    max_value=float(global_max),
    value=(1.5, float(global_max)),
    step=0.05,
    help="Removes low-intensity noise (common in proteomics)"
)

st.info(f"**Keeping proteins with log₁₀ intensity between {min_val:.2f} and {max_val:.2f}** in ALL replicates")

# === 2. SPECIES SELECTION & FILTERING ===
st.subheader("2. Select Data for Replicate Testing")

test_scope = st.radio(
    "Test replicate differences on:",
    ["All proteins (after intensity filter)", "Only the most frequent species (constant proteome)"],
    index=1
)

# Apply intensity filter first
mask = pd.Series(True, index=df.index)
for rep in all_reps:
    mask &= (log10_all[rep] >= min_val) & (log10_all[rep] <= max_val)
df_intensity_filtered = df[mask].copy()

if test_scope == "Only the most frequent species (constant proteome)":
    if "Species" not in df_intensity_filtered.columns:
        st.error("Species column missing")
        st.stop()
    most_common = df_intensity_filtered["Species"].value_counts().index[0]
    df_test = df_intensity_filtered[df_intensity_filtered["Species"] == most_common].copy()
    st.write(f"Using **{len(df_test):,}** {most_common} proteins (constant proteome)")
else:
    df_test = df_intensity_filtered.copy()
    st.write(f"Using **{len(df_test):,}** proteins (all after intensity filter)")

# === 3. 6 LOG10 DENSITY PLOTS WITH WHITE SHADOW ===
st.subheader("3. Intensity Density Plots (log₁₀) — Selected Range Highlighted")

log10_test = np.log10(df_test[all_reps].replace(0, np.nan))

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = log10_test[rep].dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=80,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        # WHITE SHADOW OVERLAY
        fig.add_vrect(x0=min_val, x1=max_val, fillcolor="white", opacity=0.4, line_width=2, line_color="black")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 4. REPLICATE DIFFERENCE TESTING (KS TEST) ===
st.subheader("4. Replicate Difference Testing (Kolmogorov-Smirnov)")

significant_pairs = []
for r1, r2 in combinations(all_reps, 2):
    d1 = df_test[r1].replace(0, np.nan).dropna()
    d2 = df_test[r2].replace(0, np.nan).dropna()
    if len(d1) > 1 and len(d2) > 1:
        _, p = stats.ks_2samp(d1, d2)
        if p < 0.05:
            significant_pairs.append(f"{r1} vs {r2}")

if significant_pairs:
    st.error(f"**Significant differences detected** (p < 0.05):\n" + " • ".join(significant_pairs))
    st.info("Consider normalization or batch correction")
else:
    st.success("**All replicates have statistically similar distributions** — excellent technical quality")

# === 5. NORMALITY & TRANSFORMATION ===
st.markdown("### 5. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂", "log₁₀", "Yeo-Johnson", "None"],
    index=1
)

if transform == "log₂":
    transformed = np.log2(df_test[all_reps].replace(0, np.nan))
elif transform == "log₁₀":
    transformed = np.log10(df_test[all_reps].replace(0, np.nan))
elif transform == "Yeo-Johnson":
    from scipy.stats import yeojohnson
    transformed = pd.DataFrame(yeojohnson(df_test[all_reps].values.flatten())[0].reshape(df_test[all_reps].shape),
                               index=df_test.index, columns=all_reps)
else:
    transformed = df_test[all_reps].replace(0, np.nan)

# === 6. POST-TRANSFORMATION TABLE ===
st.markdown("### 6. After Transformation")
post_stats = []
for rep in all_reps:
    vals = transformed[rep].dropna()
    if len(vals) < 8: continue
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    _, p = stats.shapiro(vals)
    normal = "Yes" if p > 0.05 else "No"
    post_stats.append({
        "Replicate": rep,
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{p:.2e}",
        "Normal": normal
    })
st.dataframe(pd.DataFrame(post_stats), use_container_width=True)

# === 7. ACCEPT ===
st.markdown("### 7. Confirm Setup")
if st.button("Accept This Filtering & Transformation", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_test
    st.session_state.qc_accepted = True
    st.success("**Accepted** — ready for analysis")

if st.session_state.get("qc_accepted", False):
    if st.button("Go to Differential Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/4_Differential_Analysis.py")
