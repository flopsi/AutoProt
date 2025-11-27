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

st.title("Protein-Level QC & Transformation")

# === 1. SPECIES SELECTION ===
if "Species" not in df.columns:
    st.error("Species column missing — please re-upload")
    st.stop()

species_counts = df["Species"].value_counts()
most_common = species_counts.index[0]

selected_species = st.selectbox(
    "Select species for QC",
    options=species_counts.index.tolist(),
    index=0,
    format_func=lambda x: f"{x} ({species_counts[x]:,} proteins)"
)

df_species = df[df["Species"] == selected_species].copy()
st.write(f"Using **{len(df_species):,}** {selected_species} proteins")

# === 2. 6 LOG10 DENSITY PLOTS ===
st.subheader("2. Intensity Density Plots (log₁₀)")

raw_data = df_species[all_reps].replace(0, np.nan)
log10_data = np.log10(raw_data)

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=log10_data[rep].dropna(),
            nbinsx=80,
            histnorm="density",
            name=rep,
            marker_color="#E71316" if rep in c1 else "#1f77b4",
            opacity=0.75
        ))
        median = log10_data[rep].median()
        fig.add_vline(x=median, line_dash="dash", line_color="black")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

# === 3. KS TEST ON RAW DATA (CONSTANT SPECIES) ===
st.subheader("3. Technical Reproducibility (KS Test — Raw Intensity)")

significant_pairs = []
for r1, r2 in combinations(all_reps, 2):
    d1, d2 = raw_data[r1].dropna(), raw_data[r2].dropna()
    if len(d1) > 1 and len(d2) > 1:
        _, p = stats.ks_2samp(d1, d2)
        if p < 0.05:
            significant_pairs.append(f"{r1} vs {r2}")

if significant_pairs:
    st.warning(f"**Significant differences** in {selected_species} distributions:\n" + " • ".join(significant_pairs))
else:
    st.success("**All replicates have similar distributions** — excellent technical quality")

# === 4. NORMALITY TEST ON RAW INTENSITY ===
st.subheader("4. Raw Data Distribution Diagnosis")

stats_raw = []
for rep in all_reps:
    vals = raw_data[rep].dropna()
    if len(vals) < 8:
        stats_raw.append({"Replicate": rep, "n": len(vals), "Skew": "N/A", "Kurtosis": "N/A", "Shapiro p": "N/A", "Normal": "N/A"})
        continue
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    _, p = stats.shapiro(vals)
    normal = "Yes" if p > 0.05 else "No"
    stats_raw.append({
        "Replicate": rep,
        "n": len(vals),
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{p:.2e}",
        "Normal": normal
    })

st.dataframe(pd.DataFrame(stats_raw), use_container_width=True)

# === 5. SUGGESTION BASED ON RAW DATA ===
non_normal = sum(1 for r in stats_raw if r.get("Normal") == "No")
high_skew = any(abs(float(r["Skew"].replace("N/A","0"))) > 1.0 for r in stats_raw if r["Skew"] != "N/A")
high_kurt = any(abs(float(r["Kurtosis"].replace("N/A","0"))) > 3.0 for r in stats_raw if r["Kurtosis"] != "N/A")

if non_normal == 0:
    st.success("Raw data appears normal — parametric tests possible (rare!)")
else:
    st.warning("**Strong non-normality detected** — transformation REQUIRED")
    if high_skew or high_kurt:
        st.info("→ **log₁₀ transformation strongly recommended**")

# === 6. TRANSFORMATION SELECTION ===
st.markdown("### 6. Apply Transformation")
transform = st.selectbox(
    "Choose transformation",
    ["log₂", "log₁₀", "Box-Cox", "Yeo-Johnson", "None"],
    index=1
)

# Apply transformation (safe)
if transform == "log₂":
    transformed = np.log2(raw_data)
elif transform == "log₁₀":
    transformed = np.log10(raw_data)
elif transform == "Box-Cox":
    shifted = raw_data + 1
    transformed = pd.DataFrame(stats.boxcox(shifted.values.flatten())[0].reshape(shifted.shape),
                               index=raw_data.index, columns=raw_data.columns)
elif transform == "Yeo-Johnson":
    transformed = pd.DataFrame(stats.yeojohnson(raw_data.values.flatten())[0].reshape(raw_data.shape),
                               index=raw_data.index, columns=raw_data.columns)
else:
    transformed = raw_data

# === 7. POST-TRANSFORMATION TABLE (LIVE UPDATE) ===
st.markdown("### 7. After Transformation")
post_stats = []
for rep in all_reps:
    vals = transformed[rep].dropna()
    if len(vals) < 8:
        post_stats.append({"Replicate": rep, "Skew": "N/A", "Kurtosis": "N/A", "Shapiro p": "N/A", "Normal": "N/A"})
        continue
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

# === 8. LIVE NORMALITY STATEMENT ===
post_non_normal = sum(1 for r in post_stats if r.get("Normal") == "No")
if post_non_normal == 0:
    st.success("**Perfect — data is normal after transformation**")
elif post_non_normal <= 2:
    st.info("**Good — mild non-normality remains**")
else:
    st.warning("**Still non-normal — try another transformation**")

# === 9. ACCEPT BUTTON ONLY ===
st.markdown("### 9. Confirm Transformation")
if st.button("Accept This Transformation", type="primary"):
    st.session_state.intensity_transformed = transformed
    st.session_state.transform_applied = transform
    st.session_state.qc_accepted = True
    st.success(f"**{transform} accepted** — ready for next plots")

# No "proceed" button — as requested
st.info("More plots coming soon...")
