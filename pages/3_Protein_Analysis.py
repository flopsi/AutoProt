# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats


if "last_plot_df_hash" not in st.session_state:
    st.session_state.last_plot_df_hash = None
# === SAFETY CHECK ===
required_keys = ["prot_df", "prot_c1", "prot_c2"]
missing = [k for k in required_keys if k not in st.session_state]
if missing or st.session_state.prot_df is None or len(st.session_state.prot_df) == 0:
    st.error("No protein data found! Please complete **Protein Import** first.")
    if st.button("Go to Protein Import"):
        st.switch_page("pages/1_Protein_Import.py")
    st.stop()

# === LOAD DATA ===
df = st.session_state.prot_df.copy()
c1 = st.session_state.prot_c1
c2 = st.session_state.prot_c2
all_reps = c1 + c2



# === 1. LOW INTENSITY FILTER FOR PLOTS ONLY (NOW 100% WORKING) ===
st.subheader("Plot Filter (Visual QC Only)")
remove_low_plot = st.checkbox(
    "Remove proteins with log₁₀ intensity < 0.5 in ALL replicates (plots only)",
    value=False
)

# Always start from original data
df_plot = df.copy()

if remove_low_plot:
    # Build mask with correct index
    mask = pd.Series(True, index=df_plot.index)
    for rep in all_reps:
        log_vals = np.log10(df_plot[rep].replace(0, np.nan))
        mask &= (log_vals >= 0.5)
    df_plot = df_plot.loc[mask]
    st.success(f"Loaded **{len(df_plot):,}** proteins | Condition A: **{len(c1)}** reps | Condition B: **{len(c2)}** reps")
else:
    st.info(f"No plot filter → showing all {len(df_plot):,} proteins")

# === 2. RECALCULATE LOG10 CACHE BASED ON CURRENT df_plot ===
# This is the key: cache must reflect the current df_plot!
if "log10_plot_cache" not in st.session_state or st.session_state.get("last_plot_df_hash") != hash(df_plot.to_string()):
    intensity_cols = all_reps
    log10_data = np.log10(df_plot[intensity_cols].replace(0, np.nan))

    cache = {"All proteins": log10_data}

    if "Species" in df_plot.columns:
        for species in df_plot["Species"].unique():
            if pd.notna(species) and species != "Other":
                mask_sp = df_plot["Species"] == species
                subset = df_plot[mask_sp][intensity_cols].replace(0, np.nan)
                if not subset.empty:
                    cache[species] = np.log10(subset)

    st.session_state.log10_plot_cache = cache
    st.session_state.last_plot_df_hash = hash(df_plot.to_string())  # prevent re-caching

selected_species = st.radio(
    "Show in plots:",
    options=["All proteins"] + (["HUMAN", "ECOLI", "YEAST"] if "Species" in df.columns else []),
    index=0,
    horizontal=True
)

current_log10 = st.session_state.log10_plot_cache.get(selected_species, st.session_state.log10_plot_cache["All proteins"])

# === 4. 6 LOG10 DENSITY PLOTS + TABLE UNDER EACH ===
st.subheader("Intensity Density Plots (log₁₀)")

rows = st.columns(3)
for i, rep in enumerate(all_reps):
    with rows[i % 3]:
        vals = current_log10[rep].dropna()
        if len(vals) == 0:
            st.write(f"**{rep}**")
            st.write("No data")
            continue

        mean_val = vals.mean()
        std_val = vals.std()
        lower = mean_val - 2 * std_val
        upper = mean_val + 2 * std_val

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
        fig.add_vline(x=mean_val, line_dash="dash", line_color="black")

        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False,
            margin=dict(t=50)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table under plot
        table_data = []
        for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
            if sp in st.session_state.log10_plot_cache:
                sp_vals = st.session_state.log10_plot_cache[sp][rep].dropna()
                if len(sp_vals) == 0:
                    mean_s = var_s = std_s = "—"
                else:
                    mean_s = f"{sp_vals.mean():.3f}"
                    var_s = f"{sp_vals.var():.3f}"
                    std_s = f"{sp_vals.std():.3f}"
                table_data.append({"Species": sp, "Mean": mean_s, "Var": var_s, "Std": std_s})
        st.table(pd.DataFrame(table_data).set_index("Species"))

# === FINAL FILTER STRATEGY (100% FIXED) ===
st.subheader("Final Filter Strategy")
filter_strategy = st.radio(
    "Choose filtering strategy for downstream analysis:",
    ["Raw data", "Low intensity filtered", "±2σ filtered", "Combined"],
    index=0
)

df_final = df.copy()
log10_full = np.log10(df_final[all_reps].replace(0, np.nan))

# Start with full mask
mask = pd.Series(True, index=df_final.index)

if filter_strategy in ["Low intensity filtered", "Combined"]:
    # Must have ≥0.5 in ALL replicates
    low_intensity_mask = (log10_full >= 0.5).all(axis=1)
    mask &= low_intensity_mask

if filter_strategy in ["±2σ filtered", "Combined"]:
    # ±2σ per replicate
    sigma_mask = pd.Series(True, index=df_final.index)
    for rep in all_reps:
        vals = log10_full[rep].dropna()
        if len(vals) == 0:
            continue
        m, s = vals.mean(), vals.std()
        rep_mask = (log10_full[rep] >= m - 2*s) & (log10_full[rep] <= m + 2*s)
        sigma_mask &= rep_mask
    mask &= sigma_mask

df_final = df_final.loc[mask]

# === 6. REPLICATE DIFFERENCE TESTING (KS TEST) ===
st.subheader("Replicate Difference Testing (Kolmogorov–Smirnov)")

test_mode = st.radio(
    "Test using:",
    ["All proteins", "Constant proteome only"],
    index=1
)

if test_mode == "Constant proteome only":
    if "Species" not in df_final.columns:
        st.error("Species column missing")
        st.stop()
    constant_species = st.selectbox("Reference proteome", ["HUMAN", "ECOLI", "YEAST"], index=0)
    ref_df = df_final[df_final["Species"] == constant_species]
    ref_label = constant_species
else:
    ref_df = df_final
    ref_label = "All proteins"

ks_results = []
for rep in all_reps:
    ref_vals = np.log10(ref_df[rep].replace(0, np.nan).dropna())
    rep_vals = np.log10(df_final[rep].replace(0, np.nan).dropna())

    if len(ref_vals) < 10 or len(rep_vals) < 10:
        ks_results.append({"Replicate": rep, "vs": ref_label, "p-value": "—", "Different?": "—"})
        continue

    _, p = stats.ks_2samp(ref_vals, rep_vals)
    different = "Yes" if p < 0.05 else "No"
    ks_results.append({"Replicate": rep, "vs": ref_label, "p-value": f"{p:.2e}", "Different?": different})

ks_df = pd.DataFrame(ks_results)
st.table(ks_df.style.apply(
    lambda x: ["background: #ffcccc" if v == "Yes" else "background: #ccffcc" for v in x],
    subset="Different?"
))

if any(r["Different?"] == "Yes" for r in ks_results if r["Different?"] != "—"):
    st.error("**Significant differences detected** — check technical reproducibility")
else:
    st.success("**Excellent technical reproducibility** — all replicates similar")

# === 7. PROTEIN COUNTS AFTER FILTER ===
st.subheader("Protein Counts After Final Filter")
count_data = []
species_list = ["All proteins", "HUMAN", "ECOLI", "YEAST"]
for sp in species_list:
    if sp == "All proteins":
        unfiltered = len(df)
        filtered = len(df_final)
    else:
        unfiltered = len(df[df["Species"] == sp]) if "Species" in df.columns else 0
        filtered = len(df_final[df_final["Species"] == sp]) if "Species" in df_final.columns else 0
    count_data.append({"Species": sp, "Before Filter": unfiltered, "After Filter": filtered})

st.table(pd.DataFrame(count_data).set_index("Species"))

# === ACCEPT FILTERING ===
if st.button("Accept Final Filtering & Proceed", type="primary"):
    st.session_state.df_filtered = df_final
    st.session_state.qc_accepted = True
    st.success("Final dataset accepted! Ready for normalization & stats.")
    st.balloons()
