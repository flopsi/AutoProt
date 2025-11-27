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
st.subheader("Visual Exploratory Data Anaylsis")
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
    st.success(f"Loaded **{len(df_plot):,}** proteins | Condition A: **{len(c1)}** reps | Condition B: **{len(c2)}** reps")

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
    "Plot:",
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

# === FINAL FILTER STRATEGY (DYNAMICALLY UPDATES COUNT TABLE & KS TEST) ===
st.subheader("Final Filter Strategy (for Downstream Analysis)")

filter_strategy = st.radio(
    "Choose filtering strategy:",
    ["Raw data", "Low intensity filtered", "±2σ filtered", "Combined"],
    index=0,
    key="final_filter_strategy_radio"
)

# === RECALCULATE FINAL DATASET BASED ON FILTER STRATEGY ===
df_final = df.copy()
log10_full = np.log10(df_final[all_reps].replace(0, np.nan))

if filter_strategy in ["Low intensity filtered", "±2σ filtered", "Combined"]:
    # Apply low-intensity filter first
    mask = pd.Series(True, index=df_final.index)
    for rep in all_reps:
        mask &= (log10_full[rep] >= 0.5)
    df_final = df_final[mask]
    log10_full = log10_full.loc[mask]

if filter_strategy in ["±2σ filtered", "Combined"]:
    # Apply ±2σ per replicate
    mask = pd.Series(True, index=df_final.index)
    for rep in all_reps:
        vals = log10_full[rep].dropna()
        if len(vals) == 0: continue
        mean = vals.mean()
        std = vals.std()
        mask &= (log10_full[rep] >= mean - 2*std) & (log10_full[rep] <= mean + 2*std)
    df_final = df_final[mask]

# === DYNAMIC PROTEIN COUNT TABLE (UPDATES INSTANTLY) ===
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
    count_data.append({
        "Species": sp,
        "Before Filter": unfiltered,
        "After Final Filter": filtered
    })

st.table(pd.DataFrame(count_data).set_index("Species"))

# === REPLICATE DIFFERENCE TESTING — WITHIN EACH CONDITION (Schessner et al., 2022 Figure 4B) ===
st.subheader("Replicate Difference Testing — Within Each Condition")

test_mode = st.radio(
    "Compare replicates using:",
    ["All proteins in condition", "Constant proteome only"],
    index=1,
    key="ks_within_condition_mode"
)

if test_mode == "Constant proteome only":
    if "Species" not in df_final.columns:
        st.error("Species column missing")
        st.stop()
    constant_species = st.selectbox(
        "Select constant proteome (reference)",
        options=["HUMAN", "ECOLI", "YEAST"],
        index=0,
        key="constant_species_ks"
    )
    ref_A = df_final[(df_final["Species"] == constant_species)]
    ref_B = ref_A.copy()
    ref_label = constant_species
else:
    ref_A = df_final[df_final.index.isin(df.index)]  # all in condition A
    ref_B = ref_A.copy()
    ref_label = "All proteins"

# KS test: pairwise within condition A and within condition B
ks_results = []

# Condition A replicates
for i, rep1 in enumerate(c1):
    for j in range(i+1, len(c1)):
        rep2 = c1[j]
        
        if test_mode == "Constant proteome only":
            ref_vals = np.log10(ref_A[rep1].replace(0, np.nan).dropna())
            test_vals = np.log10(ref_A[rep2].replace(0, np.nan).dropna())
        else:
            ref_vals = np.log10(df_final[rep1].replace(0, np.nan).dropna())
            test_vals = np.log10(df_final[rep2].replace(0, np.nan).dropna())
        
        if len(ref_vals) < 10 or len(test_vals) < 10:
            ks_results.append({"Condition": "A", "Rep1": rep1, "Rep2": rep2, "vs": ref_label, "p-value": "—", "Different?": "—"})
            continue
        
        _, p = stats.ks_2samp(ref_vals, test_vals)
        different = "Yes" if p < 0.05 else "No"
        ks_results.append({
            "Condition": "A",
            "Rep1": rep1,
            "Rep2": rep2,
            "vs": ref_label,
            "p-value": f"{p:.2e}",
            "Different?": different
        })

# Condition B replicates
for i, rep1 in enumerate(c2):
    for j in range(i+1, len(c2)):
        rep2 = c2[j]
        
        if test_mode == "Constant proteome only":
            ref_vals = np.log10(ref_B[rep1].replace(0, np.nan).dropna())
            test_vals = np.log10(ref_B[rep2].replace(0, np.nan).dropna())
        else:
            ref_vals = np.log10(df_final[rep1].replace(0, np.nan).dropna())
            test_vals = np.log10(df_final[rep2].replace(0, np.nan).dropna())
        
        if len(ref_vals) < 10 or len(test_vals) < 10:
            ks_results.append({"Condition": "B", "Rep1": rep1, "Rep2": rep2, "vs": ref_label, "p-value": "—", "Different?": "—"})
            continue
        
        _, p = stats.ks_2samp(ref_vals, test_vals)
        different = "Yes" if p < 0.05 else "No"
        ks_results.append({
            "Condition": "B",
            "Rep1": rep1,
            "Rep2": rep2,
            "vs": ref_label,
            "p-value": f"{p:.2e}",
            "Different?": different
        })

ks_df = pd.DataFrame(ks_results)
st.table(ks_df.style.apply(
    lambda x: ["background: #ffcccc" if v == "Yes" else "background: #ccffcc" for v in x],
    subset=["Different?"]
))

if any(r["Different?"] == "Yes" for r in ks_results if r["Different?"] != "—"):
    st.error("**Significant differences within condition** — check technical reproducibility")
else:
    st.success("**Excellent within-condition reproducibility** — all replicates similar")

st.info("**Kolmogorov–Smirnov test within each condition** — Schessner et al., 2022, Figure 4B")
if st.button("Accept Final Filtering & Proceed", type="primary"):
    st.session_state.df_filtered = df_final
    st.session_state.qc_accepted = True
    st.success("Final dataset accepted! Ready for normalization & stats.")
    st.balloons()


# === 8. NORMALITY TESTING & TRANSFORMATION RECOMMENDATION ===
st.subheader("Normality Testing & Transformation Recommendation")

# Test multiple transformations
transformations = {
    "Raw": lambda x: x,
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
    "Square root": np.sqrt,
    "Cube root": lambda x: np.cbrt(x + 1),
    "Box-Cox": lambda x: stats.boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: stats.yeojohnson(x + 1)[0],
    "Inverse": lambda x: 1 / (x + 1)
}

results = []
best_score = float('inf')
best_transform = "Raw"

for rep in all_reps:
    raw_vals = df_final[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8:
        continue
        
    row = {"Replicate": rep}
    rep_best = "Raw"
    rep_score = float('inf')
    
    for name, func in transformations.items():
        try:
            if name == "Box-Cox" and not (raw_vals > 0).all():
                continue
            t_vals = func(raw_vals)
            if t_vals is None or np.any(np.isnan(t_vals)) or np.any(np.isinf(t_vals)):
                continue
                
            skew = stats.skew(t_vals)
            kurt = stats.kurtosis(t_vals)
            _, p = stats.shapiro(t_vals)
            
            # Score: lower = better (Schessner et al. logic)
            score = abs(skew) + abs(kurt - 3) + (0 if p > 0.05 else 10)
            
            row[name + " skew"] = f"{skew:+.3f}"
            row[name + " kurt"] = f"{kurt:+.3f}"
            row[name + " p"] = f"{p:.2e}"
            
            if score < rep_score:
                rep_score = score
                rep_best = name
                
        except:
            continue
            
    row["Recommended"] = rep_best
    results.append(row)
    
    if rep_score < best_score:
        best_score = rep_score
        best_transform = rep_best

# Display results
results_df = pd.DataFrame(results)
st.table(results_df)

# Final recommendation
st.success(f"**Recommended transformation: {best_transform}**")
st.info(f"Based on minimizing skewness + excess kurtosis (Schessner et al., 2022)")

if best_transform in ["log₂", "log₁₀"]:
    st.info("Log transformation is the gold standard in proteomics — stabilizes variance and normalizes distributions")
elif best_transform in ["Box-Cox", "Yeo-Johnson"]:
    st.info("Power transformation optimal — handles non-constant variance")
else:
    st.warning("No strong improvement — consider experimental design")
