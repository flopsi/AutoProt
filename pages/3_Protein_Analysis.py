# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# pages/3_Protein_Analysis.py
import streamlit as st

# === GUARD: MUST HAVE PROTEIN DATA ===
if "prot_df" not in st.session_state or st.session_state.prot_df is None:
    st.error("No protein data found! Please go to **Protein Import** first.")
    if st.button("Go to Protein Import"):
        st.switch_page("pages/1_Protein_Import.py")
    st.stop()


# NOW SAFE TO USE
df = st.session_state.prot_df
c1 = st.session_state.prot_c1
c2 = st.session_state.prot_c2

st.success(f"Loaded {len(df):,} proteins | Condition A: {len(c1)} reps | Condition B: {len(c2)} reps")

st.error("No protein data found! Redirecting to Protein Import...")
import time
time.sleep(3)
st.switch_page("pages/1_Protein_Import.py")
st.title("Protein-Level QC & Replicate Difference Testing")

# === Schessner et al., 2022 ===")

# === 1. LOW INTENSITY FILTER FOR PLOTS ONLY ===
st.subheader("Plot Filter (Visual QC Only)")
remove_low_plot = st.checkbox(
    "Remove proteins with log₁₀ intensity < 0.5 in ALL replicates (plots only)",
    value=False
)

# Apply to plot data
df_plot = df.copy()
if remove_low_plot:
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        mask &= (np.log10(df[rep].replace(0, np.nan)) >= 0.5)
    df_plot = df[mask]

# === 2. PRE-CALCULATE LOG10 FOR PLOTS ===
if "log10_plot_cache" not in st.session_state or st.session_state.get("last_plot_filter") != remove_low_plot:
    raw = df_plot[all_reps].replace(0, np.nan)
    log10_all = np.log10(raw)

    cache = {"All proteins": log10_all}
    if "Species" in df_plot.columns:
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            subset = df_plot[df_plot["Species"] == sp][all_reps].replace(0, np.nan)
            cache[sp] = np.log10(subset) if len(subset) > 0 else pd.DataFrame()
    
    st.session_state.log10_plot_cache = cache
    st.session_state.last_plot_filter = remove_low_plot

# === 3. SPECIES SELECTION FOR PLOTS ===
#st.subheader("Select Species for Plots")
selected_species = st.radio(
    "Show in plots:",
    ["All proteins", "HUMAN", "ECOLI", "YEAST"],
    index=0,
    key="plot_species"
)

current_data = st.session_state.log10_plot_cache[selected_species]

# === 4. 6 LOG10 DENSITY PLOTS + TABLE UNDER EACH ===
#st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = current_data[rep].dropna()
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

        # === TABLE UNDER EACH PLOT ===
        table_data = []
        for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
            sp_data = st.session_state.log10_plot_cache[sp]
            sp_vals = sp_data[rep].dropna()
            if len(sp_vals) == 0:
                mean_str = variance_str = std_str = "—"
            else:
                mean_str = f"{sp_vals.mean():.3f}"
                variance_str = f"{sp_vals.var():.3f}"
                std_str = f"{sp_vals.std():.3f}"
            table_data.append({
                "Species": sp,
                "Mean": mean_str,
                "Variance": variance_str,
                "Std Dev": std_str
            })
        st.table(pd.DataFrame(table_data).set_index("Species"))

# === 5. REPLICATE DIFFERENCE TESTING ===
st.subheader("Replicate Difference Testing")

test_mode = st.radio(
    "Test replicate similarity using:",
    ["All proteins", "Constant proteome only"],
    index=1
)

if test_mode == "Constant proteome only":
    if "Species" not in df.columns:
        st.error("Species column missing")
        st.stop()
    constant_species = st.selectbox(
        "Select constant proteome (reference)",
        options=["HUMAN", "ECOLI", "YEAST"],
        index=0
    )
    reference_df = df[df["Species"] == constant_species]
else:
    constant_species = "All proteins"
    reference_df = df


# === 6. FINAL FILTER & ACCEPT ===
st.subheader("Final Filter Strategy")
filter_strategy = st.radio(
    "Choose filtering strategy:",
    ["Raw data", "Low intensity filtered", "±2σ filtered (on raw data)", "Combined"],
    index=0
)
# === REPLICATE DIFFERENCE TESTING (AFTER FINAL FILTERING) ===
st.subheader("Replicate Difference Testing (Kolmogorov–Smirnov)")

test_mode = st.radio(
    "Test replicate similarity using:",
    ["All proteins", "Constant proteome only"],
    index=1
)

if test_mode == "Constant proteome only":
    if "Species" not in df_final.columns:
        st.error("Species column missing in filtered data")
        st.stop()
    constant_species = st.selectbox(
        "Select constant proteome (reference)",
        options=["HUMAN", "ECOLI", "YEAST"],
        index=0
    )
    reference_df = df_final[df_final["Species"] == constant_species]
    ref_label = constant_species
else:
    reference_df = df_final
    ref_label = "All proteins"

# KS test: each replicate vs reference (after final filtering)
ks_results = []
for rep in all_reps:
    ref_vals = np.log10(reference_df[rep].replace(0, np.nan).dropna())
    rep_vals = np.log10(df_final[rep].replace(0, np.nan).dropna())
    
    if len(ref_vals) < 10 or len(rep_vals) < 10:
        ks_results.append({
            "Replicate": rep,
            "vs Reference": ref_label,
            "KS p-value": "—",
            "Different?": "—"
        })
        continue
    
    _, p = stats.ks_2samp(ref_vals, rep_vals)
    different = "Yes" if p < 0.05 else "No"
    ks_results.append({
        "Replicate": rep,
        "vs Reference": ref_label,
        "KS p-value": f"{p:.2e}",
        "Different?": different
    })

ks_df = pd.DataFrame(ks_results)

# Styled table
st.table(ks_df.style.apply(
    lambda x: ["background: #ffcccc" if v == "Yes" else "background: #ccffcc" for v in x],
    subset=["Different?"]
))

# Interpretation
if any(r["Different?"] == "Yes" for r in ks_results if r["Different?"] != "—"):
    st.error("**Significant differences detected** — potential technical bias")
else:
    st.success("**All replicates similar** — excellent technical reproducibility")

st.info("**Kolmogorov–Smirnov test** — compares full distribution shape (Schessner et al., 2022, Figure 4B)")
# [Your filtering & count table code here]

if st.button("Accept Final Filtering", type="primary"):
    st.session_state.df_filtered = df_final
    st.session_state.qc_accepted = True
    st.success("**Final filtering accepted** — ready for transformation")

# === DYNAMIC PROTEIN COUNT TABLE BASED ON FILTER STRATEGY ===
st.subheader("Protein Counts After Final Filter")

# Recalculate final filtered dataset
df_final = df.copy()
log10_full = np.log10(df[all_reps].replace(0, np.nan))

if filter_strategy in ["Low intensity filtered", "±2σ filtered (on raw data)", "Combined"]:
    # Low intensity filter
    mask = pd.Series(True, index=df.index)
    for rep in all_reps:
        mask &= (log10_full[rep] >= 0.5)
    df_final = df[mask]
    log10_full = log10_full.loc[mask]

if filter_strategy == "±2σ filtered (on raw data)":
    mask = pd.Series(True, index=df_final.index)
    for rep in all_reps:
        vals = log10_full[rep].dropna()
        if len(vals) == 0: continue
        mean = vals.mean()
        std = vals.std()
        mask &= (log10_full[rep] >= mean - 2*std) & (log10_full[rep] <= mean + 2*std)
    df_final = df_final[mask]

elif filter_strategy == "Combined":
    mask = pd.Series(True, index=df_final.index)
    for rep in all_reps:
        vals = log10_full[rep].dropna()
        if len(vals) == 0: continue
        mean = vals.mean()
        std = vals.std()
        mask &= (log10_full[rep] >= mean - 2*std) & (log10_full[rep] <= mean + 2*std)
    df_final = df_final[mask]

# Build count table
count_data = []
for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
    base = len(df[df["Species"] == sp]) if sp != "All proteins" and "Species" in df.columns else len(df)
    filtered = len(df_final[df_final["Species"] == sp]) if sp != "All proteins" and "Species" in df_final.columns else len(df_final)
    count_data.append({
        "Species": sp,
        "Unfiltered": base,
        "After Filter": filtered
    })

st.table(pd.DataFrame(count_data).set_index("Species"))
