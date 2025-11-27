# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from itertools import combinations

# pages/3_Protein_Analysis.py
import streamlit as st

# === GUARD: MUST HAVE PROTEIN DATA ===
if "prot_df" not in st.session_state or st.session_state.prot_df is None:
    st.error("No protein data found! Please go to **Protein Import** first.")
    if st.button("Go to Protein Import"):
        st.switch_page("pages/1_Protein_Import.py")
    st.stop()

# === 1. LOW INTENSITY FILTER FOR PLOTS ONLY ===
st.subheader("Plot Filter (Visual QC Only)")
remove_low_plot = st.checkbox(
    "Remove proteins with log₁₀ intensity < 0.5 in ALL replicates (plots only)",
    value=False
)

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
st.subheader("Select Species for Plots")
selected_species = st.radio(
    "Show in plots:",
    ["All proteins", "HUMAN", "ECOLI", "YEAST"],
    index=0,
    key="plot_species"
)

current_data = st.session_state.log10_plot_cache.get(selected_species, st.session_state.log10_plot_cache["All proteins"])

# === 4. 6 LOG10 DENSITY PLOTS + TABLE UNDER EACH ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i <  3 else row2[i-3]
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

        # Table under each plot
        table_data = []
        for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
            sp_data = st.session_state.log10_plot_cache.get(sp, pd.DataFrame())
            sp_vals = sp_data[rep].dropna() if not sp_data.empty else pd.Series()
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

# === 5. FINAL FILTER STRATEGY ===
st.subheader("Final Filter Strategy (for Downstream Analysis)")
filter_strategy = st.radio(
    "Choose filtering strategy:",
    ["Raw data", "Low intensity filtered", "±2σ filtered", "Combined"],
    index=0,
    key="final_filter_strategy_radio"
)

# Apply final filtering
df_final = df.copy()
log10_full = np.log10(df_final[all_reps].replace(0, np.nan))

if filter_strategy in ["Low intensity filtered", "Combined"]:
    mask = pd.Series(True, index=df_final.index)
    for rep in all_reps:
        mask &= (log10_full[rep] >= 0.5)
    df_final = df_final[mask]
    log10_full = log10_full.loc[mask]

if filter_strategy in ["±2σ filtered", "Combined"]:
    mask = pd.Series(True, index=df_final.index)
    for rep in all_reps:
        vals = log10_full[rep].dropna()
        if len(vals) == 0: continue
        mean = vals.mean()
        std = vals.std()
        mask &= (log10_full[rep] >= mean - 2*std) & (log10_full[rep] <= mean + 2*std)
    df_final = df_final[mask]

# === 6. DYNAMIC PROTEIN COUNT TABLE ===
st.subheader("Protein Counts After Final Filter")
count_data = []
for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
    base = len(df[df["Species"] == sp]) if sp != "All proteins" and "Species" in df.columns else len(df)
    filtered = len(df_final[df_final["Species"] == sp]) if sp != "All proteins" and "Species" in df_final.columns else len(df_final)
    count_data.append({"Species": sp, "Unfiltered": base, "After Filter": filtered})

st.table(pd.DataFrame(count_data).set_index("Species"))

# === 7. REPLICATE DIFFERENCE TESTING (WITHIN CONDITION) ===
st.subheader("Replicate Difference Testing — Within Each Condition")
test_mode = st.radio(
    "Test using:",
    ["All proteins", "Constant proteome only"],
    index=1,
    key="ks_within_condition_mode"
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
# Condition A
for i, rep1 in enumerate(c1):
    for j in range(i+1, len(c1)):
        rep2 = c1[j]
        ref_vals = np.log10(ref_df[rep1].replace(0, np.nan).dropna())
        test_vals = np.log10(df_final[rep2].replace(0, np.nan).dropna())
        if len(ref_vals) < 10 or len(test_vals) < 10:
            ks_results.append({"Condition": "A", "Rep1": rep1, "Rep2": rep2, "vs": ref_label, "p-value": "—", "Different?": "—"})
            continue
        _, p = stats.ks_2samp(ref_vals, test_vals)
        different = "Yes" if p < 0.05 else "No"
        ks_results.append({"Condition": "A", "Rep1": rep1, "Rep2": rep2, "vs": ref_label, "p-value": f"{p:.2e}", "Different?": different})

# Condition B
for i, rep1 in enumerate(c2):
    for j in range(i+1, len(c2)):
        rep2 = c2[j]
        ref_vals = np.log10(ref_df[rep1].replace(0, np.nan).dropna())
        test_vals = np.log10(df_final[rep2].replace(0, np.nan).dropna())
        if len(ref_vals) < 10 or len(test_vals) < 10:
            ks_results.append({"Condition": "B", "Rep1": rep1, "Rep2": rep2, "vs": ref_label, "p-value": "—", "Different?": "—"})
            continue
        _, p = stats.ks_2samp(ref_vals, test_vals)
        different = "Yes" if p < 0.05 else "No"
        ks_results.append({"Condition": "B", "Rep1": rep1, "Rep2": rep2, "vs": ref_label, "p-value": f"{p:.2e}", "Different?": different})

ks_df = pd.DataFrame(ks_results)
st.table(ks_df.style.apply(lambda x: ["background: #ffcccc" if v == "Yes" else "background: #ccffcc" for v in x], subset=["Different?"]))
if any(r["Different?"] == "Yes" for r in ks_results if r["Different?"] != "—"):
    st.error("**Significant differences within condition** — check technical reproducibility")
else:
    st.success("**Excellent within-condition reproducibility** — all replicates similar")

st.info("**Kolmogorov–Smirnov test within each condition** — Schessner et al., 2022, Figure 4B")

# === 8. ACCEPT ===
if st.button("Accept Final Filtering & Proceed", type="primary"):
    st.session_state.df_filtered = df_final
    st.session_state.qc_accepted = True
    st.success("**Final dataset accepted!** Ready for transformation & differential analysis.")
    st.balloons()
