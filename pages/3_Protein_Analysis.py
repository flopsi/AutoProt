# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import hashlib

# Load data
if "prot_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_df.copy()
c1 = st.session_state.prot_c1.copy()
c2 = st.session_state.prot_c2.copy()
all_reps = c1 + c2

st.title("Protein-Level QC & Transformation")

# === 1. NORMALITY TESTING ON RAW DATA — SHAPIRO-WILK W STATISTIC (Schessner et al., 2022) ===
st.subheader("1. Normality Testing on Raw Data (Shapiro-Wilk)")

transform_options = {
    "Raw": lambda x: x,
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
    "Square root": lambda x: np.sqrt(x + 1),
    "Box-Cox": lambda x: boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: yeojohnson(x + 1)[0],
}

results = []
best_transform = "Raw"
best_w = 0

for rep in all_reps:
    raw_vals = df[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8:
        continue
        
    row = {"Replicate": rep}
    
    # Raw W statistic
    w_raw, p_raw = stats.shapiro(raw_vals)
    row["Raw W"] = f"{w_raw:.4f}"
    row["Raw p"] = f"{p_raw:.2e}"
    
    rep_best = "Raw"
    rep_w = w_raw
    
    for name, func in transform_options.items():
        if name == "Raw": continue
        try:
            t_vals = func(raw_vals)
            if t_vals is None or np.any(np.isnan(t_vals)): continue
                
            w, p = stats.shapiro(t_vals)
            row[f"{name} W"] = f"{w:.4f}"
            row[f"{name} p"] = f"{p:.2e}"
            
            if w > rep_w:
                rep_w = w
                rep_best = name
        except:
            row[f"{name} W"] = "—"
            row[f"{name} p"] = "—"
    
    row["Best Transform"] = rep_best
    row["Best W"] = f"{rep_w:.4f}"
    
    if rep_w > best_w:
        best_w = rep_w
        best_transform = rep_best
        
    results.append(row)

st.table(pd.DataFrame(results))

st.success(f"**Recommended transformation: {best_transform}** (highest Shapiro-Wilk W)")
st.info("**Shapiro-Wilk W statistic** — closer to 1 = more normal (Schessner et al., 2022, Figure 4)")

# === 2. USER DECIDES: RAW OR TRANSFORMED ===
st.subheader("2. Proceed With")
proceed_choice = st.radio(
    "Proceed to downstream analysis with:",
    ["Raw data", f"Transformed data ({best_transform})"],
    index=1
)
# === 1. LOW INTENSITY FILTER FOR PLOTS ONLY ===
st.subheader("Plot Filter (Visual QC Only)")
remove_low_plot = st.checkbox(
    "Remove proteins with log₁₀ intensity < 0.5 in ALL replicates (plots only)",
    value=False,
    key="remove_low_plot"
)

# Apply to plot data
df_plot = df.copy()
if remove_low_plot:
    try:
        mask = pd.Series(True, index=df.index)
        for rep in all_reps:
            if rep in df.columns:
                mask &= (np.log10(df[rep].replace(0, np.nan)) >= 0.5)
        df_plot = df.loc[mask]
    except:
        st.warning("Low-intensity filter failed — using full data for plots")

# === 2. PRE-CALCULATE LOG10 FOR PLOTS (ROBUST CACHING) ===
plot_hash = hashlib.md5(pd.util.hash_pandas_object(df_plot[all_reps]).values).hexdigest()
if ("log10_plot_cache" not in st.session_state or 
    "last_plot_hash" not in st.session_state or 
    st.session_state.last_plot_hash != plot_hash):
    
    raw = df_plot[all_reps].replace(0, np.nan)
    log10_all = np.log10(raw)

    cache = {"All proteins": log10_all}
    if "Species" in df_plot.columns:
        for sp in df_plot["Species"].dropna().unique():
            if pd.notna(sp):
                subset = df_plot[df_plot["Species"] == sp][all_reps].replace(0, np.nan)
                if len(subset) > 0:
                    cache[str(sp)] = np.log10(subset)
    
    st.session_state.log10_plot_cache = cache
    st.session_state.last_plot_hash = plot_hash

# === 3. SPECIES SELECTION FOR PLOTS ===
st.subheader("Select Species for Plots")
available_species = ["All proteins"] + [k for k in st.session_state.log10_plot_cache.keys() if k != "All proteins"]
selected_species = st.radio(
    "Show in plots:",
    options=available_species,
    index=0,
    key="plot_species_radio"
)

current_data = st.session_state.log10_plot_cache.get(selected_species, st.session_state.log10_plot_cache["All proteins"])

# === 4. 6 LOG10 DENSITY PLOTS + TABLE UNDER EACH ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        if rep not in current_data.columns:
            st.write("No data")
            continue
            
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
        for sp in available_species:
            sp_data = st.session_state.log10_plot_cache.get(sp, pd.DataFrame())
            if rep in sp_data.columns:
                sp_vals = sp_data[rep].dropna()
                if len(sp_vals) == 0:
                    mean_str = variance_str = std_str = "—"
                else:
                    mean_str = f"{sp_vals.mean():.3f}"
                    variance_str = f"{sp_vals.var():.3f}"
                    std_str = f"{sp_vals.std():.3f}"
            else:
                mean_str = variance_str = std_str = "—"
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
    key="final_filter_strategy"
)

# Apply final filtering
df_final = df.copy()
log10_full = np.log10(df_final[all_reps].replace(0, np.nan))

try:
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
except:
    st.warning("Filtering failed — using raw data")

# === 6. DYNAMIC PROTEIN COUNT TABLE ===
st.subheader("Protein Counts After Final Filter")
count_data = []
for sp in available_species:
    if sp == "All proteins":
        unfiltered = len(df)
        filtered = len(df_final)
    else:
        unfiltered = len(df[df["Species"] == sp]) if "Species" in df.columns else 0
        filtered = len(df_final[df_final["Species"] == sp]) if "Species" in df_final.columns else 0
    count_data.append({"Species": sp, "Unfiltered": unfiltered, "After Filter": filtered})

st.table(pd.DataFrame(count_data).set_index("Species"))

# === 7. REPLICATE DIFFERENCE TESTING ===
st.subheader("Replicate Difference Testing (Kolmogorov–Smirnov)")

test_mode = st.radio(
    "Test using:",
    ["All proteins", "Constant proteome only"],
    index=1,
    key="ks_mode"
)

if test_mode == "Constant proteome only":
    if "Species" not in df_final.columns:
        st.error("Species column missing")
        st.stop()
    constant_species = st.selectbox("Reference proteome", ["HUMAN", "ECOLI", "YEAST"], key="const_sp")
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
st.table(ks_df.style.apply(lambda x: ["background: #ffcccc" if v == "Yes" else "background: #ccffcc" for v in x], subset=["Different?"]))

if any(r["Different?"] == "Yes" for r in ks_results if r["Different?"] != "—"):
    st.error("**Significant differences** — check technical bias")
else:
    st.success("**Excellent reproducibility**")

st.info("**Kolmogorov–Smirnov test** — Schessner et al., 2022, Figure 4B")

# === 4. PROCEED BUTTON AT BOTTOM ===
st.markdown("### Confirm & Proceed")
if st.button("Accept & Proceed to Differential Analysis", type="primary"):
    # Apply final filtering first
    # [Your df_final calculation here]
    
    # Then apply transformation if chosen
    if proceed_choice == f"Transformed data ({best_transform})" and best_transform != "Raw":
        func = transform_options[best_transform]
        transformed = df_final[all_reps].apply(func)
    else:
        transformed = df_final[all_reps]
    
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_final
    st.session_state.transform_applied = best_transform if "Transformed" in proceed_choice else "Raw"
    st.session_state.qc_accepted = True
    st.success("**Ready for differential analysis!**")
    st.balloons()
