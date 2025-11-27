# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson

# Load data
if "prot_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_df.copy()
c1 = st.session_state.prot_c1.copy()
c2 = st.session_state.prot_c2.copy()
all_reps = c1 + c2

st.title("Protein-Level QC & Transformation")

# === 1. FINAL FILTERING (Schessner et al., 2022) ===
st.subheader("1. Final Filtering Strategy")

filter_strategy = st.radio(
    "Apply filtering before visualization:",
    ["Raw data", "Low intensity filtered", "±2σ filtered", "Combined"],
    index=0,
    key="final_filter_strategy"
)

# Apply filtering
df_filtered = df.copy()
log10_full = np.log10(df_filtered[all_reps].replace(0, np.nan))

if filter_strategy in ["Low intensity filtered", "Combined"]:
    mask = pd.Series(True, index=df_filtered.index)
    for rep in all_reps:
        mask &= (log10_full[rep] >= 0.5)
    df_filtered = df_filtered[mask]

if filter_strategy in ["±2σ filtered", "Combined"]:
    mask = pd.Series(True, index=df_filtered.index)
    log10_filtered = np.log10(df_filtered[all_reps].replace(0, np.nan))
    for rep in all_reps:
        vals = log10_filtered[rep].dropna()
        if len(vals) == 0: continue
        mean = vals.mean()
        std = vals.std()
        mask &= (log10_filtered[rep] >= mean - 2*std) & (log10_filtered[rep] <= mean + 2*std)
    df_filtered = df_filtered[mask]

# === 2. NORMALITY TESTING ON RAW DATA ===
st.subheader("2. Normality Testing on Raw Data (Shapiro-Wilk)")

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
            
            if w > rep_w:
                rep_w = w
                rep_best = name
        except:
            row[f"{name} W"] = "—"
    
    row["Best Transform"] = rep_best
    row["Best W"] = f"{rep_w:.4f}"
    
    if rep_w > best_w:
        best_w = rep_w
        best_transform = rep_best
        
    results.append(row)

st.table(pd.DataFrame(results))
st.success(f"**Recommended transformation: {best_transform}** (highest W)")

# === 3. USER SELECTS: RAW OR TRANSFORMED + SPECIES ===
st.subheader("3. Data View")
col1, col2 = st.columns(2)
with col1:
    proceed_choice = st.radio(
        "Proceed with:",
        ["Raw data", f"Transformed data ({best_transform})"],
        index=1
    )
with col2:
    available_species = ["All proteins"]
    if "Species" in df_filtered.columns:
        available_species += df_filtered["Species"].dropna().unique().tolist()
    species_choice = st.radio(
        "Show species:",
        available_species,
        index=0
    )

# Apply transformation and species filter
current_data = df_filtered.copy()
if proceed_choice.startswith("Transformed"):
    func = transform_options[best_transform]
    current_data[all_reps] = current_data[all_reps].apply(func)

if species_choice != "All proteins":
    current_data = current_data[current_data["Species"] == species_choice]

# === 4. 6 DENSITY PLOTS (Schessner et al., 2022 Figure 4) ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = current_data[rep].replace(0, np.nan).dropna()
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

# === 5. PROTEIN COUNT TABLE ===
st.subheader("Protein Counts After Final Filtering")
count_data = []
count_data.append({
    "Species": "All proteins",
    "Before Filter": len(df),
    "After Filter": len(df_filtered)
})
if "Species" in df.columns:
    for sp in ["HUMAN", "ECOLI", "YEAST"]:
        if sp in df["Species"].values:
            before = len(df[df["Species"] == sp])
            after = len(df_filtered[df_filtered["Species"] == sp])
            count_data.append({
                "Species": sp,
                "Before Filter": before,
                "After Filter": after
            })

st.table(pd.DataFrame(count_data).set_index("Species").style.format("{:,}"))

# === 6. ACCEPT ===
if st.button("Accept & Proceed to Differential Analysis", type="primary"):
    if proceed_choice.startswith("Transformed"):
        func = transform_options[best_transform]
        transformed = df_filtered[all_reps].apply(func)
    else:
        transformed = df_filtered[all_reps]
    
    st.session_state.intensity_transformed = transformed
    st.session_state.df_filtered = df_filtered
    st.session_state.transform_applied = best_transform if "Transformed" in proceed_choice else "Raw"
    st.session_state.qc_accepted = True
    st.success("**Ready for differential analysis!**")
    st.balloons()
