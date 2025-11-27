# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load data
if "prot_final_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2
all_reps = c1 + c2

st.title("Protein-Level QC & Advanced Filtering")

# === 1. PRE-CALCULATE LOG10 DATA ===
if "log10_cache" not in st.session_state:
    raw = df[all_reps].replace(0, np.nan)
    log10_all = np.log10(raw)

    cache = {"All proteins": log10_all}
    if "Species" in df.columns:
        for sp in ["HUMAN", "ECOLI", "YEAST"]:
            subset = df[df["Species"] == sp][all_reps].replace(0, np.nan)
            cache[sp] = np.log10(subset) if len(subset) > 0 else pd.DataFrame()
    
    st.session_state.log10_cache = cache

# === 2. LOW INTENSITY FILTER CHECKBOX ===
remove_low_intensity = st.checkbox(
    "Remove proteins with log₁₀ intensity < 0.5 in ALL replicates",
    value=False
)

# Apply low intensity filter
df_low_filtered = df.copy()
if remove_low_intensity:
    mask = pd.Series(True, index=df.index)
    
    for rep in all_reps:
        mask &= (np.log10(df[rep].replace(0, np.nan)) >= 0.5)
    df_low_filtered = df[mask]

# === 3. RADIO BUTTONS OUTSIDE PLOT AREA ===
st.subheader("Select Species to Display")
selected_species = st.radio(
    "Choose which data to show in all plots:",
    ["All proteins", "HUMAN", "ECOLI", "YEAST"],
    index=0,
    key="species_selector"
)

# === 4. 6 LOG10 DENSITY PLOTS ===
st.subheader("Intensity Density Plots (log₁₀)")

current_data = st.session_state.log10_cache[selected_species]

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i- [i-3]]
    with col:
        vals = current_data[rep].dropna()
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
            title=f"<b>{rep}</b><br>{selected_species}",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # === CLEAN TABLE UNDER EACH PLOT ===
        table_data = []
        for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
            sp_data = st.session_state.log10_cache[sp]
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

# === 5. FILTER TYPE SELECTION ===
st.subheader("Filter Type")
filter_type = st.radio(
    "Choose filtering strategy:",
    [
        "Raw data",
        "Low intensity filtered",
        "±2σ filtered (after low intensity)",
        "Combined (low intensity → ±2σ)"
    ],
    index=0
)

# === 6. DYNAMIC PROTEIN COUNT TABLE ===
st.subheader("Protein Counts")

# Base counts
counts = {"All proteins": len(df)}
if "Species" in df.columns:
    for sp in ["HUMAN", "ECOLI", "YEAST"]:
        counts[sp] = len(df[df["Species"] == sp])

# Compute filtered counts
df_for_count = df_low_filtered.copy()  # start from low-intensity filtered

if filter_type == "Raw data":
    final_count = len(df)
elif filter_type == "Low intensity filtered":
    final_count = len(df_low_filtered)
else:
    # Apply ±2σ on top of low intensity
    mask = pd.Series(True, index=df_low_filtered.index)
    log10_for_sigma = np.log10(df_low_filtered[all_reps].replace(0, np.nan))
    for rep in all_reps:
        vals = log10_for_sigma[rep].dropna()
        mean = vals.mean()
        std = vals.std()
        lower = mean - 2*std
        upper = mean + 2*std
        mask &= (log10_for_sigma[rep] >= lower) & (log10_for_sigma[rep] <= upper)
    final_count = len(df_low_filtered[mask])

# Build dynamic table
count_table = []
for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
    base = counts.get(sp, 0)
    if filter_type == "Raw data":
        filtered = base
    elif filter_type == "Low intensity filtered":
        if sp == "All proteins":
            filtered = len(df_low_filtered)
        else:
            :
            filtered = len(df_low_filtered[df_low_filtered["Species"] == sp]) if "Species" in df_low_filtered.columns else 0
    else:
        # ±2σ or combined
        subset = df_low_filtered[df_low_filtered["Species"] == sp] if sp != "All proteins" else df_low_filtered
        mask = pd.Series(True, index=subset.index)
        log10_sub = np.log10(subset[all_reps].replace(0, np.nan))
        for rep in all_reps:
            vals = log10_sub[rep].dropna()
            mean = vals.mean()
            std = vals.std()
            lower = mean - 2*std
            upper = mean + 2*std
            mask &= (log10_sub[rep] >= lower) & (log10_sub[rep] <= upper)
        filtered = len(subset[mask])
    
    count_table.append({
        "Species": sp,
        "Unfiltered": base,
        "After Filter": filtered
    })

st.table(pd.DataFrame(count_table).set_index("Species"))

# === 7. ACCEPT ===
st.markdown("### Confirm Setup")
if st.button("Accept This Filtering", type="primary"):
    # Store final filtered data based on selection
    if filter_type == "Raw data":
        final_df = df
    elif filter_type == "Low intensity filtered":
        final_df = df_low_filtered
    else:
        # ±2σ or combined
        mask = pd.Series(True, index=df_low_filtered.index)
        log10_for_sigma = np.log10(df_low_filtered[all_reps].replace(0, np.nan))
        for rep in all_reps:
            vals = log10_for_sigma[rep].dropna()
            mean = vals.mean()
            std = vals.std()
            lower = mean - 2*std
            upper = mean + 2*std
            mask &= (log10_for_sigma[rep] >= lower) & (log10_for_sigma[rep] <= upper)
        final_df = df_low_filtered[mask]
    
    st.session_state.df_filtered = final_df
    st.session_state.qc_accepted = True
    st.success("**Filtering accepted** — ready for next step")
