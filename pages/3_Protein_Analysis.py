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

# === 1. DROPDOWN: ALL / HUMAN / ECOLI / YEAST ===
st.subheader("1. Select View Mode")
view_species = st.selectbox(
    "Show density plots for:",
    ["All proteins", "HUMAN", "ECOLI", "YEAST"],
    index=0
)

# Filter data for plotting
if view_species == "All proteins":
    df_plot = df.copy()
else:
    if "Species" not in df.columns:
        st.error("Species column missing")
        st.stop()
    df_plot = df[df["Species"] == view_species].copy()

# === 2. 6 LOG10 DENSITY PLOTS + TABLE BELOW EACH ===
st.subheader("2. Intensity Density Plots (log₁₀)")

raw_plot = df_plot[all_reps].replace(0, np.nan)
log10_plot = np.log10(raw_plot)

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = log10_plot[rep].dropna()
        mean = vals.mean()
        std = vals.std()
        lower = mean - 2*std
        upper = mean + 2*std

        # Plot
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

        # === 4×3 TABLE BELOW EACH PLOT ===
        table_data = []
        for sp in ["All proteins", "HUMAN", "ECOLI", "YEAST"]:
            if sp == "All proteins":
                subset = df
            else:
                subset = df[df["Species"] == sp] if "Species" in df.columns else pd.DataFrame()
            
            if len(subset) == 0:
                table_data.append({"Species": sp, "Mean": "—", "Variance": "—", "Std Dev": "—"})
                continue
            
            rep_vals = np.log10(subset[rep].replace(0, np.nan).dropna())
            if len(rep_vals) == 0:
                table_data.append({"Species": sp, "Mean": "—", "Variance": "—", "Std Dev": "—"})
            else:
                table_data.append({
                    "Species": sp,
                    "Mean": f"{rep_vals.mean():.3f}",
                    "Variance": f"{rep_vals.var():.3f}",
                    "Std Dev": f"{rep_vals.std():.3f}"
                })
        
        st.table(pd.DataFrame(table_data).set_index("Species"))

# === 3. FILTERING OPTIONS — TWO COLUMNS ===
st.subheader("3. Filtering Options (Choose One)")

col_left, col_right = st.columns(2)

intensity_filter = False
stdev_filter = False

with col_left:
    st.markdown("**Intensity-based Filtering**")
    intensity_filter = st.checkbox("Enable intensity filter", key="intensity_cb")
    if intensity_filter:
        lower_input = st.number_input("Lower bound (log₁₀)", value=-1.0, step=0.1, format="%.2f")
        upper_input = st.number_input("Upper bound (log₁₀)", value=5.0, step=0.1, format="%.2f")

with col_right:
    st.markdown("**StDev-based Filtering**")
    stdev_filter = st.checkbox("Enable ±σ filter", key="stdev_cb")
    if stdev_filter:
        stdev_scope = st.radio("Apply ±σ:", ["Global", "Per species"], index=1)

# Enforce mutual exclusion
if intensity_filter and stdev_filter:
    st.error("Please select only ONE filtering method")
    st.stop()

# === 4. APPLY FILTERING ===
mask = pd.Series(False, index=df.index)

if intensity_filter:
    for rep in all_reps:
        rep_mask = (np.log10(df[rep].replace(0, np.nan)) >= lower_input) & \
                   (np.log10(df[rep].replace(0, np.nan)) <= upper_input)
        mask |= rep_mask

elif stdev_filter:
    if stdev_scope == "Global":
        all_vals = np.log10(df[all_reps].replace(0, np.nan)).stack()
        mean = all_vals.mean()
        std = all_vals.std()
        lower, upper = mean - 2*std, mean + 2*std
        for rep in all_reps:
            rep_mask = (np.log10(df[rep].replace(0, np.nan)) >= lower) & \
                       (np.log10(df[rep].replace(0, np.nan)) <= upper)
            mask |= rep_mask
    else:
        species_sigma = {}
        for sp in df["Species"].dropna().unique():
            sp_data = df[df["Species"] == sp][all_reps].replace(0, np.nan)
            sp_log = np.log10(sp_data)
            mean = sp_log.mean().mean()
            std = sp_log.stack().std()
            species_sigma[sp] = (mean - 2*std, mean + 2*std)
        
        selected_species_list = st.multiselect(
            "Select species to include (with their ±2σ)",
            options=list(species_sigma.keys()),
            default=list(species_sigma.keys())[:1]
        )
        for sp in selected_species_list:
            lower, upper = species_sigma[sp]
            sp_mask = df["Species"] == sp
            for rep in all_reps:
                rep_mask = sp_mask & (np.log10(df[rep].replace(0, np.nan)) >= lower) & \
                                      (np.log10(df[rep].replace(0, np.nan)) <= upper)
                mask |= rep_mask

else:
    mask = pd.Series(True, index=df.index)

df_filtered = df[mask].copy()
st.write(f"**Final dataset**: {len(df_filtered):,} proteins retained")

# === 5. ACCEPT ===
st.markdown("### Confirm Setup")
if st.button("Accept This Filtering", type="primary"):
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Filtering accepted** — ready for next step")

st.info("Next: transformation, normalization, and differential analysis")
