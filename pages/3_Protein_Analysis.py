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

# === 1. VIEW MODE ===
st.subheader("1. Select View Mode")
view_species = st.selectbox(
    "Show density plots for:",
    ["All proteins", "HUMAN", "ECOLI", "YEAST"],
    index=0
)

# Prepare data for selected view
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
        fig.add_vline(x=mean, line_dash="dash")
        fig.update_layout(
            height=380,
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # === SMALL 4×3 TABLE BELOW EACH PLOT ===
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

# === 3. FILTERING (unchanged from previous version) ===
st.subheader("3. Filtering Options (Choose One)")
# ... [your filtering code here] ...

# === 4. ACCEPT ===
st.markdown("### Confirm Setup")
if st.button("Accept This Filtering", type="primary"):
    st.session_state.df_filtered = df_filtered
    st.session_state.qc_accepted = True
    st.success("**Filtering accepted** — ready for next step")
