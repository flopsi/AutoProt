# pages/3_Protein_Analysis.py (or wherever you want it)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.stats import boxcox, yeojohnson

# Load data
df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2

st.subheader("Intensity Distribution + Transformation Explorer")

# === TRANSFORMATION SELECTION ===
transform = st.selectbox(
    "Apply transformation",
    ["None", "log₂", "log₁₀", "Box-Cox", "Yeo-Johnson"],
    index=1  # default log2
)

# Prepare raw intensities
intensity_raw = df[c1 + c2].replace(0, np.nan)

# Apply selected transformation
if transform == "log₂":
    intensity = np.log2(intensity_raw)
    y_label = "log₂(Intensity)"
elif transform == "log₁₀":
    intensity = np.log10(intensity_raw)
    y_label = "log₁₀(Intensity)"
elif transform == "Box-Cox":
    # Box-Cox requires positive values
    shifted = intensity_raw - intensity_raw.min().min() + 1
    intensity = pd.DataFrame(boxcox(shifted.values.flatten())[0].reshape(shifted.shape),
                             index=shifted.index, columns=shifted.columns)
    y_label = "Box-Cox(Intensity)"
elif transform == "Yeo-Johnson":
    intensity = pd.DataFrame(yeojohnson(intensity_raw.values.flatten())[0].reshape(intensity_raw.shape),
                             index=intensity_raw.index, columns=intensity_raw.columns)
    y_label = "Yeo-Johnson(Intensity)"
else:
    intensity = intensity_raw
    y_label = "Raw Intensity"

# Store both versions
st.session_state.intensity_original = intensity_raw
st.session_state.intensity_transformed = intensity
st.session_state.current_transform = transform

# === PLOT ===
melted = intensity.melt(var_name="Replicate", value_name=y_label).dropna()
melted["Condition"] = melted["Replicate"].apply(lambda x: "A" if x in c1 else "B")
melted["Replicate"] = pd.Categorical(melted["Replicate"], c1 + c2, ordered=True)
melted = melted.sort_values("Replicate")

fig = px.violin(
    melted, x="Replicate", y=y_label, color="Condition",
    color_discrete_map={"A": "#E71316", "B": "#1f77b4"},
    box=True, points=False, violinmode="overlay"
)
fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(height=650, yaxis_title=y_label, plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

# === STATISTICS TABLE ===
st.markdown("### Distribution Statistics")
stats_list = []
for rep in c1 + c2:
    vals = intensity[rep].dropna()
    if len(vals) < 4:
        stats_list.append({"Replicate": rep, "n": len(vals), "Skew": "N/A", "Kurtosis": "N/A", "Shapiro p": "N/A", "Normal": "N/A"})
        continue
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    _, p = stats.shapiro(vals)
    normal = "Yes" if p > 0.05 else "No"
    stats_list.append({
        "Replicate": rep,
        "n": len(vals),
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{p:.2e}",
        "Normal": normal
    })

stats_df = pd.DataFrame(stats_list)
st.dataframe(stats_df, use_container_width=True)

# === RECOMMENDATION ===
skew_ok = all(abs(float(r["Skew"].replace("N/A","0") or "0")) < 0.5 for r in stats_list)
kurt_ok = all(abs(float(r["Kurtosis"].replace("N/A","0") or "0")) < 1.0 for r in stats_list)
normal = all(r["Normal"] == "Yes" for r in stats_list if r["Normal"] != "N/A")

if normal and skew_ok and kurt_ok:
    st.success("**Data is approximately normal** — parametric tests OK")
elif transform == "log₂":
    st.success("**log₂ transformation is excellent** — use this for analysis")
elif transform in ["Box-Cox", "Yeo-Johnson"]:
    st.info(f"**{transform} improved symmetry** — consider using")
else:
    st.warning("**Strongly recommend log₂ transformation** — raw proteomics data is never normal")

st.info("""
**Next step**: Use `st.session_state.intensity_transformed` in all downstream analysis  
**Original data** preserved in `st.session_state.intensity_original`
""")
