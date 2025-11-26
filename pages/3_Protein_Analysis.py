import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# Load data
df = st.session_state.prot_final_df
c1 = st.session_state.prot_final_c1
c2 = st.session_state.prot_final_c2

st.subheader("Replicate Intensity Distribution + Full Statistical Diagnosis")

scale = st.selectbox("Intensity scale", ["Raw", "log₂", "log₁₀"], index=1)

# Transform
data = df[c1 + c2].replace(0, np.nan)
if scale == "log₂": 
    data = np.log2(data)
    y_label = "log₂(Intensity)"
elif scale == "log₁₀": 
    data = np.log10(data)
    y_label = "log₁₀(Intensity)"
else: 
    y_label = "Raw Intensity"

# Melt
melted = data.melt(var_name="Replicate", value_name=y_label).dropna()
melted["Condition"] = melted["Replicate"].apply(lambda x: "A" if x in c1 else "B")
melted["Replicate"] = pd.Categorical(melted["Replicate"], categories=c1 + c2, ordered=True)
melted = melted.sort_values("Replicate")

# Violin + Box
fig = px.violin(
    melted, x="Replicate", y=y_label, color="Condition",
    color_discrete_map={"A": "#E71316", "B": "#1f77b4"},
    box=True, points=False, violinmode="overlay"
)
fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(height=650, yaxis_title=y_label, plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

# === STATISTICAL DIAGNOSIS ===
st.markdown("### Statistical Diagnosis per Replicate")
results = []
for rep in c1 + c2:
    values = data[rep].dropna()
    if len(values) < 4:
        results.append({"Replicate": rep, "n": len(values), "Skew": "N/A", "Kurtosis": "N/A", "Shapiro p": "N/A", "Normal": "N/A"})
        continue
    
    skew = stats.skew(values)
    kurt = stats.kurtosis(values)  # excess kurtosis
    shapiro_stat, shapiro_p = stats.shapiro(values)
    normal = "Yes" if shapiro_p > 0.05 else "No"
    
    results.append({
        "Replicate": rep,
        "n": len(values),
        "Skew": f"{skew:+.3f}",
        "Kurtosis": f"{kurt:+.3f}",
        "Shapiro p": f"{shapiro_p:.2e}",
        "Normal": normal
    })

stats_df = pd.DataFrame(results)
st.dataframe(stats_df, use_container_width=True)

# === AUTOMATIC TRANSFORMATION RECOMMENDATION ===
st.markdown("### Recommended Transformation")
high_skew = any(abs(float(r["Skew"].replace("+","").replace("-","") or "0")) > 0.5 for r in results if r["Skew"] != "N/A")
high_kurt = any(abs(float(r["Kurtosis"].replace("+","").replace("-","") or "0")) > 1.0 for r in results if r["Kurtosis"] != "N/A")
non_normal = any(r["Normal"] == "No" for r in results if r["Normal"] != "N/A")

if scale == "Raw" and (high_skew or high_kurt):
    st.warning("**Strongly recommend log₂ transformation**\n"
               "- High skewness and/or kurtosis detected\n"
               "- Raw proteomics intensities are almost never normal")
elif non_normal:
    st.info("**log₂ transformation recommended**\n"
            "- Most replicates are non-normal (expected in proteomics)\n"
            "- Use non-parametric tests (Wilcoxon, Mann-Whitney)")
else:
    st.success("**Data appears suitable for parametric tests** (rare!)")

st.info("""
**Interpretation (Schessner et al., 2022)**  
- **Skewness > |0.5|** → asymmetric → log transform  
- **Kurtosis > |1|** → heavy tails → log transform  
- **Shapiro-Wilk p < 0.05** → reject normality → use non-parametric stats  
- In practice: **always log₂ transform proteomics intensities**
""")
