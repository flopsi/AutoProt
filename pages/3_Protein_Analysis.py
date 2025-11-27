# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
if "prot_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_df.copy()
c1 = st.session_state.prot_c1.copy()
c2 = st.session_state.prot_c2.copy()
all_reps = c1 + c2

st.title("Protein-Level QC & Visualization (Schessner et al., 2022 Figure 4)")

# === 1. NORMALITY TESTING & RECOMMENDED TRANSFORMATION ===
st.subheader("1. Normality Testing & Recommended Transformation")

transform_options = {
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
    "Square root": lambda x: np.sqrt(x + 1),
    "Box-Cox": lambda x: boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: yeojohnson(x + 1)[0],
}

results = []
best_transform = "log₁₀"  # default
best_w = 0

for rep in all_reps:
    raw_vals = df[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8: continue
        
    row = {"Replicate": rep}
    w_raw, p_raw = stats.shapiro(raw_vals)
    row["Raw W"] = f"{w_raw:.4f}"
    
    rep_best = "log₁₀"
    rep_w = 0
    
    for name, func in transform_options.items():
        try:
            t_vals = func(raw_vals)
            if t_vals is None or np.any(np.isnan(t_vals)): continue
            w, _ = stats.shapiro(t_vals)
            row[f"{name} W"] = f"{w:.4f}"
            if w > rep_w:
                rep_w = w
                rep_best = name
        except:
            row[f"{name} W"] = "—"
    
    row["Best"] = rep_best
    if rep_w > best_w:
        best_w = rep_w
        best_transform = rep_best
        
    results.append(row)

st.table(pd.DataFrame(results))
st.success(f"**Recommended transformation: {best_transform}**")

# === 2. DATA VIEW PANEL — LEFT TO RIGHT ===
st.subheader("2. Data Processing & Visualization Options")

col_t, col_s, col_f = st.columns(3)

with col_t:
    transformation = st.radio(
        "Transformation",
        ["Recommended", "Raw"],
        index=0,
        help="Recommended = highest Shapiro-Wilk W"
    )

with col_s:
    available_species = ["All proteins"]
    if "Species" in df.columns:
        available_species += sorted(df["Species"].dropna().unique().tolist())
    visual_species = st.radio("Visualize species", available_species, index=0)

with col_f:
    filtering = st.radio(
        "Filtering",
        ["Low intensity", "±2σ filtered", "Combined"],
        index=2  # default Combined
    )

# === APPLY TRANSFORMATION & FILTERING (FOR ALL DOWNSTREAM) ===
df_processed = df.copy()

# Transformation
if transformation == "Recommended":
    func = transform_options[best_transform]
    df_processed[all_reps] = df_processed[all_reps].apply(func)

# Filtering
log10_full = np.log10(df_processed[all_reps].replace(0, np.nan))

if filtering in ["Low intensity", "Combined"]:
    mask = (log10_full >= 0.5).all(axis=1)
    df_processed = df_processed[mask]

if filtering in ["±2σ filtered", "Combined"]:
    mask = pd.Series(True, index=df_processed.index)
    log10_current = np.log10(df_processed[all_reps].replace(0, np.nan))
    for rep in all_reps:
        vals = log10_current[rep].dropna()
        if len(vals) == 0: continue
        mean, std = vals.mean(), vals.std()
        mask &= (log10_current[rep] >= mean - 2*std) & (log10_current[rep] <= mean + 2*std)
    df_processed = df_processed[mask]

# Species filter — only for visualization
df_visual = df_processed.copy()
if visual_species != "All proteins":
    df_visual = df_visual[df_visual["Species"] == visual_species]

# === 3. 6 DENSITY PLOTS (Schessner et al., 2022 Figure 4) ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = df_visual[rep].replace(0, np.nan).dropna()
        if len(vals) == 0:
            st.write("No data")
            continue
            
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
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 4. PROTEIN COUNT TABLE (Schessner et al., 2022 Table 1) ===
st.subheader("Protein Counts After Processing")
count_data = []
count_data.append({"Species": "All proteins", "Proteins": len(df_processed)})
if "Species" in df_processed.columns:
    for sp in df_processed["Species"].value_counts().index:
        count_data.append({"Species": sp, "Proteins": df_processed["Species"].value_counts()[sp]})
st.table(pd.DataFrame(count_data).style.format("{:,}"))

# === 5. PCA ON FINAL DATA (Schessner et al., 2022 Figure 4) ===
st.subheader("PCA on Final Processed Data")

# Mean intensity per replicate
def get_pca_data(df_subset):
    mean_vals = []
    labels = []
    colors = []
    for rep in all_reps:
        vals = df_subset[rep].replace(0, np.nan).dropna()
        if len(vals) == 0: continue
        mean_vals.append(vals.mean())
        labels.append(rep)
        colors.append("#E71316" if rep in c1 else "#1f77b4")
    return np.array(mean_vals).reshape(-1, 1), labels, colors

# All proteins
df_pca_all = df_processed.copy()
X_all, labels_all, colors_all = get_pca_data(df_pca_all)

# Human only (if exists)
df_pca_human = df_processed[df_processed["Species"] == "HUMAN"] if "HUMAN" in df_processed["Species"].values else pd.DataFrame()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**All Proteins**")
    if len(X_all) < 2:
        st.write("Not enough data")
    else:
        pca = PCA(n_components=2)
        pc = pca.fit_transform(StandardScaler().fit_transform(X_all))
        fig = go.Figure()
        for i, label in enumerate(labels_all):
            fig.add_trace(go.Scatter(
                x=[pc[i, 0]], y=[pc[i, 1]],
                mode='markers+text',
                name=label,
                marker=dict(color=colors_all[i], size=16),
                text=label,
                textposition="top center"
            ))
        fig.update_layout(
            title=f"PCA (All: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%})",
            xaxis_title="PC1", yaxis_title="PC2",
            height=500, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key="pca_all")

with col2:
    st.markdown("**Human Proteins Only**")
    if df_pca_human.empty or len(df_pca_human) < 10:
        st.write("Not enough human proteins")
    else:
        X_h, labels_h, colors_h = get_pca_data(df_pca_human)
        if len(X_h) < 2:
            st.write("Not enough data")
        else:
            pca_h = PCA(n_components=2)
            pc_h = pca_h.fit_transform(StandardScaler().fit_transform(X_h))
            fig_h = go.Figure()
            for i, label in enumerate(labels_h):
                fig_h.add_trace(go.Scatter(
                    x=[pc_h[i, 0]], y=[pc_h[i, 1]],
                    mode='markers+text',
                    name=label,
                    marker=dict(color=colors_h[i], size=16),
                    text=label,
                    textposition="top center"
                ))
            fig_h.update_layout(
                title=f"PCA (Human: {pca_h.explained_variance_ratio_[0]:.1%} + {pca_h.explained_variance_ratio_[1]:.1%})",
                xaxis_title="PC1", yaxis_title="PC2",
                height=500, showlegend=False
            )
            st.plotly_chart(fig_h, use_container_width=True, key="pca_human")

# === 6. ACCEPT ===
if st.button("Accept & Proceed to Differential Analysis", type="primary"):
    st.session_state.intensity_transformed = df_processed[all_reps]
    st.session_state.df_filtered = df_processed
    st.session_state.transform_applied = best_transform
    st.session_state.qc_accepted = True
    st.success("Ready for differential analysis!")
    st.balloons()# pages/3_Protein_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
if "prot_df" not in st.session_state:
    st.error("No protein data found! Please go to Protein Import first.")
    st.stop()

df = st.session_state.prot_df.copy()
c1 = st.session_state.prot_c1.copy()
c2 = st.session_state.prot_c2.copy()
all_reps = c1 + c2

st.title("Protein-Level QC & Visualization (Schessner et al., 2022 Figure 4)")

# === 1. NORMALITY TESTING & RECOMMENDED TRANSFORMATION ===
st.subheader("1. Normality Testing & Recommended Transformation")

transform_options = {
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
    "Square root": lambda x: np.sqrt(x + 1),
    "Box-Cox": lambda x: boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: yeojohnson(x + 1)[0],
}

results = []
best_transform = "log₁₀"  # default
best_w = 0

for rep in all_reps:
    raw_vals = df[rep].replace(0, np.nan).dropna()
    if len(raw_vals) < 8: continue
        
    row = {"Replicate": rep}
    w_raw, p_raw = stats.shapiro(raw_vals)
    row["Raw W"] = f"{w_raw:.4f}"
    
    rep_best = "log₁₀"
    rep_w = 0
    
    for name, func in transform_options.items():
        try:
            t_vals = func(raw_vals)
            if t_vals is None or np.any(np.isnan(t_vals)): continue
            w, _ = stats.shapiro(t_vals)
            row[f"{name} W"] = f"{w:.4f}"
            if w > rep_w:
                rep_w = w
                rep_best = name
        except:
            row[f"{name} W"] = "—"
    
    row["Best"] = rep_best
    if rep_w > best_w:
        best_w = rep_w
        best_transform = rep_best
        
    results.append(row)

st.table(pd.DataFrame(results))
st.success(f"**Recommended transformation: {best_transform}**")

# === 2. DATA VIEW PANEL — LEFT TO RIGHT ===
st.subheader("2. Data Processing & Visualization Options")

col_t, col_s, col_f = st.columns(3)

with col_t:
    transformation = st.radio(
        "Transformation",
        ["Recommended", "Raw"],
        index=0,
        help="Recommended = highest Shapiro-Wilk W"
    )

with col_s:
    available_species = ["All proteins"]
    if "Species" in df.columns:
        available_species += sorted(df["Species"].dropna().unique().tolist())
    visual_species = st.radio("Visualize species", available_species, index=0)

with col_f:
    filtering = st.radio(
        "Filtering",
        ["Low intensity", "±2σ filtered", "Combined"],
        index=2  # default Combined
    )

# === APPLY TRANSFORMATION & FILTERING (FOR ALL DOWNSTREAM) ===
df_processed = df.copy()

# Transformation
if transformation == "Recommended":
    func = transform_options[best_transform]
    df_processed[all_reps] = df_processed[all_reps].apply(func)

# Filtering
log10_full = np.log10(df_processed[all_reps].replace(0, np.nan))

if filtering in ["Low intensity", "Combined"]:
    mask = (log10_full >= 0.5).all(axis=1)
    df_processed = df_processed[mask]

if filtering in ["±2σ filtered", "Combined"]:
    mask = pd.Series(True, index=df_processed.index)
    log10_current = np.log10(df_processed[all_reps].replace(0, np.nan))
    for rep in all_reps:
        vals = log10_current[rep].dropna()
        if len(vals) == 0: continue
        mean, std = vals.mean(), vals.std()
        mask &= (log10_current[rep] >= mean - 2*std) & (log10_current[rep] <= mean + 2*std)
    df_processed = df_processed[mask]

# Species filter — only for visualization
df_visual = df_processed.copy()
if visual_species != "All proteins":
    df_visual = df_visual[df_visual["Species"] == visual_species]

# === 3. 6 DENSITY PLOTS (Schessner et al., 2022 Figure 4) ===
st.subheader("Intensity Density Plots (log₁₀)")

row1, row2 = st.columns(3), st.columns(3)
for i, rep in enumerate(all_reps):
    col = row1[i] if i < 3 else row2[i-3]
    with col:
        vals = df_visual[rep].replace(0, np.nan).dropna()
        if len(vals) == 0:
            st.write("No data")
            continue
            
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
            title=f"<b>{rep}</b>",
            xaxis_title="log₁₀(Intensity)",
            yaxis_title="Density",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# === 4. PROTEIN COUNT TABLE (Schessner et al., 2022 Table 1) ===
st.subheader("Protein Counts After Processing")
count_data = []
count_data.append({"Species": "All proteins", "Proteins": len(df_processed)})
if "Species" in df_processed.columns:
    for sp in df_processed["Species"].value_counts().index:
        count_data.append({"Species": sp, "Proteins": df_processed["Species"].value_counts()[sp]})
st.table(pd.DataFrame(count_data).style.format("{:,}"))

# === 5. PCA ON FINAL DATA (Schessner et al., 2022 Figure 4) ===
st.subheader("PCA on Final Processed Data")

# Mean intensity per replicate
def get_pca_data(df_subset):
    mean_vals = []
    labels = []
    colors = []
    for rep in all_reps:
        vals = df_subset[rep].replace(0, np.nan).dropna()
        if len(vals) == 0: continue
        mean_vals.append(vals.mean())
        labels.append(rep)
        colors.append("#E71316" if rep in c1 else "#1f77b4")
    return np.array(mean_vals).reshape(-1, 1), labels, colors

# All proteins
df_pca_all = df_processed.copy()
X_all, labels_all, colors_all = get_pca_data(df_pca_all)

# Human only (if exists)
df_pca_human = df_processed[df_processed["Species"] == "HUMAN"] if "HUMAN" in df_processed["Species"].values else pd.DataFrame()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**All Proteins**")
    if len(X_all) < 2:
        st.write("Not enough data")
    else:
        pca = PCA(n_components=2)
        pc = pca.fit_transform(StandardScaler().fit_transform(X_all))
        fig = go.Figure()
        for i, label in enumerate(labels_all):
            fig.add_trace(go.Scatter(
                x=[pc[i, 0]], y=[pc[i, 1]],
                mode='markers+text',
                name=label,
                marker=dict(color=colors_all[i], size=16),
                text=label,
                textposition="top center"
            ))
        fig.update_layout(
            title=f"PCA (All: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%})",
            xaxis_title="PC1", yaxis_title="PC2",
            height=500, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key="pca_all")

with col2:
    st.markdown("**Human Proteins Only**")
    if df_pca_human.empty or len(df_pca_human) < 10:
        st.write("Not enough human proteins")
    else:
        X_h, labels_h, colors_h = get_pca_data(df_pca_human)
        if len(X_h) < 2:
            st.write("Not enough data")
        else:
            pca_h = PCA(n_components=2)
            pc_h = pca_h.fit_transform(StandardScaler().fit_transform(X_h))
            fig_h = go.Figure()
            for i, label in enumerate(labels_h):
                fig_h.add_trace(go.Scatter(
                    x=[pc_h[i, 0]], y=[pc_h[i, 1]],
                    mode='markers+text',
                    name=label,
                    marker=dict(color=colors_h[i], size=16),
                    text=label,
                    textposition="top center"
                ))
            fig_h.update_layout(
                title=f"PCA (Human: {pca_h.explained_variance_ratio_[0]:.1%} + {pca_h.explained_variance_ratio_[1]:.1%})",
                xaxis_title="PC1", yaxis_title="PC2",
                height=500, showlegend=False
            )
            st.plotly_chart(fig_h, use_container_width=True, key="pca_human")

# === 6. ACCEPT ===
if st.button("Accept & Proceed to Differential Analysis", type="primary"):
    st.session_state.intensity_transformed = df_processed[all_reps]
    st.session_state.df_filtered = df_processed
    st.session_state.transform_applied = best_transform
    st.session_state.qc_accepted = True
    st.success("Ready for differential analysis!")
    st.balloons()
