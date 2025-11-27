# pages/1_peptide_Import.py
import streamlit as st
import pandas as pd
import io
import numpy as np

def clear_all_session():
    keys = ["peptide_bytes", "metadata_bytes", "peptide_name", "metadata_name",
            "pep_df", "pep_c1", "pep_c2", "pep_seq_col", "pep_pg_col"]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

st.set_page_config(page_title="Peptide Import", layout="wide")

# === STYLING ===
st.markdown("""
<style>
    .header {background:linear-gradient(90deg,#E71316,#A6192E); padding:20px 40px; color:white; margin:-80px -80px 40px;}
    .header h1,.header p {margin:0;}
    .stButton>button {background:#E71316 !important; color:white !important;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><h1>DIA Proteomics Pipeline</h1><p>Peptide Import + Metadata</p></div>', unsafe_allow_html=True)

# === FILE UPLOAD ===
col1, col2 = st.columns(2)
with col1:
    if "peptide_bytes" not in st.session_state:
        uploaded_pep = st.file_uploader("Upload Wide-Format Peptide File", type=["csv", "tsv", "txt"])
        if uploaded_pep:
            st.session_state.peptide_bytes = uploaded_pep.getvalue()
            st.session_state.peptide_name = uploaded_pep.name
            st.rerun()
    else:
        st.success(f"Peptide: **{st.session_state.peptide_name}**")
with col2:
    if "metadata_bytes" not in st.session_state:
        uploaded_meta = st.file_uploader("Upload Metadata File (metadata.tsv)", type=["tsv", "csv", "txt"])
        if uploaded_meta:
            st.session_state.metadata_bytes = uploaded_meta.getvalue()
            st.session_state.metadata_name = uploaded_meta.name
            st.rerun()
    else:
        st.success(f"Metadata: **{st.session_state.metadata_name}**")

if "peptide_bytes" not in st.session_state or "metadata_bytes" not in st.session_state:
    st.info("Please upload both files.")
    if st.button("Restart / Clear All"):
        clear_all_session()
        st.rerun()
    st.stop()

# === LOAD DATA ===
@st.cache_data(show_spinner="Loading files...")
def load_dataframe(bytes_data):
    text = bytes_data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_raw = load_dataframe(st.session_state.peptide_bytes)
df_meta = load_dataframe(st.session_state.metadata_bytes)

# === METADATA MATCHING ===
rename_dict = {}
used_columns = set()
for _, row in df_meta.iterrows():
    run_label = str(row["Run Label"]).strip()
    condition = str(row["Condition"]).strip()
    replicate = str(row["Replicate"]).strip()
    new_name = f"{condition}{replicate}"
    matches = [c for c in df_raw.columns if run_label in str(c)]
    if not matches:
        st.warning(f"Run Label not found: `{run_label}`")
        continue
    if len(matches) > 1:
        st.error(f"Multiple matches for `{run_label}`: {matches}")
        st.stop()
    col = matches[0]
    if col in used_columns:
        st.error(f"Column `{col}` matched twice!")
        st.stop()
    rename_dict[col] = new_name
    used_columns.add(col)

if not rename_dict:
    st.error("No intensity columns matched!")
    st.stop()

df = df_raw.rename(columns=rename_dict).copy()
c1 = sorted([name for name in rename_dict.values() if name.startswith("A")])
c2 = sorted([name for name in rename_dict.values() if name.startswith("B")])
all_intensity_cols = c1 + c2

# Convert to numeric
for col in all_intensity_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
df[all_intensity_cols] = df[all_intensity_cols].replace([0, np.nan], 1.0)

# === AUTO-DETECT PEPTIDE SEQUENCE COLUMN (>90% end with K or R before _) ===
def detect_peptide_sequence_column(df):
    candidates = []
    for col in df.columns:
        if df[col].dtype != "object": continue
        sample = df[col].dropna().astype(str).head(2000)
        if sample.empty: continue
        pattern = r'[KR](?=[_\.]|$)'
        matches = sample.str.contains(pattern, regex=True)
        ratio = matches.mean()
        if ratio > 0.90:
            candidates.append((col, ratio))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return None

auto_seq_col = detect_peptide_sequence_column(df)

# === AUTO-DETECT PROTEIN GROUP COLUMN ===
pg_candidates = ["Leading razor protein", "Protein IDs", "Fasta headers", "Protein names"]
auto_pg_col = next((c for c in pg_candidates if c in df.columns), df.columns[1])

# === USER COLUMN ASSIGNMENT ===
st.subheader("Column Assignment")

rows = []
for col in df.columns:
    preview = " | ".join(df[col].dropna().astype(str).unique()[:3])
    rows.append({
        "Rename": col,
        "Peptide Sequence": col == auto_seq_col,
        "Protein Group": col == auto_pg_col,
        "Original Name": col,
        "Preview": preview,
        "Type": "Intensity" if col in all_intensity_cols else "Metadata"
    })

edited = st.data_editor(
    pd.DataFrame(rows),
    column_config={
        "Rename": st.column_config.TextColumn("Rename"),
        "Peptide Sequence": st.column_config.CheckboxColumn("Peptide Sequence"),
        "Protein Group": st.column_config.CheckboxColumn("Protein Group"),
        "Original Name": st.column_config.TextColumn("Original", disabled=True),
        "Preview": st.column_config.TextColumn("Preview", disabled=True),
        "Type": st.column_config.TextColumn("Type", disabled=True),
    },
    disabled=["Original Name", "Preview", "Type"],
    hide_index=True,
    use_container_width=True,
    key="pep_col_table"
)

# Extract selections
seq_checked = edited[edited["Peptide Sequence"]]
pg_checked = edited[edited["Protein Group"]]

seq_cols = seq_checked["Original Name"].tolist()
pg_cols = pg_checked["Original Name"].tolist()

errors = []
if len(seq_cols) != 1: errors.append("Select exactly 1 Peptide Sequence column")
if len(pg_cols) != 1: errors.append("Select exactly 1 Protein Group column")
if errors:
    for e in errors: st.error(e)
    st.stop()

pep_seq_col = seq_cols[0]
pep_pg_col = pg_cols[0]

# Rename
rename_map = {}
for _, row in edited.iterrows():
    new = row["Rename"].strip()
    if new and new != row["Original Name"]:
        rename_map[row["Original Name"]] = new

df_final = df.rename(columns=rename_map).copy()

# === FINAL CLEANUP ===
df_final["Sequence"] = df_final[pep_seq_col]
df_final["PG"] = df_final[pep_pg_col].astype(str).str.split(";").str[0]

# Species from protein accession
species_map = {
    "HUMAN": "HUMAN", "ECOLI": "ECOLI", "YEAST": "YEAST",
    "HOMO": "HUMAN", "SACCHA": "YEAST", "ESCHERICHIA": "ECOLI"
}
def get_species(pg):
    if pd.isna(pg): return "Other"
    pg_up = str(pg).upper()
    for key, sp in species_map.items():
        if key in pg_up:
            return sp
    return "Other"

df_final["Species"] = df_final["PG"].apply(get_species)

final_cols = ["Sequence", "PG", "Species"] + all_intensity_cols
df_final = df_final[final_cols].copy()

# === SAVE TO SESSION — EXACT KEYS USED IN ANALYSIS PAGE ===
st.session_state.pep_df = df_final
st.session_state.pep_c1 = c1
st.session_state.pep_c2 = c2
st.session_state.pep_seq_col = "Sequence"
st.session_state.pep_pg_col = "PG"

# === DISPLAY ===
st.success(f"Final dataset: **{len(df_final):,} peptides**")
colA, colB = st.columns(2)
with colA: st.subheader("Condition A"); st.code(" | ".join(c1))
with colB: st.subheader("Condition B"); st.code(" | ".join(c2))

st.write("**Peptides per species:**")
for sp, count in df_final["Species"].value_counts().items():
    st.write(f"• **{sp}**: {count:,}")

st.subheader("Data Preview")
st.dataframe(df_final.head(12), use_container_width=True)

if st.button("Go to Peptide Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Peptide_Analysis.py")

if st.button("Restart Everything"):
    clear_all_session()
    st.rerun()# pages/4_Peptide_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f

# Load peptide data
if "pep_df" not in st.session_state:
    st.error("No peptide data found! Please go to Peptide Import first.")
    if st.button("Go to Peptide Import"):
        st.switch_page("pages/2_Peptide_Import.py")
    st.stop()

df = st.session_state.pep_df.copy()
c1 = st.session_state.pep_c1.copy()
c2 = st.session_state.pep_c2.copy()
all_reps = c1 + c2

st.title("Peptide-Level QC & Visualization (Schessner et al., 2022 Figure 4)")

# === 1. NORMALITY TESTING ON RAW DATA ===
st.subheader("1. Normality Testing on Raw Data (Shapiro-Wilk)")

transform_options = {
    "log₂": lambda x: np.log2(x + 1),
    "log₁₀": lambda x: np.log10(x + 1),
    "Square root": lambda x: np.sqrt(x + 1),
    "Box-Cox": lambda x: boxcox(x + 1)[0] if (x + 1 > 0).all() else None,
    "Yeo-Johnson": lambda x: yeojohnson(x + 1)[0],
}

results = []
best_transform = "log₁₀"
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
st.success(f"Recommended transformation: {best_transform}")

# === 2. DATA VIEW & FILTERING PANEL ===
st.subheader("2. Data Processing & Visualization")

col_t, col_s, col_f = st.columns(3)

with col_t:
    transformation = st.radio("Transformation", ["Recommended", "Raw"], index=0)

with col_s:
    available_species = ["All peptides"]
    if "Species" in df.columns:
        available_species += sorted(df["Species"].dropna().unique().tolist())
    visual_species = st.radio("Visualize species", available_species, index=0)

with col_f:
    filtering = st.radio("Filtering", ["Low intensity", "±2σ filtered", "Combined"], index=2)

# === APPLY TRANSFORMATION & FILTERING ===
df_processed = df.copy()

# Transformation
if transformation == "Recommended":
    func = transform_options[best_transform]
    df_processed[all_reps] = df_processed[all_reps].apply(func)

# Filtering
if filtering in ["Low intensity", "Combined"]:
    mask = (np.log10(df_processed[all_reps].replace(0, np.nan)) >= 0.5).all(axis=1)
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

# Visual species filter
df_visual = df_processed.copy()
if visual_species != "All peptides":
    df_visual = df_visual[df_visual["Species"] == visual_species]

# === 3. 6 DENSITY PLOTS ===
st.subheader("Peptide Intensity Density Plots (log₁₀)")

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

# === 4. PEPTIDE COUNT TABLE ===
st.subheader("Peptide Counts After Processing")
count_data = []
count_data.append({"Category": "All peptides", "Count": len(df_processed)})
if "Species" in df_processed.columns:
    for sp in df_processed["Species"].value_counts().index:
        count_data.append({"Category": sp, "Count": df_processed["Species"].value_counts()[sp]})
st.table(pd.DataFrame(count_data))

# === 5. PCA ON FINAL DATA (6 DOTS) ===
st.subheader("PCA of Replicate Profiles (Schessner et al., 2022 Figure 4)")

df_pca = df_processed[all_reps].copy()
df_pca = df_pca.dropna(how='any')

if len(df_pca) < 10:
    st.warning("Not enough peptides for reliable PCA")
else:
    X = StandardScaler().fit_transform(df_pca.values)
    pca = PCA(n_components=2)
    pc_scores = pca.fit_transform(X.T)  # replicates as samples

    fig = go.Figure()
    for i, rep in enumerate(all_reps):
        color = "#E71316" if rep in c1 else "#1f77b4"
        fig.add_trace(go.Scatter(
            x=[pc_scores[i, 0]],
            y=[pc_scores[i, 1]],
            mode='markers+text',
            name=rep,
            marker=dict(color=color, size=18, line=dict(width=3, color='black')),
            text=rep,
            textposition="top center",
            textfont=dict(size=14)
        ))

    fig.update_layout(
        title=f"PCA (PC1: {pca.explained_variance_ratio_[0]:.1%} • PC2: {pca.explained_variance_ratio_[1]:.1%})",
        xaxis_title="PC1", yaxis_title="PC2",
        height=600, showlegend=False, template="simple_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# === 6. PERMANOVA TEST ===
st.subheader("Replicate Similarity (PERMANOVA)")
dist = squareform(pdist(pc_scores))
groups = ['A'] * len(c1) + ['B'] * len(c2)
n = len(groups)
a = 2
SST = np.sum(dist**2) / (2*n)
SSW = 0
for g in set(groups):
    idx = [i for i, x in enumerate(groups) if x == g]
    if len(idx) > 1:
        SSW += np.sum(dist[np.ix_(idx,idx)]**2) / (2*len(idx))
SSB = SST - SSW
F_stat = (SSB/(a-1)) / (SSW/(n-a)) if (n-a) > 0 else float('inf')
p_val = 1 - f.cdf(F_stat, a-1, n-a)

col1, col2 = st.columns(2)
with col1:
    st.metric("PERMANOVA F", f"{F_stat:.3f}")
with col2:
    st.metric("p-value", f"{p_val:.2e}")

if p_val < 0.05:
    st.error("Significant difference between conditions")
else:
    st.success("No significant difference — excellent technical reproducibility")

# === 7. ACCEPT ===
if st.button("Accept & Proceed", type="primary"):
    st.session_state.pep_intensity_transformed = df_processed[all_reps]
    st.session_state.pep_df_filtered = df_processed
    st.session_state.pep_qc_accepted = True
    st.success("Peptide data ready!")
    st.balloons()
