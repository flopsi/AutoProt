# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import io

st.set_page_config(page_title="LFQbench Quick Test", layout="wide")
st.title("LFQbench-style Analysis – Quick Test Version")

# -------------------------------------------------
# 1. Upload file
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your proteinGroups.txt or any MaxQuant/LFQ file",
    type=["txt", "csv", "tsv"]
)

if uploaded_file is None:
    st.info("Upload a file to start. You can test with the official LFQbench example file:")
    st.markdown(
        "[Download example proteinGroups.txt (3-species mix)](https://raw.githubusercontent.com/cox-labs/LFQbench/master/example_data/proteinGroups.txt)"
    )
    st.stop()

# -------------------------------------------------
# 2. Load data
# -------------------------------------------------
@st.cache_data
def load_data(file):
    content = file.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(content), sep='\t', low_memory=False)
        st.success("Tab-separated file loaded")
    except:
        df = pd.read_csv(io.BytesIO(content), sep=',')
        st.success("Comma-separated file loaded")
    return df

df = load_data(uploaded_file)

st.write("First 5 rows preview:")
st.dataframe(df.head())

# -------------------------------------------------
# 3. Auto-detect LFQ columns (most users have these)
# -------------------------------------------------
lfq_cols = [c for c in df.columns if c.startswith("LFQ intensity ")]
if not lfq_cols:
    # fallback for newer MaxQuant versions
    lfq_cols = [c for c in df.columns if "LFQ intensity" in c]

if not lfq_cols:
    st.error("No LFQ intensity columns found. Check column names.")
    st.stop()

st.success(f"Found {len(lfq_cols)} LFQ columns:")
st.write(lfq_cols)

# -------------------------------------------------
# 4. Species column – try common names
# -------------------------------------------------
possible_species_cols = ["Organism", "Species", "Organisms", "Taxonomy", "Fasta headers", "Fasta header"]
species_col = None
for col in possible_species_cols:
    if col in df.columns:
        species_col = col
        break

if species_col is None:
    st.warning("Could not auto-detect species column. Please choose manually:")
    species_col = st.selectbox("Select species/organism column", options=df.columns)

else:
    st.info(f"Auto-selected species column: **{species_col}**")
    if st.checkbox("Change species column"):
        species_col = st.selectbox("Select species column", options=df.columns, index=df.columns.get_loc(species_col))

# -------------------------------------------------
# 5. Show unique species and set expected ratios
# -------------------------------------------------
unique_species = df[species_col].dropna().unique()
st.write("Detected species:", unique_species)

expected_ratios = {}
st.subheader("Set expected log₂ ratios (Condition 2 / Condition 1)")
for sp in unique_species:
    default = 0.0
    if "Homo sapiens" in str(sp) or "human" in str(sp).lower():
        default = 0.0
    elif "Saccharomyces" in str(sp) or "yeast" in str(sp).lower():
        default = 1.0   # 2:1 example
    elif "Escherichia" in str(sp) or "E.coli" in str(sp):
        default = -2.0  # 1:4 example

    expected_ratios[sp] = st.number_input(
        f"Expected log₂ ratio for {sp}",
        value=default,
        step=0.1,
        key=f"exp_{sp}"
    )

# -------------------------------------------------
# 6. Select condition columns
# -------------------------------------------------
st.subheader("Assign replicates to conditions")
cond1_cols = st.multiselect("Condition 1 replicates (e.g., control)", options=lfq_cols)
cond2_cols = st.multiselect("Condition 2 replicates (e.g., treatment)", options=lfq_cols)

if set(cond1_cols) & set(cond2_cols):
    st.error("Same column cannot be in both conditions!")
    st.stop()

if st.button("Run LFQbench Analysis", type="primary"):
    if len(cond1_cols) == 0 or len(cond2_cols) == 0:
        st.error("Select at least one column per condition")
        st.stop()

    # -------------------------------------------------
    # 7. Core LFQbench calculations
    # -------------------------------------------------
    work = df.copy()
    intensity_cols = cond1_cols + cond2_cols
    work = work.dropna(subset=intensity_cols + [species_col])
    work = work[(work[intensity_cols] > 0).all(axis=1)]

    work["mean_cond1"] = work[cond1_cols].mean(axis=1)
    work["mean_cond2"] = work[cond2_cols].mean(axis=1)
    work["log2_ratio"] = np.log2(work["mean_cond2"] / work["mean_cond1"])   # note: reversed for classic LFQbench view
    work["species"] = work[species_col]

    # p-values
    pvals = []
    for _, row in work.iterrows():
        _, p = ttest_ind(row[cond1_cols], row[cond2_cols], equal_var=False)
        pvals.append(p if not np.isnan(p) else 1.0)
    work["p_value"] = pvals
    work["-log10_p"] = -np.log10(work["p_value"].replace(0, np.nan)).fillna(0)

    # -------------------------------------------------
    # 8. Results
    # -------------------------------------------------
    st.success("Analysis complete!")

    # Summary table
    summary = (
        work.groupby("species")["log2_ratio"]
        .agg(["count", "mean", "std"])
        .round(4)
        .reset_index()
    )
    summary["expected"] = summary["species"].map(expected_ratios)
    summary["bias"] = summary["mean"] - summary["expected"]
    summary.rename(columns={"mean": "observed_mean", "std": "precision_1SD"}, inplace=True)

    st.subheader("LFQbench Summary Table")
    st.dataframe(summary.style.format("{:.3f}"))

    # Boxplot
    st.subheader("Log₂ Ratio Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=work, x="species", y="log2_ratio", ax=ax)
    for i, sp in enumerate(unique_species):
        ax.axhline(expected_ratios[sp], color="red", linestyle="--", linewidth=2)
    ax.set_title("Observed vs Expected Log₂ Ratios")
    st.pyplot(fig)

    # Density plot
    st.subheader("Density Plot")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for sp in unique_species:
        subset = work[work["species"] == sp]
        if len(subset) > 0:
            sns.kdeplot(subset["log2_ratio"], label=str(sp), fill=True, alpha=0.5, ax=ax2)
    for sp in unique_species:
        ax2.axvline(expected_ratios[sp], color="red", linestyle="--")
    ax2.legend()
    st.pyplot(fig2)

    # Download results
    csv = work.to_csv(index=False).encode()
    st.download_button("Download full results (CSV)", csv, "lfqbench_results.csv", "text/csv")
