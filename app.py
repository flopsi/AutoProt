import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import io

st.set_page_config(page_title="LFQbench Analysis with Column Renaming", layout="wide")
st.title("LFQbench-style Analysis – With Column Renaming & Species Extraction")

# -------------------------------------------------
# 1. Upload file
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your proteomics file (CSV/TSV/TXT)",
    type=["csv", "tsv", "txt"]
)

if uploaded_file is None:
    st.info("Upload a file to start. Test with your provided prot3.csv.")
    st.stop()

# -------------------------------------------------
# 2. Load data with robust parsing
# -------------------------------------------------
@st.cache_data
def load_data(file):
    content = file.getvalue().decode("utf-8")
    # Handle BOM if present
    if content.startswith('\ufeff'):
        content = content[1:]
    df = pd.read_csv(io.StringIO(content), sep=None, engine='python')
    
    # Convert all potential numeric columns to float, coercing errors
    for col in df.columns:
        if col not in ['pg', 'name']:  # Exclude known non-numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    st.success("File loaded and numeric columns converted to float.")
    return df

df_raw = load_data(uploaded_file)

st.write("Raw Data Preview (first 5 rows):")
st.dataframe(df_raw.head())

# -------------------------------------------------
# 3. Extract Species from 'name' column (assumes "ID,species" format)
# -------------------------------------------------
if 'name' in df_raw.columns:
    split_cols = df_raw['name'].str.split(',', expand=True, n=1)
    if split_cols.shape[1] == 2:
        df_raw.insert(1, 'ProteinID', split_cols[0])
        df_raw.insert(2, 'Species', split_cols[1])
        df_raw = df_raw.drop(columns=['name'])
        st.success("Extracted 'ProteinID' and 'Species' from 'name' column.")
    else:
        st.warning("'name' column does not contain comma-separated values. Proceeding without split.")

# Update columns after split
df = df_raw.copy()

# -------------------------------------------------
# 4. Rename Columns (User Editable)
# -------------------------------------------------
st.subheader("Rename Columns")
st.write("Edit the new names below. Empty fields keep original names.")

rename_dict = {}
cols = df.columns.tolist()
rename_inputs = {}
for col in cols:
    new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
    if new_name != col and new_name.strip() != "":
        rename_dict[col] = new_name.strip()

if rename_dict:
    df = df.rename(columns=rename_dict)
    st.success(f"Renamed columns: {rename_dict}")
else:
    st.info("No renames applied.")

st.write("DataFrame after renaming:")
st.dataframe(df.head())

# -------------------------------------------------
# 5. Auto-detect numeric (intensity) columns
# -------------------------------------------------
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.error("No numeric columns detected. Check data types.")
    st.stop()

st.success(f"Found {len(numeric_cols)} numeric columns:")
st.write(numeric_cols)

# -------------------------------------------------
# 6. Select Species column
# -------------------------------------------------
species_col_options = [c for c in df.columns if 'Species' in c or df[c].dtype == 'object']
if 'Species' in df.columns:
    default_species = 'Species'
else:
    default_species = st.selectbox("Select species column", options=species_col_options)

species_col = st.selectbox("Confirm species column:", options=df.columns, index=df.columns.get_loc(default_species) if default_species in df.columns else 0)

unique_species = df[species_col].dropna().unique()
st.write("Detected species:", unique_species)

# -------------------------------------------------
# 7. Set expected log2 ratios
# -------------------------------------------------
expected_ratios = {}
st.subheader("Set expected log₂ ratios (Condition 2 / Condition 1)")
for sp in unique_species:
    default = 0.0
    sp_str = str(sp).lower()
    if 'human' in sp_str:
        default = 0.0
    elif 'yeast' in sp_str:
        default = 1.0  # Example: 2:1
    elif 'ecoli' in sp_str or 'e.coli' in sp_str:
        default = -2.0  # Example: 1:4

    expected_ratios[sp] = st.number_input(
        f"Expected log₂ for {sp}",
        value=default,
        step=0.1,
        key=f"exp_{sp}"
    )

# -------------------------------------------------
# 8. Select condition columns (using renamed names)
# -------------------------------------------------
st.subheader("Assign Replicates to Conditions")
cond1_cols = st.multiselect("Condition 1 (e.g., A replicates)", options=numeric_cols)
cond2_cols = st.multiselect("Condition 2 (e.g., B replicates)", options=numeric_cols)

if set(cond1_cols) & set(cond2_cols):
    st.error("Cannot assign same column to both conditions!")
    st.stop()

# -------------------------------------------------
# 9. Run Analysis
# -------------------------------------------------
if st.button("Run LFQbench Analysis", type="primary"):
    if len(cond1_cols) == 0 or len(cond2_cols) == 0:
        st.error("Select at least one column per condition.")
        st.stop()

    # Prepare working df (use renamed columns)
    work = df.copy()
    intensity_cols = cond1_cols + cond2_cols
    work = work.dropna(subset=intensity_cols + [species_col])
    work = work[(work[intensity_cols] > 0).all(axis=1)]  # Filter zeros

    # Means and log2 ratios (Condition2 / Condition1)
    work["mean_cond1"] = work[cond1_cols].mean(axis=1)
    work["mean_cond2"] = work[cond2_cols].mean(axis=1)
    work["log2_ratio"] = np.log2(work["mean_cond2"] / work["mean_cond1"])

    # p-values
    pvals = []
    for _, row in work.iterrows():
        _, p = ttest_ind(row[cond1_cols], row[cond2_cols], equal_var=False)
        pvals.append(p if not np.isnan(p) else 1.0)
    work["p_value"] = pvals
    work["-log10_p"] = -np.log10(work["p_value"].clip(lower=1e-10))

    # -------------------------------------------------
    # 10. Results
    # -------------------------------------------------
    st.success("Analysis Complete!")

    # Summary Table
    summary = (
        work.groupby(species_col)["log2_ratio"]
        .agg(["count", "mean", "std"])
        .round(4)
        .reset_index()
    )
    summary["expected"] = summary[species_col].map(expected_ratios)
    summary["bias"] = summary["mean"] - summary["expected"]
    summary.rename(columns={"mean": "observed_mean", "std": "precision_1SD"}, inplace=True)

    st.subheader("LFQbench Summary Table")
    st.dataframe(summary.style.format("{:.3f}"))

    # Boxplot
    st.subheader("Log₂ Ratio Distribution by Species")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=work, x=species_col, y="log2_ratio", ax=ax)
    for i, sp in enumerate(unique_species):
        ax.axhline(expected_ratios[sp], color="red", linestyle="--", linewidth=2,
                   xmin=i/len(unique_species), xmax=(i+1)/len(unique_species))
    ax.set_title("Observed vs Expected Log₂ Ratios")
    st.pyplot(fig)

    # Density Plot
    st.subheader("Density Plot of Log₂ Ratios")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for sp in unique_species:
        subset = work[work[species_col] == sp]
        if not subset.empty:
            sns.kdeplot(subset["log2_ratio"], label=str(sp), fill=True, alpha=0.5, ax=ax2)
    for sp in unique_species:
        ax2.axvline(expected_ratios[sp], color="red", linestyle="--")
    ax2.legend()
    st.pyplot(fig2)

    # Download full results (with renamed columns)
    csv = work.to_csv(index=False).encode()
    st.download_button(
        "Download Full Results (CSV)",
        csv,
        "lfqbench_results_renamed.csv",
        "text/csv"
    )
