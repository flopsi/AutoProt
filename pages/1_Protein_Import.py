# === LOAD DATA ===
@st.cache_data(show_spinner="Loading data...")
def load_df(b):
    text = b.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"): text = text[1:]
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")

df_data = load_df(st.session_state.protein_bytes)   # or peptide_bytes
df_meta = load_df(st.session_state.metadata_bytes)

st.write(f"Data: **{df_data.shape[0]:,}** rows × **{df_data.shape[1]}** columns")
st.write(f"Metadata: **{len(df_meta)}** runs")

# === CONTAINS-BASED MATCHING (Run Label substring in column name) ===
rename_dict = {}
correction_dict = {}
used_columns = set()

for idx, row in df_meta.iterrows():
    run_label = str(row["Run Label"]).strip()
    condition = str(row["Condition"]).strip()
    replicate = str(row["Replicate"]).strip()
    new_col_name = f"{condition}{replicate}"
    factor = float(row.get("Quantity Correction Factor", 1.0))

    # Find columns that CONTAIN the Run Label (case-sensitive, exact substring)
    matches = [col for col in df_data.columns if run_label in str(col)]

    if len(matches) == 0:
        st.warning(f"Run Label not found: `{run_label}`")
        continue
    if len(matches) > 1:
        st.error(f"Multiple columns contain Run Label `{run_label}`: {matches}")
        st.stop()
    
    matched_col = matches[0]
    if matched_col in used_columns:
        st.error(f"Column `{matched_col}` matched by multiple Run Labels!")
        st.stop()

    rename_dict[matched_col] = new_col_name
    correction_dict[new_col_name] = factor
    used_columns.add(matched_col)

if not rename_dict:
    st.error("No Run Labels matched any column in the data file!")
    st.stop()

# === RENAME COLUMNS ===
df_data = df_data.rename(columns=rename_dict)
st.success(f"Matched and renamed **{len(rename_dict)}** columns using metadata (contains-based)")

# === EXTRACT CONDITION GROUPS ===
all_intensity_cols = list(rename_dict.values())
cond_a = sorted([c for c in all_intensity_cols if c.startswith("A")])
cond_b = sorted([c for c in all_intensity_cols if c.startswith("B")])

if not cond_a or not cond_b:
    st.error(f"Could not find both conditions. Found A: {len(cond_a)}, B: {len(cond_b)}")
    st.stop()

st.write("**Condition A** →", ", ".join(cond_a))
st.write("**Condition B** →", ", ".join(cond_b))

# === FORCE FLOAT64 ON INTENSITY COLUMNS ===
intensity_columns = cond_a + cond_b
for col in intensity_columns:
    df_data[col] = pd.to_numeric(df_data[col], errors='coerce').astype('float64')

# Replace 0 and NaN with 1.0 (standard in proteomics)
df_data[intensity_columns] = df_data[intensity_columns].replace([0, np.nan], 1.0)

# === APPLY CORRECTION FACTORS ===
if st.checkbox("Apply Quantity Correction Factors from metadata", value=True):
    for col, factor in correction_dict.items():
        if col in df_data.columns and factor != 1.0:
            before = df_data[col].median()
            df_data[col] = df_data[col] * factor
            after = df_data[col].median()
            st.info(f"Applied factor {factor:.3f} to {col} (median: {before:,.0f} → {after:,.0f})")

# === SAVE TO SESSION STATE ===
st.session_state.prot_df = df_data
st.session_state.prot_c1 = cond_a
st.session_state.prot_c2 = cond_b
st.session_state.metadata_df = df_meta  # optional: for later use

st.success("Protein data fully processed with metadata!")
st.write(f"Final intensity columns: {len(intensity_columns)} (all float64)")

if st.button("Go to Protein Analysis", type="primary", use_container_width=True):
    st.switch_page("pages/3_Protein_Analysis.py")

restart_button()
