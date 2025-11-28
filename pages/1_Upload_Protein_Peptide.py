# pages/1_Upload_Protein_Peptide.py
import io
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Loading file...")
def read_table(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")


def infer_species_from_name(df: pd.DataFrame, name_col: str | None, key_suffix: str) -> pd.Series:
    species_keywords = {
        "HUMAN": ["HUMAN", "HOMO", "HSA"],
        "MOUSE": ["MOUSE", "MUS", "MMU"],
        "RAT": ["RAT", "RATTUS", "RNO"],
        "ECOLI": ["ECOLI", "ESCHERICHIA"],
        "YEAST": ["YEAST", "SACCHA", "CEREVISIAE"],
        "BOVIN": ["BOVIN", "BOVINE", "BOS"],
    }
    if name_col is None or name_col not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index)

    selected = st.multiselect(
        "Species tags to use (name-based)",
        options=list(species_keywords.keys()),
        default=["HUMAN", "ECOLI", "YEAST"],
        key=f"species_tags_{key_suffix}",
    )
    lookup = {kw: sp for sp in selected for kw in species_keywords[sp]}

    def detect(v):
        if pd.isna(v):
            return "Other"
        s = str(v).upper()
        for kw, sp in lookup.items():
            if kw in s:
                return sp
        return "Other"

    return df[name_col].apply(detect)


def protein_mapping(df_raw: pd.DataFrame):
    st.subheader("Protein mapping")

    roles = [
        "Ignore",
        "Protein Group ID",
        "Protein Name / Description",
        "Species from this column",
        "Quantitative",
    ]

    numeric = set(df_raw.select_dtypes(include="number").columns.tolist())
    data = []
    for col in df_raw.columns:
        role = "Quantitative" if col in numeric else "Ignore"
        cl = col.lower()
        if "pg." in cl or "protein group" in cl or "protein ids" in cl:
            role = "Protein Group ID"
        elif "name" in cl:
            role = "Protein Name / Description"
        data.append({"column_name": col, "role": role})

    mapping_df = pd.DataFrame(data)

    edited = st.data_editor(
        mapping_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "column_name": st.column_config.TextColumn("Column name", disabled=True),
            "role": st.column_config.SelectboxColumn(
                "Role",
                options=roles,
                help="Assign how each column should be used.",
            ),
        },
        hide_index=True,
        key="protein_mapping_editor",
    )

    prot_id_cols = edited.loc[edited["role"] == "Protein Group ID", "column_name"]
    if prot_id_cols.empty:
        st.error("Please assign exactly one 'Protein Group ID' column.")
        return
    if len(prot_id_cols) > 1:
        st.error("Multiple columns marked as 'Protein Group ID'. Choose only one.")
        return
    protein_id_col = prot_id_cols.iloc[0]

    quant_cols = edited.loc[edited["role"] == "Quantitative", "column_name"].tolist()
    if not quant_cols:
        st.error("Please mark at least one column as 'Quantitative'.")
        return

    species_cols = edited.loc[edited["role"] == "Species from this column", "column_name"]
    name_cols = edited.loc[edited["role"] == "Protein Name / Description", "column_name"]

    df = df_raw.copy().set_index(protein_id_col)

    if not species_cols.empty:
        species = df[species_cols.iloc[0]].astype(str)
    else:
        name_col_for_species = name_cols.iloc[0] if not name_cols.empty else None
        species = infer_species_from_name(df, name_col_for_species, key_suffix="protein")

    quant_df = df[quant_cols]

    st.markdown("**Preview transformed protein table**")
    st.write(f"Proteins: {df.shape[0]:,}, Quant columns: {len(quant_cols)}")
    st.write("Species counts:")
    st.write(species.value_counts())

    preview = df.assign(Species=species)[quant_cols + ["Species"]].head(10)
    st.dataframe(preview, use_container_width=True)

    st.session_state["protein_upload"] = {
        "df_raw": df_raw,
        "df": df,
        "quant_cols": quant_cols,
        "species": species,
        "protein_id_col": protein_id_col,
        "mapping_table": edited,
        "level": "protein",
    }


def peptide_mapping(df_raw: pd.DataFrame):
    st.subheader("Peptide mapping")

    roles = [
        "Ignore",
        "Peptide Sequence",
        "Protein Group ID",
        "Protein / Peptide Name",
        "Species from this column",
        "Quantitative",
    ]

    numeric = set(df_raw.select_dtypes(include="number").columns.tolist())
    data = []
    for col in df_raw.columns:
        role = "Quantitative" if col in numeric else "Ignore"
        cl = col.lower()
        if "sequence" in cl:
            role = "Peptide Sequence"
        elif "pg." in cl or "protein group" in cl or "protein ids" in cl:
            role = "Protein Group ID"
        data.append({"column_name": col, "role": role})
    mapping_df = pd.DataFrame(data)

    edited = st.data_editor(
        mapping_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "column_name": st.column_config.TextColumn("Column name", disabled=True),
            "role": st.column_config.SelectboxColumn("Role", options=roles),
        },
        hide_index=True,
        key="peptide_mapping_editor",
    )

    pep_seq_cols = edited.loc[edited["role"] == "Peptide Sequence", "column_name"]
    if pep_seq_cols.empty:
        st.error("Please assign at least one 'Peptide Sequence' column.")
        return
    peptide_seq_col = pep_seq_cols.iloc[0]

    quant_cols = edited.loc[edited["role"] == "Quantitative", "column_name"].tolist()
    if not quant_cols:
        st.error("Please mark at least one column as 'Quantitative'.")
        return

    species_cols = edited.loc[edited["role"] == "Species from this column", "column_name"]
    name_cols = edited.loc[edited["role"].isin(["Protein / Peptide Name"]), "column_name"]
    protein_id_cols = edited.loc[edited["role"] == "Protein Group ID", "column_name"]

    df = df_raw.copy()
    # index: peptide sequence, or generated pepN_<proteinID>
    if peptide_seq_col in df.columns:
        df = df.set_index(peptide_seq_col)
    else:
        base = protein_id_cols.iloc[0] if not protein_id_cols.empty else None
        if base and base in df.columns:
            df = df.copy()
            df.index = [f"pep{i+1}_{pg}" for i, pg in enumerate(df[base].astype(str))]
        else:
            df = df.copy()
            df.index = [f"pep{i+1}" for i in range(len(df))]

    if not species_cols.empty:
        species = df[species_cols.iloc[0]].astype(str)
    else:
        name_col_for_species = name_cols.iloc[0] if not name_cols.empty else None
        species = infer_species_from_name(df, name_col_for_species, key_suffix="peptide")

    quant_df = df[quant_cols]

    st.markdown("**Preview transformed peptide table**")
    st.write(f"Peptides: {df.shape[0]:,}, Quant columns: {len(quant_cols)}")
    st.write("Species counts:")
    st.write(species.value_counts())

    preview = df.assign(Species=species)[quant_cols + ["Species"]].head(10)
    st.dataframe(preview, use_container_width=True)

    st.session_state["peptide_upload"] = {
        "df_raw": df_raw,
        "df": df,
        "quant_cols": quant_cols,
        "species": species,
        "peptide_index_source": peptide_seq_col,
        "mapping_table": edited,
        "level": "peptide",
    }


def main():
    st.title("Upload Protein & Peptide")

    st.markdown(
        "1. Upload a table.\n"
        "2. Use the 2‑column mapping table to assign roles.\n"
        "3. Check the preview of the transformed DataFrame."
    )

    level = st.radio(
        "Data level",
        options=["Protein", "Peptide"],
        horizontal=True,
    )

    uploaded = st.file_uploader(
        f"Upload {level.lower()} table (CSV/TSV/TXT)",
        type=["csv", "tsv", "txt"],
        key=f"{level.lower()}_uploader",
    )
    if not uploaded:
        st.info("Upload a file to start mapping.")
        return

    df_raw = read_table(uploaded.getvalue())
    st.write(f"Detected {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")

    if level == "Protein":
        protein_mapping(df_raw)
    else:
        peptide_mapping(df_raw)


if __name__ == "__main__":
    main()
