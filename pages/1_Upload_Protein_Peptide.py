import io
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(show_spinner="Loading file...")
def read_table(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")


def protein_mapping_ui():
    if "protein_bytes" not in st.session_state:
        st.stop()

    df_raw = read_table(st.session_state["protein_bytes"])

    st.subheader("1. Column mapping")
    st.write(f"Table: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")

    # Build 2-column mapping table
    roles = [
        "Ignore",
        "Protein Group ID",
        "Protein Name / Description",
        "Species from this column",
        "Quantitative",
    ]
    # Heuristic defaults
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

    # Validate roles
    prot_id_cols = edited.loc[edited["role"] == "Protein Group ID", "column_name"]
    if prot_id_cols.empty:
        st.error("Please assign exactly one 'Protein Group ID' column.")
        st.stop()
    if len(prot_id_cols) > 1:
        st.error("Multiple columns marked as 'Protein Group ID'. Choose only one.")
        st.stop()
    protein_id_col = prot_id_cols.iloc[0]

    quant_cols = edited.loc[edited["role"] == "Quantitative", "column_name"].tolist()
    if not quant_cols:
        st.error("Please mark at least one column as 'Quantitative'.")
        st.stop()

    species_cols = edited.loc[edited["role"] == "Species from this column", "column_name"]
    name_cols = edited.loc[edited["role"] == "Protein Name / Description", "column_name"]

    # Build transformed frame
    df = df_raw.copy().set_index(protein_id_col)

    # Species assignment
    if not species_cols.empty:
        species = df[species_cols.iloc[0]].astype(str)
    else:
        # fallback to predefined tags in name column(s)
        name_col_for_species = name_cols.iloc[0] if not name_cols.empty else None
        species = infer_species_from_name(df, name_col_for_species)

    # Quant matrix (keep original names for now; could also rename)
    quant_df = df[quant_cols]

    st.subheader("2. Preview transformed protein table")
    st.write(f"Proteins: {df.shape[0]:,}, Quant columns: {len(quant_cols)}")
    st.write("Species counts:")
    st.write(species.value_counts())

    preview = df.assign(Species=species)[quant_cols + ["Species"]].head(10)
    st.dataframe(preview, use_container_width=True)

    # Store bundle for downstream use
    st.session_state["protein_upload"] = {
        "df_raw": df_raw,
        "df": df,                     # indexed by Protein Group ID
        "quant_cols": quant_cols,
        "species": species,
        "protein_id_col": protein_id_col,
        "mapping_table": edited,
    }


def infer_species_from_name(df: pd.DataFrame, name_col: str | None) -> pd.Series:
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
        key=f"species_tags_{name_col}",
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
