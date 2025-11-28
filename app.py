import io
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Protein Header Mapping", layout="wide")


@st.cache_data(show_spinner="Loading file...")
def read_table(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")


def main():
    st.title("Protein Header Mapping (2‑column dynamic table)")

    uploaded = st.file_uploader(
        "Upload wide-format protein table (CSV/TSV/TXT)",
        type=["csv", "tsv", "txt"],
    )
    if not uploaded:
        st.info("Upload a file to start.")
        return

    df_raw = read_table(uploaded.getvalue())
    st.write(f"Table: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")

    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = df_raw.columns.difference(numeric_cols).tolist()

    # ---------- 1) Dynamic 2‑column mapping table ----------
    st.subheader("1. Assign roles (widgets in column 2)")

    role_options = ["Drop", "Protein Group", "Species Information", "Quantitative"]

    # Build initial guess
    rows = []
    for col in df_raw.columns:
        cl = col.lower()
        if col in numeric_cols:
            default_role = "Quantitative"
        elif "pg.proteingroups" in cl or "protein group" in cl or "protein ids" in cl:
            default_role = "Protein Group"
        elif "name" in cl:
            default_role = "Species Information"
        else:
            default_role = "Drop"
        rows.append({"Header": col, "Role": default_role})

    mapping_df = pd.DataFrame(rows)

    edited = st.data_editor(
        mapping_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Header": st.column_config.TextColumn("Header Information", disabled=True),
            "Role": st.column_config.SelectboxColumn(
                "Widget / Role",
                options=role_options,
                help="Choose how this column is used.",
            ),
        },
        hide_index=True,
        key="header_mapping_editor",
    )

    # ---------- 2) Validate unique Protein Group & Species ----------
    protein_group_cols = edited.loc[edited["Role"] == "Protein Group", "Header"].tolist()
    species_cols = edited.loc[edited["Role"] == "Species Information", "Header"].tolist()
    quant_cols = edited.loc[edited["Role"] == "Quantitative", "Header"].tolist()

    if len(protein_group_cols) != 1:
        st.error("Select **exactly one** column as 'Protein Group'.")
        return
    if len(species_cols) != 1:
        st.error("Select **exactly one** column as 'Species Information'.")
        return
    if not quant_cols:
        st.error("Mark at least one column as 'Quantitative'.")
        return

    protein_group_col = protein_group_cols[0]
    species_source_col = species_cols[0]

    # ---------- 3) Rename quantitative columns (second dynamic table) ----------
    st.subheader("2. Rename quantitative columns")

    ren_rows = [{"Original": c, "New name": c} for c in quant_cols]
    ren_df = pd.DataFrame(ren_rows)

    ren_edited = st.data_editor(
        ren_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Original": st.column_config.TextColumn("Original name", disabled=True),
            "New name": st.column_config.TextColumn(
                "New name for downstream", help="Change e.g. long.raw.PG.Quantity → A1"
            ),
        },
        hide_index=True,
        key="quant_rename_editor",
    )

    rename_map = {
        row["Original"]: (row["New name"] or row["Original"])
        for _, row in ren_edited.iterrows()
    }
    quant_cols_renamed = [rename_map[c] for c in quant_cols]

    # ---------- 4) Species mapping via tags on selected species column ----------
    st.subheader("3. Species mapping via tags")

    tags = st.multiselect(
        "Species tags to search for (e.g. HUMAN, ECOLI)",
        options=["HUMAN", "MOUSE", "RAT", "ECOLI", "YEAST", "BOVIN"],
        default=["HUMAN", "ECOLI", "YEAST"],
    )

    def detect_species(v):
        if pd.isna(v):
            return "Other"
        s = str(v).upper()
        for t in tags:
            if t in s:
                return t
        return "Other"

    species_series = df_raw[species_source_col].apply(detect_species)
    st.write("Species counts:")
    st.write(species_series.value_counts())

    # ---------- 5) Apply renaming + build final preview ----------
    df = df_raw.rename(columns=rename_map)
    df_indexed = df.set_index(protein_group_col)
    species_aligned = species_series.reindex(df_indexed.index)

    st.subheader("4. Preview transformed DataFrame")
    preview = df_indexed[quant_cols_renamed].copy()
    preview["Species"] = species_aligned
    st.dataframe(preview.head(10), use_container_width=True)

    # ---------- 6) Store for downstream use ----------
    st.session_state["protein_upload"] = {
        "df_raw": df_raw,
        "df": df_indexed,
        "mapping_table": edited,
        "quant_rename_table": ren_edited,
        "protein_group_col": protein_group_col,
        "species_source_col": species_source_col,
        "quant_cols": quant_cols_renamed,
        "species": species_aligned,
    }
    st.success("Protein data mapped and stored for downstream analysis.")


if __name__ == "__main__":
    main()
