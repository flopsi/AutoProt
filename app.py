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
    st.title("Protein Header Mapping")

    uploaded = st.file_uploader(
        "Upload wide-format protein table (CSV/TSV/TXT)",
        type=["csv", "tsv", "txt"],
    )
    if not uploaded:
        st.info("Upload a file to see the header mapping table.")
        return

    df_raw = read_table(uploaded.getvalue())
    st.write(f"Table: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")

    # Detect numeric vs string
    numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = df_raw.columns.difference(numeric_cols).tolist()

    # Build a 2-column mapping dataframe: Header Information + Widget description
    rows = []
    for col in df_raw.columns:
        if col in non_numeric_cols:
            # PG.ProteinGroups / PG.ProteinNames get a special radio widget
            if col.lower().startswith("pg.proteingroups") or "proteingroup" in col.lower():
                widget_desc = "Radio: Protein Group / Species / Drop"
                dtype = "string"
            elif col.lower().startswith("pg.proteinnames") or "proteinname" in col.lower():
                widget_desc = "Radio: Protein Group / Species / Drop"
                dtype = "string"
            else:
                widget_desc = "Radio: Protein Group / Species / Drop"
                dtype = "string"
        else:
            widget_desc = "Text input to change name for downstream"
            dtype = "numeric"

        rows.append(
            {
                "Header Information": col,
                "Widget": widget_desc,
                "Data Type": dtype,
            }
        )

    mapping_table = pd.DataFrame(rows)

    st.subheader("Header mapping layout")
    st.dataframe(mapping_table, use_container_width=True)

    # ---- Actual interactive widgets, following the table spec ----
    st.subheader("Interactive configuration")

    # 1) Radio for PG.* and other string meta columns
    meta_roles = {}
    st.markdown("**Meta columns (Protein Group / Species / Drop)**")
    for col in non_numeric_cols:
        role = st.radio(
            f"{col}",
            options=["Protein Group", "Species Information", "Drop"],
            index=0 if "pg.proteingroups" in col.lower() else 1 if "pg.proteinnames" in col.lower() else 2,
        )
        meta_roles[col] = role

    # 2) Text inputs for numeric columns
    st.markdown("**Quantitative columns (rename for downstream)**")
    quant_rename = {}
    for col in numeric_cols:
        new_name = st.text_input(
            f"New name for `{col}`",
            value=col,
        )
        quant_rename[col] = new_name or col

    # Apply mapping
    df = df_raw.rename(columns=quant_rename)

    # Determine index for proteins (Protein Group)
    protein_group_cols = [c for c, r in meta_roles.items() if r == "Protein Group"]
    protein_group_col = protein_group_cols[0] if protein_group_cols else None
    if protein_group_col:
        df = df.set_index(protein_group_col)

    # Species assignment from any column marked as Species Information
    species_cols = [c for c, r in meta_roles.items() if r == "Species Information"]
    if species_cols:
        source_col = species_cols[0]
        # simple tag-based detection (e.g. HUMAN in A0A0B4J2E5_HUMAN)
        tags = st.multiselect(
            "Species tags to search for (e.g. HUMAN, ECOLI)",
            options=["HUMAN", "MOUSE", "RAT", "ECOLI", "YEAST", "BOVIN"],
            default=["HUMAN", "ECOLI", "YEAST"],
        )
        def detect(v):
            if pd.isna(v):
                return "Other"
            s = str(v).upper()
            for t in tags:
                if t in s:
                    return t
            return "Other"
        species = df_raw[source_col].apply(detect)
        species = species.reindex(df.index)
    else:
        species = pd.Series(["Unknown"] * len(df), index=df.index)

    # Quantitative columns are all renamed numeric columns
    quant_cols_renamed = [quant_rename[c] for c in numeric_cols]

    st.subheader("Preview transformed DataFrame")
    preview = df[quant_cols_renamed].copy()
    preview["Species"] = species
    st.dataframe(preview.head(10), use_container_width=True)

    # Store in session_state for downstream pages
    st.session_state["protein_upload"] = {
        "df_raw": df_raw,
        "df": df,
        "quant_cols": quant_cols_renamed,
        "meta_roles": meta_roles,
        "quant_rename": quant_rename,
        "species": species,
    }
    st.success("Protein data mapped and stored for downstream use.")


if __name__ == "__main__":
    main()
