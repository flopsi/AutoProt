import streamlit as st
import pandas as pd
from config import (
    get_numeric_columns, get_metadata_columns,
    get_default_species_mapping_cols,
    get_default_group_col, get_default_peptide_id_col
)

st.set_page_config('Proteomics Multi-Uploader', layout='wide')

colA, colB = st.columns(2)

with colA:
    st.header("Protein-level Upload")
    protein_file = st.file_uploader("Upload protein-level file", key="upl_protein", type=["csv", "tsv", "txt"])
    if protein_file:
        sep = "\t" if protein_file.name.endswith(('.tsv', '.txt')) else ','
        df_prot = pd.read_csv(protein_file, sep=sep)
        st.write("File loaded:", protein_file.name)
        num_cols = get_numeric_columns(df_prot)
        meta_cols = get_metadata_columns(df_prot, num_cols)

        st.markdown("**Select quantitative columns to keep:**")
        quant_cols_sel = st.multiselect("Quant columns", num_cols, default=num_cols, key="quant_cols_prot")
        st.markdown("**Select species mapping column:**")
        mapping_sel = st.selectbox("Species column", get_default_species_mapping_cols(df_prot) or meta_cols, key="specmap_prot")
        st.markdown("**Select protein group column:**")
        group_col_sel = st.selectbox("Protein group", [get_default_group_col(df_prot)] + meta_cols, key="groupcol_prot")
        st.write("")

        st.markdown("**Assign Control/Treatment**")
        auto_split = st.radio("Assignment mode", ["Auto-split", "All Control", "All Treatment", "Manual"], horizontal=True, key="mode_prot")
        if auto_split == "Auto-split":
            annot = ['Control' if i < len(quant_cols_sel)//2 else 'Treatment' for i in range(len(quant_cols_sel))]
        elif auto_split == "All Control":
            annot = ['Control' for _ in quant_cols_sel]
        elif auto_split == "All Treatment":
            annot = ['Treatment' for _ in quant_cols_sel]
        else:
            annot = []
            for q in quant_cols_sel:
                annot.append(st.selectbox(f"{q}", options=["Control", "Treatment"], key=f"man_assign_prot_{q}"))

        st.session_state["protein_upload"] = {
            "data": df_prot,
            "quant_cols": quant_cols_sel,
            "meta_cols": meta_cols,
            "species_col": mapping_sel,
            "group_col": group_col_sel,
            "condition": dict(zip(quant_cols_sel, annot))
        }
        st.success("Protein-level data loaded and annotated.")

with colB:
    st.header("Peptide-level Upload")
    peptide_file = st.file_uploader("Upload peptide-level file", key="upl_peptide", type=["csv", "tsv", "txt"])
    if peptide_file:
        sep = "\t" if peptide_file.name.endswith(('.tsv', '.txt')) else ','
        df_pept = pd.read_csv(peptide_file, sep=sep)
        st.write("File loaded:", peptide_file.name)
        num_cols = get_numeric_columns(df_pept)
        meta_cols = get_metadata_columns(df_pept, num_cols)

        st.markdown("**Select quantitative columns to keep:**")
        quant_cols_sel = st.multiselect("Quant columns", num_cols, default=num_cols, key="quant_cols_pept")
        st.markdown("**Select species mapping column:**")
        mapping_sel = st.selectbox("Species column", get_default_species_mapping_cols(df_pept) or meta_cols, key="specmap_pept")
        st.markdown("**Select protein group column:**")
        group_col_sel = st.selectbox("Protein group", [get_default_group_col(df_pept)] + meta_cols, key="groupcol_pept")
        st.markdown("**Select peptide identifier column:**")
        pept_id_sel = st.selectbox("Peptide/Precursor ID", [get_default_peptide_id_col(df_pept)] + meta_cols, key="peptidcol_pept")
        st.write("")

        st.markdown("**Assign Control/Treatment**")
        auto_split = st.radio("Assignment mode", ["Auto-split", "All Control", "All Treatment", "Manual"], horizontal=True, key="mode_pept")
        if auto_split == "Auto-split":
            annot = ['Control' if i < len(quant_cols_sel)//2 else 'Treatment' for i in range(len(quant_cols_sel))]
        elif auto_split == "All Control":
            annot = ['Control' for _ in quant_cols_sel]
        elif auto_split == "All Treatment":
            annot = ['Treatment' for _ in quant_cols_sel]
        else:
            annot = []
            for q in quant_cols_sel:
                annot.append(st.selectbox(f"{q}", options=["Control", "Treatment"], key=f"man_assign_pept_{q}"))

        st.session_state["peptide_upload"] = {
            "data": df_pept,
            "quant_cols": quant_cols_sel,
            "meta_cols": meta_cols,
            "species_col": mapping_sel,
            "group_col": group_col_sel,
            "peptide_id_col": pept_id_sel,
            "condition": dict(zip(quant_cols_sel, annot))
        }
        st.success("Peptide-level data loaded and annotated.")
