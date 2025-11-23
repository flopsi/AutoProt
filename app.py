import streamlit as st
import pandas as pd

from config import (
    get_numeric_columns, get_metadata_columns, get_default_species_mapping_cols,
    get_default_group_col, get_default_peptide_id_col
)

st.set_page_config('Proteomics Multi-Uploader', layout='wide')

def upload_annotation_block(kind, help_txt, id_keys, col):
    with col:
        st.header(f"{kind.capitalize()}-level Upload")
        user_file = st.file_uploader(
            f"Upload {kind}-level file", 
            key=f"upl_{kind}", 
            type=["csv", "tsv", "txt"]
        )
        session_key = f"{kind}_upload"
        quant_cols_sel = []
        annot = []
        if user_file:
            sep = "\t" if user_file.name.endswith(('.tsv', '.txt')) else ','
            df = pd.read_csv(user_file, sep=sep)
            st.write("File loaded:", user_file.name)
            num_cols = get_numeric_columns(df)
            meta_cols = get_metadata_columns(df, num_cols)

            # Quantitative columns to keep (exclude any)
            quant_cols_sel = st.multiselect(
                "Select quantitative columns to keep:",
                num_cols, default=num_cols, key=f"quant_cols_{kind}"
            )

            mapping_options = get_default_species_mapping_cols(df) or meta_cols
            mapping_sel = st.selectbox(
                "Select species mapping column:",
                mapping_options, key=f"specmap_{kind}"
            )

            group_col_default = get_default_group_col(df)
            group_col_sel = st.selectbox(
                "Select protein group column:",
                ([group_col_default] if group_col_default else []) + meta_cols, key=f"groupcol_{kind}"
            )

            pept_id_sel = None
            if kind == "peptide":
                pept_id_default = get_default_peptide_id_col(df)
                pept_id_sel = st.selectbox(
                    "Select peptide/precursor identifier column:",
                    ([pept_id_default] if pept_id_default else []) + meta_cols, key="peptidcol_pept"
                )

            # Control/Treatment mode
            st.markdown("**Assign control/treatment conditions:**")
            mode = st.radio(
                "Assignment mode", ["Auto-split", "All Control", "All Treatment", "Manual"], 
                horizontal=True, key=f"mode_{kind}"
            )
            if mode == "Auto-split":
                annot = ['Control' if i < len(quant_cols_sel)//2 else 'Treatment' for i in range(len(quant_cols_sel))]
            elif mode == "All Control":
                annot = ['Control'] * len(quant_cols_sel)
            elif mode == "All Treatment":
                annot = ['Treatment'] * len(quant_cols_sel)
            else:
                annot = []
                for q in quant_cols_sel:
                    annot.append(st.selectbox(
                        f"{q}", options=["Control", "Treatment"], key=f"man_assign_{kind}_{q}"
                    ))

            result = {
                "data": df,
                "quant_cols": quant_cols_sel,
                "meta_cols": meta_cols,
                "species_col": mapping_sel,
                "group_col": group_col_sel,
                "condition": dict(zip(quant_cols_sel, annot))
            }
            if kind == "peptide":
                result["peptide_id_col"] = pept_id_sel

            st.session_state[session_key] = result
            st.success(f"{kind.capitalize()}-level data loaded and annotated.")

        else:
            st.info(help_txt)

colA, colB = st.columns(2)
upload_annotation_block(
    "protein",
    "Upload a protein-level quantification file. You may upload either or both levels below.",
    id_keys=["quant_cols_prot", "specmap_prot", "groupcol_prot", "mode_prot"],
    col=colA
)
upload_annotation_block(
    "peptide",
    "Upload a peptide-level quantification file (optional, for advanced stats). You may upload either or both levels below.",
    id_keys=["quant_cols_pept", "specmap_pept", "groupcol_pept", "peptidcol_pept", "mode_pept"],
    col=colB
)

st.divider()

# (Optional: show summary panel if either or both present)
if st.session_state.get("protein_upload") or st.session_state.get("peptide_upload"):
    st.markdown("### Data Upload Summary")
    prot_loaded = "✅" if st.session_state.get("protein_upload") else "—"
    pept_loaded = "✅" if st.session_state.get("peptide_upload") else "—"
    st.write(f"Protein-level loaded: {prot_loaded}")
    st.write(f"Peptide-level loaded: {pept_loaded}")

    # Show simple preview tables (head) for user's confirmation
    if st.session_state.get("protein_upload"):
        st.markdown("##### Protein-level preview")
        st.dataframe(st.session_state["protein_upload"]["data"].head(5), use_container_width=True)
    if st.session_state.get("peptide_upload"):
        st.markdown("##### Peptide-level preview")
        st.dataframe(st.session_state["peptide_upload"]["data"].head(5), use_container_width=True)
