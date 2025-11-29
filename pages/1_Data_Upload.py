import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

# ---------- MSData model ----------

@dataclass
class MSData:
    original: pd.DataFrame
    filled: pd.DataFrame
    log2_filled: pd.DataFrame
    numeric_cols: List[str]
    ones_count: int


# ---------- cached builder ----------

@st.cache_data
def get_msdata(df: pd.DataFrame, quant_cols: List[str]) -> MSData:
    """Build and cache MS data object from raw upload + selected quant columns."""
    return build_msdata(df, quant_cols)


def build_msdata(processed_df: pd.DataFrame, numeric_cols_renamed: List[str]) -> MSData:
    original = processed_df.copy()

    # 1) quant columns → float64 (non‑numeric → NaN)
    original[numeric_cols_renamed] = (
        original[numeric_cols_renamed]
        .apply(pd.to_numeric, errors="coerce")  # strings → NaN [web:69]
        .astype("float64")                      # consistent numeric dtype [web:61]
    )

    # 2) filled: NaN, 0, 1 → 1
    filled = original.copy()
    vals = filled[numeric_cols_renamed]
    vals = vals.fillna(1.0)
    vals = vals.where(~vals.isin([0.0, 1.0]), 1.0)
    filled[numeric_cols_renamed] = vals

    ones_count = (filled[numeric_cols_renamed] == 1.0).to_numpy().sum()

    # 3) log2(filled)
    log2_filled = filled.copy()
    log2_filled[numeric_cols_renamed] = np.log2(log2_filled[numeric_cols_renamed])

    return MSData(
        original=original,
        filled=filled,
        log2_filled=log2_filled,
        numeric_cols=numeric_cols_renamed,
        ones_count=ones_count,
    )


# ---------- session state defaults ----------

DEFAULTS = {
    "raw_df": None,
    "selected_quant_cols": [],
    "column_renames": {},
    "upload_key": 0,
    "protein_model": None,   # your MSData object
}


def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------- layout / page ----------

def main():
    st.set_page_config(page_title="Data Upload", layout="wide")
    init_state()

    st.title("Data Upload")

    # file upload
    uploaded = st.file_uploader(
        "Drop CSV / TSV file", 
        type=["csv", "tsv", "txt"], 
        key=f"uploader_{st.session_state.upload_key}",
    )

    if uploaded is None:
        st.info("Upload a file to begin.")
        return

    # detect sep and read
    sep = "\t" if uploaded.name.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(uploaded, sep=sep)
    st.session_state.raw_df = df

    st.write("Shape:", df.shape)
    st.dataframe(df.head(), use_container_width=True)

    # simple numeric column detection
    numeric_cols = []
    for col in df.columns:
        numeric_values = pd.to_numeric(df[col], errors="coerce")
        if not numeric_values.isna().all():
            numeric_cols.append(col)

    st.subheader("Select quantitative columns")
    quant_cols = st.multiselect(
        "Quantitative columns",
        options=numeric_cols,
        default=numeric_cols,
        key="selected_quant_cols",
    )

    # build MSData when user confirms
    if quant_cols and st.button("Process data"):
        msdata = get_msdata(df, quant_cols)
        st.session_state.protein_model = msdata

        st.success("MS data processed.")
        st.write("Quant dtypes:", msdata.original[quant_cols].dtypes)
        st.metric("Cells equal to 1 after filling", msdata.ones_count)

        st.subheader("Log2-filled preview")
        st.dataframe(msdata.log2_filled[quant_cols].head(), use_container_width=True)


if __name__ == "__main__":
    main()
