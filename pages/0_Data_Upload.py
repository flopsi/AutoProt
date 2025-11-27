# pages/1_Import.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Import", layout="wide")
st.title("Proteomics Data Import — Professional Mode (Schessner et al., 2022)")

st.markdown("""
**Two import modes — your choice:**
- **Metadata-driven** — for complex, multi-condition experiments
- **Direct column selection** — fast, simple, no metadata needed
**Exactly as used at Max Planck Institute**
""")

# === UPLOAD ===
uploaded_prot = st.file_uploader("Upload Intensity Table (wide format)", type=["csv", "tsv", "txt"])

if not uploaded_prot:
    st.info("Please upload your intensity table.")
    st.stop()

@st.cache_data  # 👈 Add the caching decorator
def load(file):
    df = pd.read_csv(file, sep=None, engine="python", header=0,index_col=0)
    return df

data_df=load(uploaded_prot)

data_df=pd.DataFrame(
                    {
                        "widgets":["st.selectbox", "st.number_input", "st.text_area", "st.button"],
                    }
)

st.data_editor(
    data_df,
    column_config={
        "widgets": st.column_config.Column(
            "Streamlit Widgets",
            help="Streamlit **widget** commands 🎈",
            width="medium",
            required=True,
        )
    },
    hide_index=True,
    num_rows="dynamic",
)
st.dataframe(df_prot)

df_prot.head(5)

edited_df_prot = st.data_editor(df_prot)

