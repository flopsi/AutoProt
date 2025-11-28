import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Cached file upload", layout="centered")

st.title("Upload & cache data")

@st.cache_data
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded file and cache the result."""
    # Decide parser based on extension
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    elif filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xls", "xlsx"]
)

if uploaded_file is not None:
    # Read bytes once; bytes + filename form the cache key
    file_bytes = uploaded_file.getvalue()
    try:
        df = load_data(file_bytes, uploaded_file.name)
        st.success(f"Loaded {uploaded_file.name}")
        st.write(df.head())
        st.caption("Data is cached; re-running the app won't re-parse the file until you upload a different one.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a file to begin.")
