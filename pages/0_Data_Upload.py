import streamlit as st
import pandas as pd


@st.cache_data
def load_csv(file):
    return pd.read_csv(file, sep=None, engine="python")  # no fixed index/headers

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.stop()

df_raw = load_csv(uploaded)
st.dataframe(df_raw.head())
cols = df_raw.columns.tolist()

df_raw = load_csv(uploaded)
st.dataframe(df_raw.head())


meta_cols = st.multiselect(
    "Select metadata columns",
    options=df_raw.columns.tolist(),
)

cond1_col = st.selectbox("Select column for condition 1", df_raw.columns)
cond2_col = st.selectbox("Select column for condition 2", df_raw.columns)

cond1_value = st.selectbox("Value for condition 1", df_raw[cond1_col].unique())
cond2_value = st.selectbox("Value for condition 2", df_raw[cond2_col].unique())


@st.cache_data
def get_metadata_df(df, meta_cols):
    return df[meta_cols].copy()

@st.cache_data
def get_condition_df(df, cond_col, cond_value):
    return df[df[cond_col] == cond_value].copy()


if meta_cols:
    df_meta = get_metadata_df(df_raw, tuple(meta_cols))  # lists must be hashable -> tuple
    st.write("Metadata DF")
    st.dataframe(df_meta)

df_cond1 = get_condition_df(df_raw, cond1_col, cond1_value)
df_cond2 = get_condition_df(df_raw, cond2_col, cond2_value)

st.write("Condition 1 DF")
st.dataframe(df_cond1)

st.write("Condition 2 DF")
st.dataframe(df_cond2)

