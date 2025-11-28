import streamlit as st
import pandas as pd


@st.cache_data
def load_csv(file):
    return pd.read_csv(file, sep= , engine="python")  # no fixed index/headers

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.stop()

df_raw = load_csv(uploaded)
st.dataframe(df_raw.head())

row0 = df_raw.iloc[0]                 # first row (a Series)
row0_df = row0.to_frame(name="value") # index = column names, column = value
row0_df.index.name = "column"
row0_df.reset_index(inplace=True)     # columns: ["column", "value"]

edited = st.data_editor(
    row0_df,
    column_config={
        "use": st.column_config.CheckboxColumn("Use?", default=True),
        "is_condition": st.column_config.CheckboxColumn("Is condition?", default=False),
        "is_replicate": st.column_config.CheckboxColumn("Is replicate?", default=False),
        "tags": st.column_config.ListColumn("Tags"),  # simple tag-like list of strings
    },
    num_rows="fixed",
)

# Columns user wants to use
selected_cols = edited.loc[edited["use"], "column"].tolist()

# Condition and replicate columns (flexible naming)
cond_cols = edited.loc[edited["is_condition"], "column"].tolist()
rep_cols  = edited.loc[edited["is_replicate"], "column"].tolist()

# Tags per column (list of strings)
tag_map = dict(zip(edited["column"], edited["tags"]))



st.write("Condition 1 DF")
st.dataframe(df_cond1)

st.write("Condition 2 DF")
st.dataframe(df_cond2)

