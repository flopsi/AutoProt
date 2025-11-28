import streamlit as st
import pandas as pd


@st.cache_data
def load_csv(file):
    return pd.read_csv(file, sep=None , engine="python")  # no fixed index/headers

uploaded = st.file_uploader("Upload CSV", type=["csv"])


if uploaded is None:
    st.stop()
row0 = df_raw.iloc[0]
row0_df = row0.to_frame(name="value")
row0_df.index.name = "column"
row0_df.reset_index(inplace=True)  # columns: ["column", "value"]

# Add helper columns so they exist in edited
row0_df["use"] = True
row0_df["is_condition"] = False
row0_df["is_replicate"] = False
row0_df["tags"] = [[]] * len(row0_df)  # one empty list per row

edited = st.data_editor(
    row0_df,
    column_config={
        "use": st.column_config.CheckboxColumn("Use?", default=True),
        "is_condition": st.column_config.CheckboxColumn("Is condition?", default=False),
        "is_replicate": st.column_config.CheckboxColumn("Is replicate?", default=False),
        "tags": st.column_config.ListColumn("Tags"),
    },
    num_rows="fixed",
)

df_raw = load_csv(uploaded)
st.dataframe(df_raw.head())



