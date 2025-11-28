import streamlit as st
import pandas as pd

# Example DataFrame
# pages/1_Upload_Protein.py
import io
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Loading file...")
def read_table(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if txt.startswith("\ufeff"):
        txt = txt[1:]
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")

df = read_table()
# List of widgets to display for each header
widgets = ["radio", "tags", "text"]

col1, col2 = st.columns(2)

for i, header in enumerate(df.columns):
    with col1:
        st.write(header)
    with col2:
        if widgets[i] == "radio":
            st.radio(f"Select for {header}", ["Option 1", "Option 2"], key=f"radio_{i}")
        elif widgets[i] == "tags":
            st.text_input(f"Tags for {header}", key=f"tags_{i}")  # No native tags widget, use text_input
        elif widgets[i] == "text":
            st.text_area(f"Edit text for {header}", key=f"text_{i}")
