
import streamlit as st
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
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
