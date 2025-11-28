import streamlit as st
import pandas as pd

# Example DataFrame
df = pd.DataFrame(columns=["A", "B", "C"])

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
