import streamlit as st


headers = ["Radio Option", "Tags Field", "Text Edit Field"]
widgets = ["radio", "tags", "text"]

col1, col2 = st.columns(2)

for i, header in enumerate(headers):
    with col1:
        st.write(header)
    with col2:
        if widgets[i] == "radio":
            st.radio(f"Choose for {header}", ["Option 1", "Option 2"], key=f"radio_{i}")
        elif widgets[i] == "tags":
            st.text_input(f"Tags for {header}", key=f"tags_{i}")  # Streamlit does not have a native tags widget, so use text_input
        elif widgets[i] == "text":
            st.text_area(f"Edit text for {header}", key=f"text_{i}")
