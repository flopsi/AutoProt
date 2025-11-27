# shared.py
import streamlit as st

def restart_button():
    if st.button("Restart Everything"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("App restarted! Upload new files.")
        st.rerun()
