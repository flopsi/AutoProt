# shared.py
import streamlit as st
from datetime import datetime

def restart_button():
    st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:9999; width:340px;">
    """, unsafe_allow_html=True)
    if st.button("Restart Full Analysis", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith(("prot_", "pept_", "reconfig_", "debug_log")):
                del st.session_state[key]
        st.success("All data cleared!")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# FIXED DEBUG LOGGER — SAFE FOR OLD & NEW LOGS
def debug(msg, data=None):
    if st.session_state.get("DEBUG", False):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"<small style='color:#666; font-family:monospace;'>[{ts}] {msg}</small>"
        # Store as tuple if data exists, otherwise just the string
        entry = (line, data) if data is not None else line
        st.session_state.setdefault("debug_log", []).append(entry)
