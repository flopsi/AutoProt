# shared.py
import streamlit as st
from datetime import datetime

def restart_button():
    st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)  # spacer
    st.markdown(
        """
        <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 9999; width: 340px;">
        """,
        unsafe_allow_html=True
    )
    if st.button("Restart Full Analysis", type="primary", use_container_width=True, key="global_restart"):
        keys = [k for k in st.session_state.keys() if k.startswith(("prot_", "pept_", "reconfig_"))]
        for k in keys:
            del st.session_state[k]
        st.success("All data cleared!")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Universal logger
def debug(msg, data=None):
    if st.session_state.get("DEBUG", False):
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.setdefault("debug_log", []).append(f"[{ts}] {msg}")
        if data is not None:
            with st.expander(f"Details → {msg}"):
                st.code(data)
