# shared.py  ← REPLACE ENTIRE FILE WITH THIS
import streamlit as st
from datetime import datetime

def restart_button():
    """Completely wipes EVERYTHING: session state + all @st.cache_data"""
    st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:9999; width:340px;">
    """, unsafe_allow_html=True)

    if st.button("Restart Full Analysis", type="primary", use_container_width=True, key="master_restart"):
        # 1. Clear session state
        keys_to_kill = [k for k in st.session_state.keys() if k.startswith(("prot_", "pept_", "reconfig_", "debug_log"))]
        for k in keys_to_kill:
            del st.session_state[k]

        # 2. Clear ALL cached functions (this is the magic)
        st.cache_data.clear()           # ← KILLS ALL @st.cache_data
        st.cache_resource.clear()       # ← in case you ever use it

        # 3. Show message and force full rerun
        st.success("Everything cleared! Starting fresh...")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# Debug logger (unchanged, but safe)
def debug(msg, data=None):
    if st.session_state.get("DEBUG", False):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"<small style='color:#666; font-family:monospace;'>[{ts}] {msg}</small>"
        entry = (line, data) if data is not None else line
        st.session_state.setdefault("debug_log", []).append(entry)
