# shared.py
import streamlit as st

def clear_all_data():
    """Call this only from the big red Restart button"""
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith(("prot_", "pept_"))]
    for k in keys_to_remove:
        del st.session_state[k]
    st.success("All data cleared. Ready for new analysis.")
    st.rerun()

def restart_button():
    """Fixed bottom restart button — use on every page"""
    st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
    if st.button("Restart Full Analysis", type="primary", use_container_width=True, key="global_restart"):
        clear_all_data()
    st.markdown('</div>', unsafe_allow_html=True)
