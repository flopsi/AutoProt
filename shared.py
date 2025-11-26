# shared.py
# shared.py
import streamlit as st

def clear_all_data():
    keys = [k for k in st.session_state.keys() if k.startswith(("prot_", "pept_", "reconfig_"))]
    for k in keys:
        del st.session_state[k]
    st.success("All data cleared – start fresh!")
    st.rerun()

def restart_button():
    st.markdown('<div class="fixed-restart">', unsafe_allow_html=True)
    if st.button("Restart Full Analysis", type="primary", use_container_width=True, key="global_restart"):
        clear_all_data()
    st.markdown('</div>', unsafe_allow_html=True)
