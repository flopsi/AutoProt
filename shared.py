# shared.py
import streamlit as st

def restart_button():
    st.markdown('<div style="height:120px;"></div>', unsafe_allow_html=True)
    if st.button("Restart Full Analysis", type="primary", use_container_width=True):
        # Clear ONLY our app data
        keys = [k for k in st.session_state if k.startswith(("prot_", "pept_", "uploaded_"))]
        for k in keys:
            del st.session_state[k]
        st.cache_data.clear()
        st.success("All data cleared — start fresh")
        st.rerun()
