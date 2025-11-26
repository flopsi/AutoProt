
import streamlit as st
pg = st.navigation([st.Page("page_2.py",title="Protein Import"), 
                    st.Page("page_3.py", title="Peptide Import"), 
                    st.Page("page_4.py", title="Data Quality")]
    )
st.set_page_config(page_title="autoProt", page_icon ="🧬")

pg.run()
