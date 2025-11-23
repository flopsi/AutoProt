import streamlit as st
import pandas as pd
from typing import Dict, List

def render_condition_selector(column_names: List[str], auto_mapping: Dict[str, str]) -> Dict[str, str]:
    st.markdown("### Assign Conditions (A vs B)")
    st.info("Auto-detection applied. Modify assignments if needed.")
    
    df_edit = pd.DataFrame({
        'Original Column Name': column_names,
        'Condition': [auto_mapping[col][0] for col in column_names],
        'Replicate': [auto_mapping[col][1:] for col in column_names],
        'Renamed': [auto_mapping[col] for col in column_names]
    })
    
    edited_df = st.data_editor(
        df_edit,
        column_config={
            'Original Column Name': st.column_config.TextColumn('Original Column Name', disabled=True, width='large'),
            'Condition': st.column_config.SelectboxColumn('Condition', options=['A', 'B'], required=True, width='small'),
            'Replicate': st.column_config.TextColumn('Replicate', disabled=True, width='small'),
            'Renamed': st.column_config.TextColumn('Renamed', disabled=True, width='small')
        },
        hide_index=True,
        use_container_width=True
    )
    
    new_mapping = {}
    for idx, row in edited_df.iterrows():
        original = row['Original Column Name']
        condition = row['Condition']
        replicate = row['Replicate']
        new_mapping[original] = f"{condition}{replicate}"
    
    return new_mapping