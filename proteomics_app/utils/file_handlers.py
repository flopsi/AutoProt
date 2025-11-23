import pandas as pd
from pathlib import Path
import streamlit as st

def load_data_file(file, delimiter: str = None) -> pd.DataFrame:
    try:
        if isinstance(file, str):
            file_path = Path(file)
            if file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file, sep=delimiter or '\t' if file_path.suffix == '.tsv' else ',')
        else:
            if file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            elif file.name.endswith('.tsv'):
                df = pd.read_csv(file, sep='\t')
            else:
                df = pd.read_csv(file)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None