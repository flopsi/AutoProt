import streamlit as st
import pandas as pd

def render_data_table(df: pd.DataFrame):
    """
    Render an interactive data table with significant proteins

    Args:
        df: DataFrame with processed protein data
    """
    st.markdown("##### 📋 Significant Proteins")

    # Filter to significant only
    significant = df[df['significance'] != 'NS'].copy()

    if len(significant) == 0:
        st.info("No significant proteins with current thresholds")
        return

    # Sort by p-value
    significant = significant.sort_values('negLog10PValue', ascending=False)

    # Display count
    st.markdown(f"**{len(significant)} proteins** meet significance criteria")

    # Create display dataframe
    display_df = significant[['gene', 'log2FoldChange', 'negLog10PValue', 'significance']].copy()
    display_df.columns = ['Gene', 'Log2FC', '-Log10P', 'Change']

    # Format numbers
    display_df['Log2FC'] = display_df['Log2FC'].apply(lambda x: f'{x:.3f}')
    display_df['-Log10P'] = display_df['-Log10P'].apply(lambda x: f'{x:.3f}')

    # Style the dataframe
    def color_significance(val):
        if val == 'UP':
            return 'background-color: #fee2e2; color: #991b1b'
        elif val == 'DOWN':
            return 'background-color: #dbeafe; color: #1e3a8a'
        return ''

    styled_df = display_df.head(20).style.applymap(
        color_significance,
        subset=['Change']
    )

    # Display with custom height
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

    # Show selection in session state
    selected_indices = st.multiselect(
        "Select a protein to view details",
        options=significant.index.tolist(),
        format_func=lambda x: significant.loc[x, 'gene'],
        max_selections=1,
        key="protein_selector"
    )

    if selected_indices:
        st.session_state.selected_protein = significant.loc[selected_indices[0]].to_dict()
