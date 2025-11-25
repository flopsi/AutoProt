import streamlit as st
import pandas as pd
from utils.analysis import calculate_stats
def render_stats_cards(df: pd.DataFrame):
    """    Render statistics cards showing protein counts    Args:        df: DataFrame with processed protein data    """    stats = calculate_stats(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""        <div class='stat-card'>            <div style='color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>                Total Proteins            </div>            <div style='color: #1e293b; font-size: 2rem; font-weight: bold; margin-top: 0.5rem;'>                {stats['total']:,}            </div>        </div>        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""        <div class='stat-card'>            <div style='color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>                Upregulated            </div>            <div style='color: #dc2626; font-size: 2rem; font-weight: bold; margin-top: 0.5rem;'>                {stats['up']:,}            </div>            <div style='color: #dc2626; font-size: 0.75rem; margin-top: 0.25rem;'>                {(stats['up']/stats['total']*100):.1f}% of total            </div>        </div>        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""        <div class='stat-card'>            <div style='color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>                Downregulated            </div>            <div style='color: #2563eb; font-size: 2rem; font-weight: bold; margin-top: 0.5rem;'>                {stats['down']:,}            </div>            <div style='color: #2563eb; font-size: 0.75rem; margin-top: 0.25rem;'>                {(stats['down']/stats['total']*100):.1f}% of total            </div>        </div>        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""        <div class='stat-card'>            <div style='color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>                Significant            </div>            <div style='color: #059669; font-size: 2rem; font-weight: bold; margin-top: 0.5rem;'>                {stats['significant']:,}            </div>            <div style='color: #059669; font-size: 0.75rem; margin-top: 0.25rem;'>                {(stats['significant']/stats['total']*100):.1f}% of total            </div>        </div>        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
