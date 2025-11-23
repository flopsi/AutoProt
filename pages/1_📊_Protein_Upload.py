# ========== STAGE 4: SUMMARY & THREE-COLUMN CHARTS ==========
elif st.session_state.upload_stage == 'summary':
    protein_data = st.session_state.protein_data
    filename = st.session_state.protein_filename

    st.markdown("## Upload Summary")
    st.success("✓ Data processed successfully!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("File", filename.split('(')[0].strip())
    with col2:
        st.metric("Proteins", f"{protein_data.n_proteins:,}")
    with col3:
        n_a = len([c for c in protein_data.condition_mapping.values() if c.startswith('A')])
        n_b = len([c for c in protein_data.condition_mapping.values() if c.startswith('B')])
        st.metric("Samples", f"{n_a + n_b} ({n_a}A + {n_b}B)")
    with col4:
        species_counts = protein_data.get_species_counts()
        st.metric("Species", len([s for s in species_counts.values() if s > 0]))

    st.markdown("### Protein Distribution by Species")
    
    # Calculate counts for all three categories
    total_species_counts = protein_data.get_species_counts()

    a_data = protein_data.get_condition_data('A')
    a_detected_indices = a_data.dropna(how='all').index
    species_a_counts = {sp: sum(1 for idx in a_detected_indices if protein_data.species_map.get(idx) == sp)
                       for sp in ['human', 'ecoli', 'yeast']}

    b_data = protein_data.get_condition_data('B')
    b_detected_indices = b_data.dropna(how='all').index
    species_b_counts = {sp: sum(1 for idx in b_detected_indices if protein_data.species_map.get(idx) == sp)
                       for sp in ['human', 'ecoli', 'yeast']}

    # Create three-column layout with bordered containers
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        with st.container(border=True):
            fig1 = create_species_bar_chart(total_species_counts, "Total Proteins by Species")
            st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        with st.container(border=True):
            fig2 = create_species_bar_chart(species_a_counts, "Condition A Proteins")
            st.plotly_chart(fig2, use_container_width=True)

    with chart_col3:
        with st.container(border=True):
            fig3 = create_species_bar_chart(species_b_counts, "Condition B Proteins")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("### Next Steps")

    nav_col1, nav_col2 = st.columns(2)

    with nav_col1:
        if st.button("📊 Upload Peptide Data", type="primary", use_container_width=True):
            st.switch_page("pages/2_🔬_Peptide_Upload.py")

    with nav_col2:
        if st.button("✓ Go to Data Quality", type="secondary", use_container_width=True):
            st.switch_page("pages/3_✓_Data_Quality.py")

# Reset button (always visible)
st.markdown("---")
if st.button("🔄 Start Over", use_container_width=True):
    for key in ['upload_stage', 'protein_df', 'auto_species_col', 'auto_condition_mapping',
                'selected_species_col', 'selected_workflow', 'selected_quant_cols']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.upload_stage = 'upload'
    st.rerun()
