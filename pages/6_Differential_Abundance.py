# ================================================================
# VOLCANO PLOT - COLORED BY SPECIES
# ================================================================
st.markdown("### 🌋 Volcano Plot (Colored by Species)")

volcano_df = results[results['regulation'] != 'not_tested'].copy()

fig = px.scatter(
    volcano_df,
    x='log2fc',
    y='neg_log10_pval',
    color='species',  # COLOR BY SPECIES
    hover_data=['regulation'],
    title=f'Volcano Plot: {reference_group} vs {treatment_group}',
    labels={
        'log2fc': f'Log2 Fold Change ({reference_group} / {treatment_group})',
        'neg_log10_pval': '-Log10(FDR)' if use_fdr else '-Log10(p-value)',
        'species': 'Species'
    },
    height=600
)

# Add threshold lines
fig.add_hline(y=-np.log10(pval_threshold), line_dash="dash", line_color="gray", annotation_text=f"FDR = {pval_threshold}")
fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray")
fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray")

st.plotly_chart(fig, use_container_width=True)

# ================================================================
# MA PLOT - COLORED BY SPECIES
# ================================================================
st.markdown("### 📈 MA Plot (Colored by Species)")

ma_df = results[results['regulation'] != 'not_tested'].copy()
ma_df['A'] = (ma_df['mean_g1'] + ma_df['mean_g2']) / 2
ma_df['M'] = ma_df['log2fc']

fig_ma = px.scatter(
    ma_df,
    x='A',
    y='M',
    color='species',  # COLOR BY SPECIES
    hover_data=['regulation'],
    title='MA Plot: Mean vs Log2 Fold Change',
    labels={
        'A': 'Mean Log2 Intensity',
        'M': f'Log2 Fold Change ({reference_group} / {treatment_group})',
        'species': 'Species'
    },
    height=600
)

# Add threshold lines
fig_ma.add_hline(y=fc_threshold, line_dash="dash", line_color="gray")
fig_ma.add_hline(y=-fc_threshold, line_dash="dash", line_color="gray")
fig_ma.add_hline(y=0, line_dash="solid", line_color="red", opacity=0.3)

st.plotly_chart(fig_ma, use_container_width=True)
