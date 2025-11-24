# ============================================================
# 6. CV THRESHOLDS PER REPLICATE (3x2 PANEL)
# ============================================================

st.markdown("---")
st.markdown("### 6. Identification Quality by Sample")

all_samples = sorted(condition_mapping.items(), key=lambda x: x[1])

fig_cv_panel = make_subplots(
    rows=2, cols=3,
    subplot_titles=[condition_mapping[col] for col, _ in all_samples[:6]],
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

for idx, (col, condition) in enumerate(all_samples[:6]):
    row = idx // 3 + 1
    col_num = idx % 3 + 1
    
    if condition[0] == 'A':
        condition_data = a_data
        base_color = '#E71316'  # Red for A
    else:
        condition_data = b_data
        base_color = '#9BD3DD'  # Sky for B
    
    sample_data = quant_data[col].dropna()
    sample_indices = sample_data.index
    
    cv_data = condition_data.loc[sample_indices]
    cv_values = calculate_cv(cv_data)
    
    total_ids = len(sample_indices)
    cv_below_20 = (cv_values < 20).sum()
    cv_below_10 = (cv_values < 10).sum()
    
    # Add three separate bars with different opacities
    # Bar 1: Total IDs (full opacity)
    fig_cv_panel.add_trace(
        go.Bar(
            x=['Total IDs'],
            y=[total_ids],
            marker_color=base_color,
            text=[total_ids],
            textposition='outside',
            showlegend=False,
            opacity=1.0
        ),
        row=row, col=col_num
    )
    
    # Bar 2: CV<20% (0.8 opacity)
    fig_cv_panel.add_trace(
        go.Bar(
            x=['CV<20%'],
            y=[cv_below_20],
            marker_color=base_color,
            text=[cv_below_20],
            textposition='outside',
            showlegend=False,
            opacity=0.8
        ),
        row=row, col=col_num
    )
    
    # Bar 3: CV<10% (0.5 opacity)
    fig_cv_panel.add_trace(
        go.Bar(
            x=['CV<10%'],
            y=[cv_below_10],
            marker_color=base_color,
            text=[cv_below_10],
            textposition='outside',
            showlegend=False,
            opacity=0.5
        ),
        row=row, col=col_num
    )

fig_cv_panel.update_layout(
    title_text='Identification Count and CV% Quality Metrics',
    height=600,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
    showlegend=False
)

fig_cv_panel.update_xaxes(showgrid=False, tickangle=-45)
fig_cv_panel.update_yaxes(gridcolor='rgba(0,0,0,0.1)')

st.plotly_chart(fig_cv_panel, use_container_width=True)
