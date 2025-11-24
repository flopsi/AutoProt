# ============================================================
# 1. PROTEIN RANK PLOT (A and B side-by-side)
# ============================================================

st.markdown("---")
st.markdown("### 1. Protein Rank Plot")

# Calculate mean intensities for Condition A and B
a_data = current_data.get_condition_data('A')
b_data = current_data.get_condition_data('B')

mean_a = a_data.mean(axis=1).sort_values(ascending=False).reset_index(drop=True)
mean_b = b_data.mean(axis=1).sort_values(ascending=False).reset_index(drop=True)

log10_a = np.log10(mean_a[mean_a > 0])
log10_b = np.log10(mean_b[mean_b > 0])

fig_rank = go.Figure()

# Condition A
fig_rank.add_trace(go.Scatter(
    x=list(range(1, len(log10_a) + 1)),
    y=log10_a,
    mode='lines',
    line=dict(color='#E71316', width=2),
    name='Condition A',
    hovertemplate='A - Rank: %{x}<br>Log₁₀ Intensity: %{y:.2f}<extra></extra>'
))

# Condition B
fig_rank.add_trace(go.Scatter(
    x=list(range(1, len(log10_b) + 1)),
    y=log10_b,
    mode='lines',
    line=dict(color='#9BD3DD', width=2),
    name='Condition B',
    hovertemplate='B - Rank: %{x}<br>Log₁₀ Intensity: %{y:.2f}<extra></extra>'
))

fig_rank.update_layout(
    title=f'{data_type} Rank Plot (A vs B)',
    xaxis_title='Protein Rank',
    yaxis_title='Log₁₀ Mean Intensity',
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
    xaxis=dict(type='log', gridcolor='rgba(0,0,0,0.1)'),
    yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
)

st.plotly_chart(fig_rank, use_container_width=True)

# ============================================================
# 2. MISSING VALUE HEATMAP (Colored by Condition)
# ============================================================

st.markdown("---")
st.markdown("### 2. Missing Value Pattern")

# Create binary matrix
binary_matrix = (~quant_data.isna()).astype(int)

# Prepare data for heatmap with condition colors
z_data = []
y_labels = []
colors_list = []

for col in quant_data.columns:
    condition = condition_mapping.get(col, col)
    condition_letter = condition[0]
    
    # Get presence/absence values
    col_values = binary_matrix[col].values
    z_data.append(col_values)
    y_labels.append(condition)
    
    # Assign color based on condition
    colors_list.append('#E71316' if condition_letter == 'A' else '#9BD3DD')

# Create heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=z_data,
    y=y_labels,
    x=list(range(len(binary_matrix))),
    colorscale=[
        [0, 'white'],  # Missing = white
        [1, '#E71316']  # Present = will be overridden per trace
    ],
    showscale=False,
    hovertemplate='Sample: %{y}<br>Protein: %{x}<br>Present: %{z}<extra></extra>'
))

# Create custom colorscale per row
for idx, (row_data, color) in enumerate(zip(z_data, colors_list)):
    # Create mask for present values
    present_mask = np.array(row_data) == 1
    
    # Add scatter trace for coloring
    fig_heatmap.add_trace(go.Scatter(
        x=np.where(present_mask)[0],
        y=[y_labels[idx]] * sum(present_mask),
        mode='markers',
        marker=dict(color=color, size=8, symbol='square'),
        showlegend=False,
        hoverinfo='skip'
    ))

fig_heatmap.update_layout(
    title='Data Completeness Pattern',
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
    xaxis=dict(title=f'{data_type} Index', showgrid=False),
    yaxis=dict(title='Sample', showgrid=False, tickangle=0)
)

st.plotly_chart(fig_heatmap, use_container_width=True)
