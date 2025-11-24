# ============================================================
# 2. MISSING VALUE HEATMAP (Condition-based coloring)
# ============================================================

st.markdown("---")
st.markdown("### 2. Missing Value Pattern")

# Create binary matrix (1 = present, 0 = absent)
binary_matrix = (~quant_data.isna()).astype(float)

# Replace 0 with NaN for proper heatmap visualization
# 1 stays as 1, 0 becomes NaN (will show as black/white)
heatmap_matrix = binary_matrix.replace(0, np.nan)

# Prepare data by condition
z_data = []
y_labels = []
condition_colors = []

for col in quant_data.columns:
    condition = condition_mapping.get(col, col)
    condition_letter = condition[0]
    
    y_labels.append(condition)
    z_data.append(heatmap_matrix[col].values)
    condition_colors.append(condition_letter)

z_data = np.array(z_data)

# Create custom colorscale:
# NaN/missing = black
# Present for A = red
# Present for B = sky blue
# We'll create two separate heatmaps overlaid

fig_heatmap = go.Figure()

# Add heatmap for Condition A samples
a_indices = [i for i, c in enumerate(condition_colors) if c == 'A']
if len(a_indices) > 0:
    fig_heatmap.add_trace(go.Heatmap(
        z=z_data[a_indices],
        y=[y_labels[i] for i in a_indices],
        x=list(range(z_data.shape[1])),
        colorscale=[
            [0, 'black'],      # Missing
            [0.5, 'black'],    # Missing
            [0.51, '#E71316'], # Present (Red)
            [1, '#E71316']     # Present (Red)
        ],
        showscale=False,
        zmin=0,
        zmax=1,
        hovertemplate='Sample: %{y}<br>Protein: %{x}<br>Present: %{z}<extra></extra>',
        name='Condition A'
    ))

# Add heatmap for Condition B samples
b_indices = [i for i, c in enumerate(condition_colors) if c == 'B']
if len(b_indices) > 0:
    fig_heatmap.add_trace(go.Heatmap(
        z=z_data[b_indices],
        y=[y_labels[i] for i in b_indices],
        x=list(range(z_data.shape[1])),
        colorscale=[
            [0, 'black'],      # Missing
            [0.5, 'black'],    # Missing
            [0.51, '#9BD3DD'], # Present (Sky)
            [1, '#9BD3DD']     # Present (Sky)
        ],
        showscale=False,
        zmin=0,
        zmax=1,
        hovertemplate='Sample: %{y}<br>Protein: %{x}<br>Present: %{z}<extra></extra>',
        name='Condition B'
    ))

fig_heatmap.update_layout(
    title='Data Completeness (Red=A, Sky=B, Black=Missing)',
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
    xaxis=dict(title=f'{data_type} Index', showgrid=False),
    yaxis=dict(title='Sample', showgrid=False, tickangle=0),
    showlegend=False
)

st.plotly_chart(fig_heatmap, use_container_width=True)
