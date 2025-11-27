# === 2. INTENSITY DISTRIBUTION — EXACTLY LIKE SCHESSNER ET AL., 2022 FIGURE 5A ===
st.subheader("Intensity Distribution (log₂ transformed)")

fig = go.Figure()

# Condition A: left-facing violins
for rep in c1:
    fig.add_trace(go.Violin(
        x=[rep] * len(intensity_final),
        y=intensity_final[rep],
        name=rep,
        side='negative',
        line_color='#E71316',
        meanline_visible=True,
        points=False,
        showlegend=False
    ))

# Condition B: right-facing violins
for rep in c2:
    fig.add_trace(go.Violin(
        x=[rep] * len(intensity_final),
        y=intensity_final[rep],
        name=rep,
        side='positive',
        line_color='#1f77b4',
        meanline_visible=True,
        points=False,
        showlegend=False
    ))

# Overlay boxplots
for rep in all_reps:
    color = '#E71316' if rep in c1 else '#1f77b4'
    fig.add_trace(go.Box(
        y=intensity_final[rep],
        x=[rep] * len(intensity_final),
        name=rep,
        marker_color='white',
        line_color=color,
        width=0.2,
        showlegend=False
    ))

fig.update_traces(box_visible=False)  # We already added boxplots
fig.update_layout(
    title="Intensity Distribution (log₂ transformed)",
    yaxis_title="log₂(Intensity)",
    violingap=0,
    violinmode='overlay',
    height=600,
    template="simple_white",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='lightgray')
)
st.plotly_chart(fig, use_container_width=True)

# === 3. CVs — CLEAN VIOLIN PLOTS ===
st.subheader("Technical Reproducibility (CV within Conditions)")

cv_data = []
for reps, name in [(c1, "Condition A"), (c2, "Condition B")]:
    if len(reps) >= 2:
        cv_per_protein = intensity_final[reps].std(axis=1) / intensity_final[reps].mean(axis=1) * 100
        cv_data.extend([{"Replicate": rep, "CV (%)": cv, "Condition": name} 
                       for rep in reps for cv in cv_per_protein])

cv_df = pd.DataFrame(cv_data)

fig = go.Figure()

for condition, color in [("Condition A", "#E71316"), ("Condition B", "#1f77b4")]:
    data = cv_df[cv_df["Condition"] == condition]["CV (%)"]
    fig.add_trace(go.Violin(
        y=data,
        name=condition,
        line_color=color,
        meanline_visible=True,
        box_visible=True,
        points=False
    ))

fig.update_layout(
    title="Coefficient of Variation (CV %) Within Conditions",
    yaxis_title="CV (%)",
    height=600,
    template="simple_white",
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)
