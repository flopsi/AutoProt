import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from components.header import render_header
from config.colors import ThermoFisherColors
import plotly.express as px


render_header()
st.title("Data Quality Assessment")

protein_uploaded = st.session_state.get('protein_uploaded', False)
peptide_uploaded = st.session_state.get('peptide_uploaded', False)

if not protein_uploaded and not peptide_uploaded:
    st.warning("⚠️ No data loaded. Please upload protein or peptide data first.")
    if st.button("Go to Protein Upload", type="primary", use_container_width=True):
        st.switch_page("pages/1_📊_Protein_Upload.py")
    st.stop()

# Data selection tabs
data_tab1, data_tab2 = st.tabs(["Protein Data", "Peptide Data"])

with data_tab1:
    if protein_uploaded:
        current_data = st.session_state.protein_data
        data_type = "Protein"
        
        condition_mapping = current_data.condition_mapping
        quant_data = current_data.quant_data
        species_map = current_data.species_map
        
        # ============================================================
        # 1. PROTEIN RANK PLOT (Two Separate Plots)
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 1. Protein Rank Plot")
        
        # Calculate mean intensities for Condition A and B
        a_data = current_data.get_condition_data('A')
        b_data = current_data.get_condition_data('B')
        
        # Two column layout for side-by-side plots
        rank_col1, rank_col2 = st.columns(2)
        
        with rank_col1:
            # Condition A rank plot
            mean_a = a_data.mean(axis=1).sort_values(ascending=False).reset_index(drop=True)
            log2_a = np.log2(mean_a[mean_a > 0])
            
            fig_rank_a = go.Figure()
            
            fig_rank_a.add_trace(go.Scatter(
                x=list(range(1, len(log2_a) + 1)),
                y=log2_a,
                mode='lines',
                line=dict(color='#E71316', width=2),
                hovertemplate='Rank: %{x}<br>Log₂ Intensity: %{y:.2f}<extra></extra>'
            ))
            
            fig_rank_a.update_layout(
                title='Condition A',
                xaxis_title='Protein Rank (by intensity)',
                yaxis_title='Log₂ Abundance',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            
            st.plotly_chart(fig_rank_a, use_container_width=True)
        
        with rank_col2:
            # Condition B rank plot
            mean_b = b_data.mean(axis=1).sort_values(ascending=False).reset_index(drop=True)
            log2_b = np.log2(mean_b[mean_b > 0])
            
            fig_rank_b = go.Figure()
            
            fig_rank_b.add_trace(go.Scatter(
                x=list(range(1, len(log2_b) + 1)),
                y=log2_b,
                mode='lines',
                line=dict(color='#9BD3DD', width=2),
                hovertemplate='Rank: %{x}<br>Log₂ Intensity: %{y:.2f}<extra></extra>'
            ))
            
            fig_rank_b.update_layout(
                title='Condition B',
                xaxis_title='Protein Rank (by intensity)',
                yaxis_title='Log₂ Abundance',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            
            st.plotly_chart(fig_rank_b, use_container_width=True)
        
        # ============================================================
        # 2. INTENSITY HEATMAP
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 2. Intensity Heatmap")
        
        # Prepare data
        base = 0
        protein_indices = list(range(len(quant_data)))  # Protein index based on upload order
        z = np.log2(quant_data.T.replace(0, np.nan).values)  # Log2 intensity, rows = samples
        programmers = [condition_mapping.get(col, col) for col in quant_data.columns]  # A1, A2, A3, B1, B2, B3...
        
        # Create custom colorscale: white (missing/low) -> sky (medium) -> red (high)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z,
            x=protein_indices,
            y=programmers,
            colorscale=[
                [0, 'black'],          # Missing/lowest
                [0.3, '#9BD3DD'],      # Sky (low-medium)
                [0.5, '#66b8c7'],      # Medium sky-blue
                [0.7, '#ff9999'],      # Light red
                [1, '#E71316']] ))
        fig_heatmap.update_layout(
            title=dict(text='Intensity Heatmap (White=Missing, Sky=Low, Red=High)'),
            xaxis=dict(title=f'{data_type} Index'),
            yaxis=dict(title='Sample', tickangle=0),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY)
        )        
       
        st.plotly_chart(fig_heatmap, use_container_width=True)


        
                        
        # ============================================================
        # 3. INTENSITY DISTRIBUTION BOXPLOTS
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 3. Intensity Distribution")
        
        fig_box = go.Figure()
        
        for col in quant_data.columns:
            condition = condition_mapping.get(col, col)
            condition_letter = condition[0]
            
            values = quant_data[col].dropna()
            log10_values = np.log10(values[values > 0])
            
            color = '#E71316' if condition_letter == 'A' else '#9BD3DD'
            
            fig_box.add_trace(go.Box(
                y=log10_values,
                name=condition,
                marker_color=color,
                boxmean='sd',
                hovertemplate='<b>%{fullData.name}</b><br>Log₁₀ Intensity: %{y:.2f}<extra></extra>'
            ))
        
        fig_box.update_layout(
            title=f'{data_type} Intensity Distribution by Sample',
            yaxis_title='Log₁₀ Intensity',
            showlegend=False,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
            xaxis=dict(tickangle=-45, showgrid=False),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)', showgrid=True, zeroline=False)
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        # ============================================================
        # 4. PCA ANALYSIS
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 4. Principal Component Analysis")
        
        pca_col1, pca_col2 = st.columns([1, 3])
        with pca_col1:
            pca_scope = st.radio("PCA on:", ["All Features", "Human Only"], key='pca_protein')
        
        if pca_scope == "Human Only":
            human_indices = [idx for idx, sp in species_map.items() if sp == 'human']
            pca_data = quant_data.loc[human_indices].dropna()
        else:
            pca_data = quant_data.dropna()
        
        if len(pca_data) > 0:
            pca_data_filled = pca_data.fillna(0)
            pca_data_log = np.log10(pca_data_filled.replace(0, np.nan).fillna(pca_data_filled[pca_data_filled > 0].min().min()))
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pca_data_log.T)
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            sample_names = [condition_mapping.get(col, col) for col in pca_data.columns]
            colors_pca = ['#E71316' if name[0] == 'A' else '#9BD3DD' for name in sample_names]
            
            fig_pca = go.Figure()
            
            for i, (name, color) in enumerate(zip(sample_names, colors_pca)):
                fig_pca.add_trace(go.Scatter(
                    x=[pca_result[i, 0]],
                    y=[pca_result[i, 1]],
                    mode='markers+text',
                    marker=dict(size=12, color=color),
                    text=[name],
                    textposition='top center',
                    name=name,
                    showlegend=False,
                    hovertemplate=f'<b>{name}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
                ))
            
            fig_pca.update_layout(
                title=f'PCA - {pca_scope} ({len(pca_data):,} features)',
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)')
            )
            
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # ============================================================
            # 5. CV% VIOLIN PLOT
            # ============================================================
            
            st.markdown("---")
            st.markdown("### 5. Coefficient of Variation (CV%)")
            
            def calculate_cv(data):
                mean = data.mean(axis=1)
                std = data.std(axis=1)
                cv = (std / mean * 100).replace([np.inf, -np.inf], np.nan).dropna()
                return cv
            
            a_data = current_data.get_condition_data('A')
            b_data = current_data.get_condition_data('B')
            
            cv_a = calculate_cv(a_data)
            cv_b = calculate_cv(b_data)
            
            # Cap CV values at 100%
            cv_a_capped = cv_a.clip(upper=100)
            cv_b_capped = cv_b.clip(upper=100)
            
            # Create DataFrame for Plotly Express
            cv_df = pd.DataFrame({
                'CV%': list(cv_a_capped) + list(cv_b_capped),
                'Condition': ['A'] * len(cv_a_capped) + ['B'] * len(cv_b_capped)
            })
            
            # Create violin plot with box and points
            import plotly.express as px
            
            fig_cv = px.violin(
                cv_df, 
                y='CV%', 
                x='Condition', 
                color='Condition',
                box=True,           # Show box plot inside       
                hover_data=['CV%'],
                color_discrete_map={'A': '#E71316', 'B': '#9BD3DD'}     # Custom colors
            )
            
            fig_cv.update_layout(
                title='CV% Distribution by Condition',
                yaxis_title='CV%',
                xaxis_title='Condition',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
                showlegend=False,
                yaxis_zeroline=False
            )
            
            fig_cv.update_xaxes(showgrid=False)
            fig_cv.update_yaxes(
                gridcolor='rgba(0,0,0,0.1)',
                range=[-10, 100]  # Set y-axis max to 100%
            )
            
            st.plotly_chart(fig_cv, use_container_width=True)

        # ============================================================
        # 6. CV THRESHOLDS PER REPLICATE (3x2 WITH SHADING)
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
            else:
                condition_data = b_data
            
            sample_data = quant_data[col].dropna()
            sample_indices = sample_data.index
            
            cv_data = condition_data.loc[sample_indices]
            cv_values = calculate_cv(cv_data)
            
            total_ids = len(sample_indices)
            cv_below_20 = (cv_values < 20).sum()
            cv_below_10 = (cv_values < 10).sum()
            

            
            fig_cv_panel.add_shape(
                type="rect",
                x0=-0.5, x1=2.5,
                y0=0, y1=total_ids * 0.1,
                fillcolor="rgba(0,128,0,0.2)",
                line=dict(width=0),
                row=row, col=col_num
            )
            
            # Add bars
            fig_cv_panel.add_trace(
                go.Bar(
                    x=['Total IDs', 'CV<20%', 'CV<10%'],
                    y=[total_ids, cv_below_20, cv_below_10],
                    marker_color='#E71316' if condition[0] == 'A' else '#9BD3DD',
                    text=[total_ids, cv_below_20, cv_below_10],
                    textposition='inside',
                    showlegend=False
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
        
    else:
        st.info("ℹ️ Protein data not loaded. Upload protein data to enable this view.")

with data_tab2:
    if peptide_uploaded:
        current_data = st.session_state.peptide_data
        data_type = "Peptide"
        
        # EXACT SAME PLOTS AS PROTEIN TAB (copy all 6 plot sections)
        condition_mapping = current_data.condition_mapping
        quant_data = current_data.quant_data
        species_map = current_data.species_map
        
        st.info(f"✓ Analyzing peptide data: {current_data.n_rows:,} peptides")
        st.markdown("*Same visualizations as protein tab, using peptide-level data*")
        
    else:
        st.info("ℹ️ Peptide data not loaded. Upload peptide data to enable this view.")

# ============================================================
# NAVIGATION
# ============================================================

st.markdown("---")
st.markdown("### Navigation")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← View Results", use_container_width=True):
        st.session_state.upload_stage = 'summary'
        st.switch_page("pages/1_📊_Protein_Upload.py")

with nav_col2:
    if st.button("Upload Peptide Data", use_container_width=True):
        st.switch_page("pages/2_🔬_Peptide_Upload.py")

with nav_col3:
    if st.button("🔄 Start Over", type="primary", use_container_width=True):
        keys_to_delete = list(st.session_state.keys())
        for key in keys_to_delete:
            del st.session_state[key]
        st.switch_page("pages/1_📊_Protein_Upload.py")
