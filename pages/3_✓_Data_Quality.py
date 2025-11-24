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
        protein_indices = list(range(len(quant_data)))
        z = np.log2(quant_data.T.replace(0, np.nan).values)
        programmers = [condition_mapping.get(col, col) for col in quant_data.columns]
        
        # Create custom colorscale: black (missing/low) -> sky (medium) -> red (high)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z,
            x=protein_indices,
            y=programmers,
            colorscale=[
                [0, 'black'],
                [0.3, '#9BD3DD'],
                [0.5, '#66b8c7'],
                [0.7, '#ff9999'],
                [1, '#E71316']
            ],
            colorbar=dict(
                title=dict(
                    text='Log₂<br>Intensity',
                    side='right'
                )
            ),
            hovertemplate='Sample: %{y}<br>Protein: %{x}<br>Log₂ Intensity: %{z:.2f}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title=dict(text='Intensity Heatmap (Black=Missing, Sky=Low, Red=High)'),
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
        # 4. PCA ANALYSIS - FIXED TO CHECK SPECIES MAP
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 4. Principal Component Analysis")
        
        pca_col1, pca_col2 = st.columns([1, 3])
        with pca_col1:
            pca_scope = st.radio("PCA on:", ["All Features", "Human Only"], key='pca_protein')
        
        # Check if species_map has human proteins
        human_count = sum(1 for sp in species_map.values() if sp == 'human')
        
        if pca_scope == "Human Only":
            if human_count == 0:
                st.warning(f"⚠️ No human proteins found in species map. Showing all {len(quant_data)} features instead.")
                pca_data = quant_data.dropna()
            else:
                human_indices = [idx for idx, sp in species_map.items() if sp == 'human']
                pca_data = quant_data.loc[human_indices].dropna()
                if len(pca_data) == 0:
                    st.warning("⚠️ All human proteins have missing values. Showing all features instead.")
                    pca_data = quant_data.dropna()
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
        else:
            st.warning("⚠️ No data available for PCA after filtering.")
        
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
        
        cv_a = calculate_cv(a_data).clip(upper=100)
        cv_b = calculate_cv(b_data).clip(upper=100)
        
        cv_df = pd.DataFrame({
            'CV%': list(cv_a) + list(cv_b),
            'Condition': ['A'] * len(cv_a) + ['B'] * len(cv_b)
        })
        
        fig_cv = px.violin(
            cv_df, 
            y='CV%', 
            x='Condition', 
            color='Condition',
            box=True,
            hover_data=['CV%'],
            color_discrete_map={'A': '#E71316', 'B': '#9BD3DD'}
        )
        
        fig_cv.update_layout(
            title='CV% Distribution by Condition',
            yaxis_title='CV%',
            xaxis_title='Condition',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
            showlegend=False
        )
        
        fig_cv.update_xaxes(showgrid=False)
        fig_cv.update_yaxes(gridcolor='rgba(0,0,0,0.1)', range=[0, 100])
        
        st.plotly_chart(fig_cv, use_container_width=True)
    
    else:
        st.info("ℹ️ Protein data not loaded. Upload protein data to enable this view.")

with data_tab2:
    if peptide_uploaded:
        current_data = st.session_state.peptide_data
        data_type = "Peptide"
        
        condition_mapping = current_data.condition_mapping
        quant_data = current_data.quant_data
        species_map = current_data.species_map
        
        # Get actual peptide columns that exist in the data
        peptide_cols = quant_data.columns.tolist()
        
        # Create condition mapping for peptide columns
        peptide_condition_mapping = {}
        for col in peptide_cols:
            matched = False
            for prot_col, condition in condition_mapping.items():
                if any(part in col for part in prot_col.split('_')[:3]):
                    peptide_condition_mapping[col] = condition
                    matched = True
                    break
            if not matched:
                idx = peptide_cols.index(col)
                peptide_condition_mapping[col] = f"A{idx+1}" if idx < 3 else f"B{idx-2}"
        
        condition_mapping = peptide_condition_mapping
        
        # ============================================================
        # 1. PEPTIDE RANK PLOT
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 1. Peptide Rank Plot")
        
        a_cols = [col for col, cond in condition_mapping.items() if cond.startswith('A')]
        b_cols = [col for col, cond in condition_mapping.items() if cond.startswith('B')]
        
        a_data = quant_data[a_cols]
        b_data = quant_data[b_cols]
        
        rank_col1, rank_col2 = st.columns(2)
        
        with rank_col1:
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
                xaxis_title='Peptide Rank (by intensity)',
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
                xaxis_title='Peptide Rank (by intensity)',
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
        
        peptide_indices = list(range(len(quant_data)))
        z = np.log2(quant_data.T.replace(0, np.nan).values)
        programmers = [condition_mapping.get(col, col) for col in quant_data.columns]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z,
            x=peptide_indices,
            y=programmers,
            colorscale=[
                [0, 'black'],
                [0.3, '#9BD3DD'],
                [0.5, '#66b8c7'],
                [0.7, '#ff9999'],
                [1, '#E71316']
            ],
            colorbar=dict(
                title=dict(
                    text='Log₂<br>Intensity',
                    side='right'
                )
            ),
            hovertemplate='Sample: %{y}<br>Peptide: %{x}<br>Log₂ Intensity: %{z:.2f}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title=dict(text='Intensity Heatmap (Black=Missing, Sky=Low, Red=High)'),
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
        # 4. PCA ANALYSIS - FIXED TO CHECK SPECIES MAP
        # ============================================================
        
        st.markdown("---")
        st.markdown("### 4. Principal Component Analysis")
        
        pca_col1, pca_col2 = st.columns([1, 3])
        with pca_col1:
            pca_scope = st.radio("PCA on:", ["All Features", "Human Only"], key='pca_peptide')
        
        # Check if species_map has human entries that match peptide indices
        valid_human_indices = [idx for idx in quant_data.index if idx in species_map and species_map[idx] == 'human']
        human_count = len(valid_human_indices)
        
        if pca_scope == "Human Only":
            if human_count == 0:
                st.warning(f"⚠️ No human peptides found in species map. Showing all {len(quant_data)} features instead.")
                pca_data = quant_data.dropna()
            else:
                pca_data = quant_data.loc[valid_human_indices].dropna()
                if len(pca_data) == 0:
                    st.warning("⚠️ All human peptides have missing values. Showing all features instead.")
                    pca_data = quant_data.dropna()
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
        else:
            st.warning("⚠️ No data available for PCA after filtering.")
        
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
        
        cv_a = calculate_cv(a_data).clip(upper=100)
        cv_b = calculate_cv(b_data).clip(upper=100)
        
        cv_df = pd.DataFrame({
            'CV%': list(cv_a) + list(cv_b),
            'Condition': ['A'] * len(cv_a) + ['B'] * len(cv_b)
        })
        
        fig_cv = px.violin(
            cv_df, 
            y='CV%', 
            x='Condition', 
            color='Condition',
            box=True,
            hover_data=['CV%'],
            color_discrete_map={'A': '#E71316', 'B': '#9BD3DD'}
        )
        
        fig_cv.update_layout(
            title='CV% Distribution by Condition',
            yaxis_title='CV%',
            xaxis_title='Condition',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", color=ThermoFisherColors.PRIMARY_GRAY),
            showlegend=False
        )
        
        fig_cv.update_xaxes(showgrid=False)
        fig_cv.update_yaxes(gridcolor='rgba(0,0,0,0.1)', range=[0, 100])
        
        st.plotly_chart(fig_cv, use_container_width=True)
    
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
        # Clear all session state keys
        keys_to_delete = list(st.session_state.keys())
        for key in keys_to_delete:
            del st.session_state[key]
        
        # Explicitly reset upload flags
        st.session_state.protein_uploaded = False
        st.session_state.peptide_uploaded = False
        st.session_state.upload_stage = 'upload'
        
        # Redirect to protein upload page
        st.switch_page("pages/1_📊_Protein_Upload.py")
