"""
LFQbench Analysis Module - Core Analysis Engine
Implements benchmark proteomics analysis with fold-change calculations,
performance metrics, and statistical testing following LFQbench methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA


@dataclass
class BenchmarkConfig:
    """Configuration for LFQbench analysis"""
    # Expected fold-changes
    expected_fc_human: float = 0.0
    expected_fc_yeast: float = 1.0
    expected_fc_ecoli: float = -2.0
    expected_fc_celegans: float = -1.0
    
    # Filter thresholds
    limit_mv: float = 2/3  # Max missing values
    limit_cv: float = 20.0  # Max CV percentage
    limit_fc: float = 0.5   # Min fold-change for DE
    
    # Statistical threshold
    alpha_limma: float = 0.01
    
    # Species colors
    color_human: str = "#199d76"
    color_yeast: str = "#d85f02"
    color_ecoli: str = "#7570b2"
    color_celegans: str = "darkred"


class LFQbenchAnalyzer:
    """Main LFQbench analysis engine"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.expected_fc_map = {
            'Human': self.config.expected_fc_human,
            'Yeast': self.config.expected_fc_yeast,
            'E. coli': self.config.expected_fc_ecoli,
            'C.elegans': self.config.expected_fc_celegans
        }
    
    def calculate_cv(self, values: np.ndarray) -> float:
        """Calculate coefficient of variation"""
        if len(values) < 2 or np.all(np.isnan(values)):
            return np.nan
        
        clean_values = values[~np.isnan(values)]
        if len(clean_values) < 2:
            return np.nan
        
        mean = np.mean(clean_values)
        if mean == 0:
            return np.nan
        
        std = np.std(clean_values, ddof=1)
        return (std / mean) * 100
    
    def calculate_log2_fc(self, exp_values: np.ndarray, ctr_values: np.ndarray) -> float:
        """Calculate log2 fold-change"""
        exp_mean = np.nanmean(exp_values)
        ctr_mean = np.nanmean(ctr_values)
        
        if exp_mean <= 0 or ctr_mean <= 0 or np.isnan(exp_mean) or np.isnan(ctr_mean):
            return np.nan
        
        return np.log2(exp_mean / ctr_mean)
    
    def filter_by_completeness(self, df: pd.DataFrame, 
                              exp_cols: List[str], 
                              ctr_cols: List[str]) -> pd.DataFrame:
        """Filter proteins by data completeness"""
        min_valid = int(np.ceil(min(len(exp_cols), len(ctr_cols)) * (1 - self.config.limit_mv)))
        
        # Count valid values per protein
        df['exp_valid'] = df[exp_cols].notna().sum(axis=1)
        df['ctr_valid'] = df[ctr_cols].notna().sum(axis=1)
        
        # Keep proteins with sufficient data in both conditions
        filtered = df[(df['exp_valid'] >= min_valid) & (df['ctr_valid'] >= min_valid)].copy()
        
        filtered.drop(columns=['exp_valid', 'ctr_valid'], inplace=True)
        
        return filtered
    
    def calculate_cvs(self, df: pd.DataFrame,
                     exp_cols: List[str],
                     ctr_cols: List[str]) -> pd.DataFrame:
        """Calculate CVs for experimental and control conditions"""
        
        df['exp_cv'] = df[exp_cols].apply(lambda row: self.calculate_cv(row.values), axis=1)
        df['ctr_cv'] = df[ctr_cols].apply(lambda row: self.calculate_cv(row.values), axis=1)
        
        return df
    
    def filter_by_cv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter proteins by CV threshold"""
        return df[(df['exp_cv'] <= self.config.limit_cv) & 
                 (df['ctr_cv'] <= self.config.limit_cv)].copy()
    
    def calculate_fold_changes(self, df: pd.DataFrame,
                               exp_cols: List[str],
                               ctr_cols: List[str]) -> pd.DataFrame:
        """Calculate fold-changes for all proteins"""
        
        # Calculate means
        df['exp_mean'] = df[exp_cols].mean(axis=1)
        df['ctr_mean'] = df[ctr_cols].mean(axis=1)
        
        # Calculate log2 fold-change
        df['log2_fc'] = df.apply(
            lambda row: self.calculate_log2_fc(
                row[exp_cols].values,
                row[ctr_cols].values
            ),
            axis=1
        )
        
        # Add expected fold-change based on species
        df['expected_log2_fc'] = df['Species'].map(self.expected_fc_map)
        
        # Calculate deviation from expected
        df['fc_deviation'] = np.abs(df['log2_fc'] - df['expected_log2_fc'])
        
        return df
    
    def perform_limma_test(self, df: pd.DataFrame,
                          exp_cols: List[str],
                          ctr_cols: List[str]) -> pd.DataFrame:
        """
        Perform differential expression analysis - COMPLETELY FIXED
        """
        
        p_values = []
        t_stats = []
        
        print(f"  Performing t-tests on {len(df)} proteins...")
        
        for idx, row in df.iterrows():
            # Get experimental values
            exp_vals = row[exp_cols].values
            exp_vals = pd.to_numeric(exp_vals, errors='coerce')
            exp_vals = exp_vals[~pd.isna(exp_vals)]
            
            # Get control values
            ctr_vals = row[ctr_cols].values
            ctr_vals = pd.to_numeric(ctr_vals, errors='coerce')
            ctr_vals = ctr_vals[~pd.isna(ctr_vals)]
            
            # Need at least 2 values in each group
            if len(exp_vals) < 2 or len(ctr_vals) < 2:
                p_values.append(1.0)  # Non-significant
                t_stats.append(0.0)
                continue
            
            # Perform t-test
            try:
                t_stat, p_val = stats.ttest_ind(exp_vals, ctr_vals, equal_var=False)
                
                # Check for NaN results
                if pd.isna(t_stat) or pd.isna(p_val):
                    p_values.append(1.0)
                    t_stats.append(0.0)
                else:
                    p_values.append(float(p_val))
                    t_stats.append(float(t_stat))
                    
            except Exception as e:
                p_values.append(1.0)
                t_stats.append(0.0)
        
        df['p_value'] = p_values
        df['t_statistic'] = t_stats
        
        # Benjamini-Hochberg FDR correction
        df['p_adj'] = self._benjamini_hochberg(df['p_value'].values)
        
        # Debug: Check p-value distribution
        valid_p = df['p_adj'].dropna()
        print(f"  P-value range: {valid_p.min():.2e} to {valid_p.max():.2e}")
        print(f"  P-values < 0.01: {(valid_p < 0.01).sum()}")
        print(f"  P-values < 0.05: {(valid_p < 0.05).sum()}")
        
        # Classify differential abundance
        df['is_significant'] = (df['p_adj'] < self.config.alpha_limma) & \
                               (np.abs(df['log2_fc']) > self.config.limit_fc)
        
        print(f"  Significant proteins: {df['is_significant'].sum()}/{len(df)}")
    
    return df

        
        return df
    
    def _benjamini_hochberg(self, p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg FDR correction"""
        p_values = np.array(p_values)
        n = len(p_values)
        
        # Handle NaNs
        mask = ~np.isnan(p_values)
        adjusted = np.full(n, np.nan)
        
        if np.sum(mask) == 0:
            return adjusted
        
        # Sort p-values
        sorted_idx = np.argsort(p_values[mask])
        sorted_p = p_values[mask][sorted_idx]
        
        # Calculate adjusted p-values
        adjusted_p = np.minimum.accumulate(
            sorted_p * n / np.arange(1, len(sorted_p) + 1)[::-1]
        )[::-1]
        
        # Restore original order
        adjusted[mask] = adjusted_p[np.argsort(sorted_idx)]
        
        return adjusted
    
    def classify_de_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify results as TP, TN, FP, FN based on expected fold-changes"""
        
        def classify_row(row):
            expected_de = np.abs(row['expected_log2_fc']) > self.config.limit_fc
            observed_de = row['is_significant']
            
            if expected_de and observed_de:
                return 'true positive'
            elif expected_de and not observed_de:
                return 'false negative'
            elif not expected_de and observed_de:
                return 'false positive'
            else:
                return 'true negative'
        
        df['de_result'] = df.apply(classify_row, axis=1)
        
        return df
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        # Confusion matrix counts
        tp = len(df[df['de_result'] == 'true positive'])
        tn = len(df[df['de_result'] == 'true negative'])
        fp = len(df[df['de_result'] == 'false positive'])
        fn = len(df[df['de_result'] == 'false negative'])
        
        # Sensitivity and specificity
        sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        
        # Empirical FDR (deFDR)
        de_fdr = (fp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        
        # Accuracy (mean absolute deviation from expected)
        accuracy = float(df['fc_deviation'].mean()) if 'fc_deviation' in df.columns else np.nan
        
        # Trueness (cumulative systematic bias)
        trueness_values = []
        for species in self.expected_fc_map.keys():
            species_data = df[df['Species'] == species]
            if len(species_data) > 0:
                median_fc = float(species_data['log2_fc'].median())
                expected_fc = self.expected_fc_map[species]
                trueness_values.append(abs(median_fc - expected_fc))
        
        trueness = float(np.sum(trueness_values)) if trueness_values else np.nan
        
        # Precision (CV)
        cv_mean = float(np.mean([df['exp_cv'].mean(), df['ctr_cv'].mean()]))
        cv_median = float(np.median([df['exp_cv'].median(), df['ctr_cv'].median()]))
        
        return {
            'tp': float(tp),
            'tn': float(tn),
            'fp': float(fp),
            'fn': float(fn),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'de_fdr': float(de_fdr),
            'accuracy': float(accuracy),
            'trueness': float(trueness),
            'cv_mean': float(cv_mean),
            'cv_median': float(cv_median),
            'n_proteins': float(len(df))
        }
    
    def calculate_asymmetry_factor(self, log2_fc_values: np.ndarray) -> float:
        """Calculate asymmetry factor"""
        # Remove NaN values
        log2_fc_values = log2_fc_values[~np.isnan(log2_fc_values)]
        
        if len(log2_fc_values) < 10:
            return np.nan
        
        q1 = float(np.percentile(log2_fc_values, 25))
        q2 = float(np.percentile(log2_fc_values, 50))  # median
        q3 = float(np.percentile(log2_fc_values, 75))
        
        q2_q1 = abs(q2 - q1)
        q3_q2 = abs(q3 - q2)
        
        if q3_q2 == 0:
            return np.nan
        
        asymmetry = q2_q1 / q3_q2
        
        return float(asymmetry)
    
    def calculate_asymmetry_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate asymmetry factors for each species"""
        
        asymmetry_data = []
        
        for species in self.expected_fc_map.keys():
            species_data = df[df['Species'] == species]
            
            if len(species_data) >= 10:
                fc_values = species_data['log2_fc'].dropna().values
                asymmetry = self.calculate_asymmetry_factor(fc_values)
                
                asymmetry_data.append({
                    'Species': species,
                    'Asymmetry Factor': float(asymmetry) if not np.isnan(asymmetry) else np.nan,
                    'N Proteins': int(len(species_data))
                })
        
        return pd.DataFrame(asymmetry_data)
    
    def run_complete_analysis(self, df: pd.DataFrame,
                             exp_cols: List[str],
                             ctr_cols: List[str]) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """
        Run complete LFQbench analysis pipeline
        """
        
        # Ensure quantity columns are numeric
        print("Step 0: Converting columns to numeric...")
        for col in exp_cols + ctr_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("Step 1: Filtering by data completeness...")
        df_filtered = self.filter_by_completeness(df, exp_cols, ctr_cols)
        print(f"  {len(df_filtered)}/{len(df)} proteins passed completeness filter")
        
        print("Step 2: Calculating CVs...")
        df_filtered = self.calculate_cvs(df_filtered, exp_cols, ctr_cols)
        
        print("Step 3: Filtering by CV threshold...")
        df_cv_filtered = self.filter_by_cv(df_filtered)
        print(f"  {len(df_cv_filtered)}/{len(df_filtered)} proteins passed CV filter")
        
        print("Step 4: Calculating fold-changes...")
        df_fc = self.calculate_fold_changes(df_cv_filtered, exp_cols, ctr_cols)
        
        print("Step 5: Performing differential expression analysis...")
        df_de = self.perform_limma_test(df_fc, exp_cols, ctr_cols)
        
        print("Step 6: Classifying DE results...")
        df_classified = self.classify_de_results(df_de)
        
        print("Step 7: Calculating performance metrics...")
        metrics = self.calculate_performance_metrics(df_classified)
        
        print("Step 8: Calculating asymmetry factors...")
        asymmetry_df = self.calculate_asymmetry_metrics(df_classified)
        
        print("✅ Analysis complete!")
        
        return df_classified, metrics, asymmetry_df
    
    def perform_pca(self, df: pd.DataFrame, sample_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform PCA on sample data"""
        # Get data matrix (proteins x samples)
        data_matrix = df[sample_cols].T.dropna(axis=1)
        
        # Ensure numeric
        data_matrix = data_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # PCA
        pca = PCA(n_components=min(2, data_scaled.shape[1]))
        pca_result = pca.fit_transform(data_scaled)
        
        return pca_result, pca.explained_variance_ratio_


def get_lfqbench_analyzer(config: Optional[BenchmarkConfig] = None) -> LFQbenchAnalyzer:
    """Get LFQbench analyzer instance"""
    return LFQbenchAnalyzer(config)
