import pandas as pd
import numpy as np
from typing import List, Dict
def generate_mock_data(count: int = 500) -> pd.DataFrame:
    """    Generate mock proteomics data for demonstration    Args:        count: Number of proteins to generate    Returns:        DataFrame with protein data    """    genes = ['ALB', 'TP53', 'TNF', 'EGFR', 'VEGFA', 'APOE', 'IL6', 'TGFB1',
             'MTHFR', 'ESR1', 'AKT1', 'SOD2', 'COMT', 'CAT', 'BRCA1',
             'MYC', 'KRAS', 'NRAS', 'PIK3CA', 'PTEN', 'RB1', 'CDK4',
             'CDKN2A', 'SMAD4', 'APC', 'VHL', 'HIF1A', 'ERBB2', 'MET',
             'ALK', 'ROS1', 'RET', 'FGFR1', 'FGFR2', 'FGFR3', 'KIT',
             'PDGFRA', 'FLT3', 'JAK2', 'MPL', 'CALR', 'IDH1', 'IDH2',
             'SF3B1', 'SRSF2', 'U2AF1', 'ZRSR2', 'ASXL1', 'EZH2', 'DNMT3A']
    proteins = []
    for i in range(count):
        # Select gene name        if i < len(genes):
            gene = genes[i]
        else:
            gene = f'PROT{i}'        # Determine if significantly changed        is_sig = np.random.random() > 0.8        # Generate fold change (skewed for volcano plot shape)        log2FC = (np.random.random() - 0.5) * (8 if is_sig else 2)
        # Generate p-value        pVal = np.random.random() * 0.01 if is_sig else np.random.random()
        # Generate intensities        intensity_sample = np.random.random() * 100000        intensity_control = np.random.random() * 100000        proteins.append({
            'id': f'ID_{i}',
            'gene': gene,
            'description': f'{gene} protein description placeholder for scientific analysis. Involved in metabolic processes and cellular regulation.',
            'foldChange': 2 ** log2FC,
            'log2FoldChange': log2FC,
            'pValue': pVal,
            'negLog10PValue': -np.log10(pVal + 1e-300),  # Add small value to avoid log(0)            'intensitySample': intensity_sample,
            'intensityControl': intensity_control,
            'significance': 'NS'  # Will be calculated based on thresholds        })
    return pd.DataFrame(proteins)
