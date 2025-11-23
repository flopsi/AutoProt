from typing import List, Dict

def auto_detect_conditions(column_names: List[str]) -> Dict[str, str]:
    n_cols = len(column_names)
    n_half = n_cols // 2
    mapping = {}
    for i, col in enumerate(column_names[:n_half]):
        mapping[col] = f"A{i+1}"
    for i, col in enumerate(column_names[n_half:]):
        mapping[col] = f"B{i+1}"
    return mapping