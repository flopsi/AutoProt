from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QCRequest(BaseModel):
    # For a stateless API without a DB, we accept the raw data records directly
    # In a production system, this would be a datasetId pointing to a storage bucket
    records: List[Dict[str, Any]]
    intensity_cols: List[str]
    conditions: List[str]
    transform: str = "log2"  # "log2" or "glog"

class PCAScore(BaseModel):
    sample: str
    condition: str
    pc1: float
    pc2: float

class PERMANOVAResult(BaseModel):
    pseudo_f: float
    p_value: float
    r_squared: float

class CVRecord(BaseModel):
    condition: str
    cv_values: List[float]

class QCResponse(BaseModel):
    pca_scores: List[PCAScore]
    variance_explained: List[float]
    permanova: PERMANOVAResult
    cv_distributions: List[CVRecord]
