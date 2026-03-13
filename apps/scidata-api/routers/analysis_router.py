from fastapi import APIRouter, HTTPException
import polars as pl
from models.schemas import QCRequest, QCResponse, PCAScore, PERMANOVAResult
from services.analysis import apply_transform, compute_pca, compute_permanova, compute_cvs

router = APIRouter()

@router.post("/qc-dashboard", response_model=QCResponse)
def generate_qc_dashboard(req: QCRequest):
    try:
        df = pl.DataFrame(req.records)

        # 1. Transform
        df_t = apply_transform(df, req.intensity_cols, req.transform)
        df_clean = df_t.drop_nulls(subset=req.intensity_cols)

        # 2. PCA
        scores, var_explained = compute_pca(df_clean, req.intensity_cols)

        pca_scores = []
        for i, col in enumerate(req.intensity_cols):
            pca_scores.append(PCAScore(
                sample=col,
                condition=req.conditions[i],
                pc1=float(scores[i, 0]),
                pc2=float(scores[i, 1])
            ))

        # 3. PERMANOVA
        perm = compute_permanova(df_clean, req.intensity_cols, req.conditions)

        # 4. CVs
        cvs = compute_cvs(df_t, req.intensity_cols, req.conditions)

        return QCResponse(
            pca_scores=pca_scores,
            variance_explained=var_explained,
            permanova=PERMANOVAResult(**perm),
            cv_distributions=cvs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
