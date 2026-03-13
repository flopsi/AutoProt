# AutoProt — SciData Proteomics QC Dashboard

An NX monorepo for automated proteomics quality control, featuring a **FastAPI** backend and a **Next.js** frontend with the Aurora Signal glass design system.

## Architecture

```
apps/
  scidata-api/        # FastAPI backend — PCA, PERMANOVA, CV analysis
  scidata-web/        # Next.js frontend — Aurora Signal glass UI
libs/
  charts/shared/      # Reusable Plotly.js chart components & Aurora theme
```

## Backend (`scidata-api`)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/api/qc-dashboard` | POST | Full QC pipeline: transform → PCA → PERMANOVA → CV distributions |

**Stack:** FastAPI, Polars, scikit-learn, scikit-bio, NumPy, SciPy

### Transforms
- **Log2** — `log2(x + 1)` for standard intensity normalization
- **Generalized log (glog)** — variance-stabilizing transform using median row variance as λ

### Analysis Pipeline
1. Intensity transformation (log2 / glog)
2. PCA with StandardScaler (up to 3 components)
3. PERMANOVA on Euclidean distance matrix (999 permutations)
4. Per-condition CV (coefficient of variation) distributions

## Frontend (`scidata-web`)

- **Aurora Signal** glass design system (deep navy, cyan/blue/violet palette)
- Interactive Plotly.js charts via `ChartWrapper` with SSR-safe lazy loading
- Transform toggle (log2 / glog) with live re-rendering
- Metric cards: Pseudo-F, p-value, R²
- PCA scatter plot + CV violin plots

## Shared Libraries

- `libs/charts/shared/src/aurora-plotly-theme.ts` — Plotly layout, config, and color sequence
- `libs/charts/shared/src/chart-wrapper.tsx` — Reusable glass-panel chart component with error boundary

## Getting Started

```bash
# Backend
cd apps/scidata-api
pip install fastapi uvicorn polars scikit-learn scikit-bio scipy numpy
uvicorn main:app --reload

# Frontend
cd apps/scidata-web
npm install
npm run dev
```

## License

MIT
