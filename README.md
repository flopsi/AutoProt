# AutoProt — Proteomics QC Dashboard

Browser-based proteomics quality control dashboard. Upload a CSV/TSV intensity matrix and instantly get PCA, PERMANOVA, and CV analysis — all computed client-side.

**Live:** [autoprot.vercel.app](https://autoprot.vercel.app)

## Features

- **Drag-and-drop upload** — CSV or TSV with auto-delimiter detection
- **Log2 and glog transforms** — toggle between standard log2 and variance-stabilizing generalized log
- **PCA scatter plot** — with variance explained per component, color-coded by condition
- **PERMANOVA** — pseudo-F statistic, p-value (199 permutations), and R² effect size
- **CV violin plots** — coefficient of variation distributions per condition with box plots
- **Zero backend** — all analysis runs in your browser (Next.js + TypeScript)
- **Aurora Signal design** — dark glass UI with Plotly.js interactive charts

## Input Format

```
ProteinID, Sample_A1, Sample_A2, Sample_B1, Sample_B2, ...
Condition, A, A, B, B, ...        ← optional (auto-detected from sample names)
P12345, 1000000, 1100000, 2500000, 2600000, ...
P23456, 500000, 520000, 510000, 490000, ...
```

- First column: protein/gene IDs
- Remaining columns: raw intensity values per sample
- Optional second row: condition/group labels (if omitted, inferred from `_` prefix in sample names)

## Architecture

```
app/
  dashboard/           # Main QC dashboard page + upload panel
  layout.tsx           # Root layout with Inter font
  globals.css          # Aurora Signal design tokens & components
libs/
  analysis/
    engine.ts          # PCA (power iteration), PERMANOVA, CV, transforms
    csv-parser.ts      # CSV/TSV parser with papaparse
  charts/shared/       # Plotly chart wrapper + Aurora theme
```

## Development

```bash
npm install
npm run dev
```

Open [localhost:3000](http://localhost:3000).

## Deploy

Deploys automatically to Vercel on push to `main`.

## License

MIT
