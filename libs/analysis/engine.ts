/**
 * Client-side proteomics QC analysis engine.
 * Implements PCA (via power iteration), CV distributions, and PERMANOVA
 * entirely in the browser.
 */

export interface ParsedData {
  sampleNames: string[];
  conditions: string[];
  intensityMatrix: number[][];
  proteinIds: string[];
}

export interface PCAScore {
  sample: string;
  condition: string;
  pc1: number;
  pc2: number;
}

export interface QCResult {
  pcaScores: PCAScore[];
  varianceExplained: [number, number];
  permanova: { pseudoF: number; pValue: number; rSquared: number };
  cvDistributions: { condition: string; cvValues: number[] }[];
  transform: 'log2' | 'glog';
}

function transpose(m: number[][]): number[][] {
  if (m.length === 0) return [];
  const rows = m.length, cols = m[0].length;
  const t: number[][] = Array.from({ length: cols }, () => new Array(rows));
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) t[j][i] = m[i][j];
  return t;
}

function mean(arr: number[]): number {
  let s = 0; for (const v of arr) s += v; return s / arr.length;
}

function std(arr: number[], m?: number): number {
  const mu = m ?? mean(arr);
  let s = 0; for (const v of arr) s += (v - mu) ** 2;
  return Math.sqrt(s / (arr.length - 1));
}

function colMeans(mat: number[][]): number[] {
  const cols = mat[0].length;
  const means = new Array(cols).fill(0);
  for (const row of mat) for (let j = 0; j < cols; j++) means[j] += row[j];
  for (let j = 0; j < cols; j++) means[j] /= mat.length;
  return means;
}

function colStds(mat: number[][], means: number[]): number[] {
  const cols = mat[0].length;
  const stds = new Array(cols).fill(0);
  for (const row of mat)
    for (let j = 0; j < cols; j++) stds[j] += (row[j] - means[j]) ** 2;
  for (let j = 0; j < cols; j++)
    stds[j] = Math.sqrt(stds[j] / (mat.length - 1)) || 1;
  return stds;
}

function log2Transform(mat: number[][]): number[][] {
  return mat.map(row => row.map(v => Math.log2((v || 0) + 1)));
}

function glogTransform(mat: number[][]): number[][] {
  const rowVars = mat.map(row => {
    const m = mean(row);
    return row.reduce((s, v) => s + (v - m) ** 2, 0) / (row.length - 1);
  });
  rowVars.sort((a, b) => a - b);
  const lam = rowVars[Math.floor(rowVars.length / 2)];
  return mat.map(row => row.map(v => Math.log2(v + Math.sqrt(v * v + lam))));
}

function powerIterationPCA(mat: number[][], nComponents = 2) {
  const n = mat.length, p = mat[0].length;
  const means = colMeans(mat);
  const stds = colStds(mat, means);
  const centered = mat.map(row => row.map((v, j) => (v - means[j]) / stds[j]));

  const cov: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (const row of centered)
    for (let i = 0; i < p; i++)
      for (let j = i; j < p; j++) {
        cov[i][j] += row[i] * row[j];
        if (i !== j) cov[j][i] += row[i] * row[j];
      }
  for (let i = 0; i < p; i++)
    for (let j = 0; j < p; j++) cov[i][j] /= (n - 1);

  const totalVar = cov.reduce((s, row, i) => s + row[i], 0);
  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = [];
  const deflated = cov.map(r => [...r]);

  for (let comp = 0; comp < nComponents; comp++) {
    let v = Array.from({ length: p }, () => Math.random() - 0.5);
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    v = v.map(x => x / norm);

    for (let iter = 0; iter < 300; iter++) {
      const newV = new Array(p).fill(0);
      for (let i = 0; i < p; i++)
        for (let j = 0; j < p; j++) newV[i] += deflated[i][j] * v[j];
      norm = Math.sqrt(newV.reduce((s, x) => s + x * x, 0));
      if (norm < 1e-12) break;
      const oldV = v;
      v = newV.map(x => x / norm);
      if (v.reduce((s, x, i) => s + (x - oldV[i]) ** 2, 0) < 1e-12) break;
    }

    const eigenvalue = v.reduce((s, _, i) => {
      let dot = 0;
      for (let j = 0; j < p; j++) dot += deflated[i][j] * v[j];
      return s + v[i] * dot;
    }, 0);

    eigenvalues.push(eigenvalue);
    eigenvectors.push([...v]);
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++)
        deflated[i][j] -= eigenvalue * v[i] * v[j];
  }

  const scores = centered.map(row =>
    eigenvectors.map(ev => row.reduce((s, x, j) => s + x * ev[j], 0))
  );
  return { scores, varianceExplained: eigenvalues.map(e => e / totalVar) };
}

function computeCVs(mat: number[][], sampleNames: string[], conditions: string[]) {
  const uniqueConditions = [...new Set(conditions)];
  return uniqueConditions.map(cond => {
    const colIndices = conditions.map((c, i) => (c === cond ? i : -1)).filter(i => i >= 0);
    if (colIndices.length < 2) return { condition: cond, cvValues: [] as number[] };
    const cvValues: number[] = [];
    for (const row of mat) {
      const vals = colIndices.map(i => row[i]).filter(v => !isNaN(v) && v !== 0);
      if (vals.length < 2) continue;
      const m = mean(vals);
      if (m === 0) continue;
      cvValues.push((std(vals, m) / Math.abs(m)) * 100);
    }
    return { condition: cond, cvValues };
  });
}

function euclidean(a: number[], b: number[]): number {
  let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s);
}

function computePERMANOVA(sampleVectors: number[][], conditions: string[]) {
  const n = sampleVectors.length;
  const groups = [...new Set(conditions)];
  const k = groups.length;
  if (k < 2 || n <= k) return { pseudoF: 0, pValue: 1, rSquared: 0 };

  const dist: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const d = euclidean(sampleVectors[i], sampleVectors[j]);
      dist[i][j] = d; dist[j][i] = d;
    }

  const ssTotal = (() => {
    let s = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) s += dist[i][j] ** 2;
    return s / n;
  })();

  function calcSSW(conds: string[]) {
    let sw = 0;
    for (const grp of groups) {
      const idx = conds.map((c, i) => (c === grp ? i : -1)).filter(i => i >= 0);
      if (idx.length < 2) continue;
      let gs = 0;
      for (let a = 0; a < idx.length; a++)
        for (let b = a + 1; b < idx.length; b++) gs += dist[idx[a]][idx[b]] ** 2;
      sw += gs / idx.length;
    }
    return sw;
  }

  const ssWithin = calcSSW(conditions);
  const ssBetween = ssTotal - ssWithin;
  const pseudoF = (ssBetween / (k - 1)) / (ssWithin / (n - k));
  const rSquared = ssBetween / ssTotal;

  const nPerms = 199;
  let greater = 0;
  for (let p = 0; p < nPerms; p++) {
    const permCond = [...conditions];
    for (let i = permCond.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [permCond[i], permCond[j]] = [permCond[j], permCond[i]];
    }
    const permSSW = calcSSW(permCond);
    const permF = ((ssTotal - permSSW) / (k - 1)) / (permSSW / (n - k));
    if (permF >= pseudoF) greater++;
  }

  return {
    pseudoF: isFinite(pseudoF) ? pseudoF : 0,
    pValue: (greater + 1) / (nPerms + 1),
    rSquared: isFinite(rSquared) ? rSquared : 0,
  };
}

export function runAnalysis(data: ParsedData, transform: 'log2' | 'glog'): QCResult {
  const mat = transform === 'log2' ? log2Transform(data.intensityMatrix) : glogTransform(data.intensityMatrix);
  const minValid = Math.max(2, Math.floor(data.sampleNames.length * 0.5));
  const filtered = mat.filter(row => row.filter(v => isFinite(v) && !isNaN(v) && v !== 0).length >= minValid);
  if (filtered.length < 3) throw new Error(`Only ${filtered.length} proteins pass filter. Need at least 3.`);

  const clean = filtered.map(row => {
    const valid = row.filter(v => isFinite(v) && !isNaN(v));
    const rowMean = valid.length > 0 ? mean(valid) : 0;
    return row.map(v => (isFinite(v) && !isNaN(v)) ? v : rowMean);
  });

  const samplesAsRows = transpose(clean);
  const { scores, varianceExplained } = powerIterationPCA(samplesAsRows, 2);

  return {
    pcaScores: data.sampleNames.map((name, i) => ({
      sample: name, condition: data.conditions[i], pc1: scores[i][0], pc2: scores[i][1],
    })),
    varianceExplained: [varianceExplained[0], varianceExplained[1]],
    permanova: computePERMANOVA(samplesAsRows, data.conditions),
    cvDistributions: computeCVs(clean, data.sampleNames, data.conditions),
    transform,
  };
}
