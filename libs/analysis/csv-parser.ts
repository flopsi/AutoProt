import Papa from 'papaparse';
import type { ParsedData } from './engine';

export function parseCSV(text: string): ParsedData {
  const firstLine = text.split('\n')[0];
  const delimiter = firstLine.includes('\t') ? '\t' : ',';

  const parsed = Papa.parse(text.trim(), { delimiter, skipEmptyLines: true });
  const rows = parsed.data as string[][];
  if (rows.length < 3) throw new Error('File must have at least a header row and 2 data rows.');

  const header = rows[0];
  const sampleNames = header.slice(1).map(s => s.trim());
  if (sampleNames.length < 2) throw new Error('Need at least 2 sample columns. Found: ' + sampleNames.length);

  let conditions: string[];
  let dataStartRow: number;

  const secondRowFirstCell = (rows[1][0] || '').trim().toLowerCase();
  if (secondRowFirstCell === 'condition' || secondRowFirstCell === 'group' || secondRowFirstCell === 'class') {
    conditions = rows[1].slice(1).map(s => s.trim());
    dataStartRow = 2;
  } else {
    conditions = sampleNames.map(name => {
      const parts = name.split(/[_\-\.]/); return parts.length > 1 ? parts.slice(0, -1).join('_') : name;
    });
    dataStartRow = 1;
  }

  if (conditions.length !== sampleNames.length)
    throw new Error(`Condition count (${conditions.length}) doesn't match sample count (${sampleNames.length}).`);

  const proteinIds: string[] = [];
  const intensityMatrix: number[][] = [];

  for (let i = dataStartRow; i < rows.length; i++) {
    const row = rows[i]; if (!row || row.length < 2) continue;
    const proteinId = (row[0] || '').trim(); if (!proteinId) continue;
    const values = row.slice(1).map(v => {
      const s = (v || '').trim();
      if (s === '' || s === 'NA' || s === 'NaN' || s === '#N/A' || s === 'null') return 0;
      const num = parseFloat(s); return isNaN(num) ? 0 : num;
    });
    while (values.length < sampleNames.length) values.push(0);
    proteinIds.push(proteinId);
    intensityMatrix.push(values.slice(0, sampleNames.length));
  }

  if (intensityMatrix.length < 3) throw new Error(`Only ${intensityMatrix.length} proteins found. Need at least 3.`);
  return { sampleNames, conditions, intensityMatrix, proteinIds };
}
