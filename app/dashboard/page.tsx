'use client';

import { useState, useCallback } from 'react';
import { ChartWrapper } from '@scidata/charts/shared';
import { auroraLayout, auroraColorSequence } from '@scidata/charts/shared';
import { UploadPanel } from './upload-panel';
import { parseCSV } from '@/libs/analysis/csv-parser';
import { runAnalysis, type QCResult, type ParsedData } from '@/libs/analysis/engine';

export default function DashboardPage() {
  const [activeTransform, setActiveTransform] = useState<'log2' | 'glog'>('log2');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QCResult | null>(null);
  const [parsedData, setParsedData] = useState<ParsedData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dataInfo, setDataInfo] = useState<string>('');

  const handleFileLoaded = useCallback((text: string, fileName: string) => {
    setLoading(true);
    setError(null);

    setTimeout(() => {
      try {
        const data = parseCSV(text);
        setParsedData(data);

        const uniqueConditions = [...new Set(data.conditions)];
        setDataInfo(
          `${fileName} — ${data.proteinIds.length} proteins × ${data.sampleNames.length} samples — ${uniqueConditions.length} conditions (${uniqueConditions.join(', ')})`
        );

        const qcResult = runAnalysis(data, activeTransform);
        setResult(qcResult);
      } catch (e: any) {
        setError(e.message || 'Failed to parse file.');
        setResult(null);
        setParsedData(null);
      } finally {
        setLoading(false);
      }
    }, 50);
  }, [activeTransform]);

  const handleTransformChange = useCallback(
    (t: 'log2' | 'glog') => {
      setActiveTransform(t);
      if (!parsedData) return;
      setLoading(true);
      setTimeout(() => {
        try {
          const qcResult = runAnalysis(parsedData, t);
          setResult(qcResult);
        } catch (e: any) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      }, 50);
    },
    [parsedData]
  );

  const pcaChartData = result ? (() => {
    const conditions = [...new Set(result.pcaScores.map(s => s.condition))];
    return conditions.map((cond, ci) => {
      const pts = result.pcaScores.filter(s => s.condition === cond);
      return {
        type: 'scatter' as const,
        mode: 'markers+text' as const,
        name: cond,
        x: pts.map(p => p.pc1),
        y: pts.map(p => p.pc2),
        text: pts.map(p => p.sample),
        textposition: 'top center' as const,
        marker: {
          size: 11,
          color: auroraColorSequence[ci % auroraColorSequence.length],
          line: { width: 1, color: 'rgba(255,255,255,0.2)' },
        },
      };
    });
  })() : [];

  const cvChartData = result
    ? result.cvDistributions.map((cv, ci) => ({
        type: 'violin' as const,
        y: cv.cvValues,
        name: cv.condition,
        line: { color: auroraColorSequence[ci % auroraColorSequence.length] },
        fillcolor: auroraColorSequence[ci % auroraColorSequence.length].replace(')', ', 0.15)').replace('rgb', 'rgba'),
        meanline: { visible: true },
        box: { visible: true },
      }))
    : [];

  const fmt = (n: number, d: number = 3) =>
    isFinite(n) ? n.toFixed(d) : '—';

  return (
    <div className="container">
      <header className="page-header">
        <h1>QC Overview</h1>
        <p>Upload a proteomics intensity matrix to run quality control analysis</p>
      </header>

      <UploadPanel onFileLoaded={handleFileLoaded} isLoading={loading} />

      {error && (
        <div className="error-banner">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          {error}
        </div>
      )}

      {result && (
        <>
          {dataInfo && <p className="data-info">{dataInfo}</p>}

          <div className="toggle-group">
            <button
              className={`toggle-btn ${activeTransform === 'log2' ? 'active' : ''}`}
              onClick={() => handleTransformChange('log2')}
            >
              Log2 Transform
            </button>
            <button
              className={`toggle-btn ${activeTransform === 'glog' ? 'active' : ''}`}
              onClick={() => handleTransformChange('glog')}
            >
              Generalized Log (glog)
            </button>
          </div>

          <div className="stats-grid">
            <div className="stat-card">
              <div className="label">PERMANOVA Pseudo-F</div>
              <div className="value highlight">{fmt(result.permanova.pseudoF, 2)}</div>
            </div>
            <div className="stat-card">
              <div className="label">PERMANOVA p-value</div>
              <div className="value highlight">{fmt(result.permanova.pValue, 3)}</div>
            </div>
            <div className="stat-card">
              <div className="label">Effect Size (R²)</div>
              <div className="value highlight">{fmt(result.permanova.rSquared, 3)}</div>
            </div>
          </div>

          <div className="charts-grid">
            <ChartWrapper
              title={`Global PCA (${activeTransform})`}
              badge="Separation"
              height={420}
              data={pcaChartData}
              layout={{
                ...auroraLayout,
                xaxis: {
                  ...auroraLayout.xaxis,
                  title: `PC1 (${(result.varianceExplained[0] * 100).toFixed(1)}%)`,
                  tickformat: '.1f',
                },
                yaxis: {
                  ...auroraLayout.yaxis,
                  title: `PC2 (${(result.varianceExplained[1] * 100).toFixed(1)}%)`,
                  tickformat: '.1f',
                },
                legend: { font: { color: '#8894b0' }, orientation: 'h' as const, y: -0.15 },
              }}
            />

            <ChartWrapper
              title="CV Distribution"
              badge="Precision"
              height={420}
              data={cvChartData}
              layout={{
                ...auroraLayout,
                yaxis: { ...auroraLayout.yaxis, title: 'Coefficient of Variation (%)' },
                legend: { font: { color: '#8894b0' }, orientation: 'h' as const, y: -0.15 },
              }}
            />
          </div>
        </>
      )}

      <footer>
        <a href="https://github.com/flopsi/AutoProt" target="_blank" rel="noopener noreferrer">
          AutoProt on GitHub
        </a>
      </footer>
    </div>
  );
}
