'use client';

import { useState } from 'react';
import { ChartWrapper } from '@scidata/charts/shared';
import { auroraLayout, auroraConfig, auroraColorSequence } from '@scidata/charts/shared';

// Simulated API Call hook (in a real app, uses React Query + @scidata/data-access)
export default function DashboardPage() {
  const [activeTransform, setActiveTransform] = useState<'log2' | 'glog'>('log2');
  const [loading, setLoading] = useState(false);

  // Layout setup: Fewer, more logical panels combining PCA + Stats + Distributions
  return (
    <main className="section container">
      <header style={{ marginBottom: 'var(--space-8)' }}>
        <h1 className="text-gradient" style={{ fontSize: 'var(--text-3xl)', fontWeight: 700 }}>QC Overview</h1>
        <p className="text-muted">Consolidated mass spectrometry quality control analysis.</p>
      </header>

      <div style={{ marginBottom: 'var(--space-6)' }}>
        <div className="glass-panel" style={{ display: 'inline-flex', gap: 'var(--space-2)', padding: 'var(--space-2)' }}>
          <button 
            className={`btn ${activeTransform === 'log2' ? 'btn--primary' : 'btn--ghost'}`}
            onClick={() => setActiveTransform('log2')}
          >
            Log2 Transform
          </button>
          <button 
            className={`btn ${activeTransform === 'glog' ? 'btn--primary' : 'btn--ghost'}`}
            onClick={() => setActiveTransform('glog')}
          >
            Generalized Log (glog)
          </button>
        </div>
      </div>

      <div className="grid grid--3">
        {/* Metric Cards */}
        <div className="glass-card stat">
          <span className="stat__label">PERMANOVA Pseudo-F</span>
          <span className="stat__value">14.2</span>
        </div>
        <div className="glass-card stat">
          <span className="stat__label">PERMANOVA p-value</span>
          <span className="stat__value">0.001</span>
        </div>
        <div className="glass-card stat">
          <span className="stat__label">Effect Size (R²)</span>
          <span className="stat__value">0.85</span>
        </div>
      </div>

      <div className="grid grid--2" style={{ marginTop: 'var(--space-6)' }}>
        {/* PCA Plot */}
        <ChartWrapper
          title={`Global PCA (${activeTransform})`}
          badge="Separation"
          height={400}
          data={[{
            type: 'scatter',
            mode: 'markers+text',
            x: [-2.1, -1.9, -2.0, 2.5, 2.3, 2.4],
            y: [0.1, -0.2, 0.3, -0.1, 0.4, -0.2],
            text: ['A_1', 'A_2', 'A_3', 'B_1', 'B_2', 'B_3'],
            textposition: 'top center',
            marker: { size: 12, color: auroraColorSequence[0] }
          }]}
          layout={{
            ...auroraLayout,
            xaxis: { title: 'PC1 (65.2%)' },
            yaxis: { title: 'PC2 (12.1%)' }
          }}
        />

        {/* CV Violin Plot */}
        <ChartWrapper
          title="CV Distribution"
          badge="Precision"
          height={400}
          data={[{
            type: 'violin',
            y: [5, 10, 15, 20, 25, 8, 12, 18],
            name: 'Condition A',
            line: { color: auroraColorSequence[0] }
          }, {
            type: 'violin',
            y: [6, 11, 14, 22, 28, 9, 13, 19],
            name: 'Condition B',
            line: { color: auroraColorSequence[1] }
          }]}
          layout={{
            ...auroraLayout,
            yaxis: { title: 'Coefficient of Variation (%)' }
          }}
        />
      </div>
    </main>
  );
}
