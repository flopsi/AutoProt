'use client';
import { Suspense, lazy, ReactNode } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

// Safe dynamic import for Plotly to avoid SSR issues in Next.js
const Plot = lazy(() => import('react-plotly.js'));

export function ChartWrapper({ title, badge, data, layout, height = 400 }: any) {
  return (
    <div className="glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-4)' }}>
        {badge && <span className="badge badge--primary">{badge}</span>}
        {title && <h3 style={{ fontFamily: 'var(--font-display)', color: 'var(--color-text)', margin: 0 }}>{title}</h3>}
      </div>
      <Suspense fallback={<div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading...</div>}>
        <Plot
          data={data}
          layout={{ ...layout, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', autosize: true }}
          useResizeHandler
          style={{ width: '100%', height }}
        />
      </Suspense>
    </div>
  );
}
