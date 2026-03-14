'use client';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any;

export function ChartWrapper({ title, badge, data, layout, height = 400 }: any) {
  return (
    <div className="glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
        {badge && <span className="badge">{badge}</span>}
        {title && <h3 style={{ color: '#e2e8f0', margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>{title}</h3>}
      </div>
      <Plot
        data={data}
        layout={{ ...layout, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', autosize: true }}
        useResizeHandler
        config={{ responsive: true, displaylogo: false }}
        style={{ width: '100%', height }}
      />
    </div>
  );
}
