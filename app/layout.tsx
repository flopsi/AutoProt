import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AutoProt — Proteomics QC Dashboard',
  description: 'Automated proteomics quality control with PCA, PERMANOVA, and CV analysis',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
      </head>
      <body>{children}</body>
    </html>
  );
}
