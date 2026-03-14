'use client';

import { useCallback, useState, useRef } from 'react';

interface UploadPanelProps {
  onFileLoaded: (text: string, fileName: string) => void;
  isLoading: boolean;
}

export function UploadPanel({ onFileLoaded, isLoading }: UploadPanelProps) {
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        if (text) onFileLoaded(text, file.name);
      };
      reader.readAsText(file);
    },
    [onFileLoaded]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      className={`upload-zone ${dragOver ? 'upload-zone--active' : ''} ${isLoading ? 'upload-zone--loading' : ''}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.tsv,.txt"
        onChange={onInputChange}
        style={{ display: 'none' }}
      />

      {isLoading ? (
        <div className="upload-zone__content">
          <div className="upload-spinner" />
          <p className="upload-zone__title">Analyzing...</p>
          <p className="upload-zone__sub">Running PCA, PERMANOVA, and CV calculations</p>
        </div>
      ) : (
        <div className="upload-zone__content">
          <svg className="upload-zone__icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <p className="upload-zone__title">
            {fileName ? fileName : 'Drop your intensity matrix here'}
          </p>
          <p className="upload-zone__sub">
            CSV or TSV with samples as columns, proteins as rows.
            {fileName ? ' Drop another file to replace.' : ' Click to browse.'}
          </p>
          <div className="upload-zone__format">
            <code>ProteinID, Sample_A1, Sample_A2, Sample_B1, Sample_B2, ...</code>
            <br />
            <code>Condition, A, A, B, B, ...</code>
            <span className="upload-zone__optional">(optional row)</span>
          </div>
        </div>
      )}
    </div>
  );
}
