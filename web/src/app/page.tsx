'use client';

import React, { useState } from 'react';
import UploadArea from '@/components/UploadArea';
import ResultChart from '@/components/ResultChart';

interface Prediction {
  genre: string;
  confidence: number;
  all_scores: Record<string, number>;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Inference failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to analyze audio');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#F8FAFC] flex flex-col items-center py-16 px-4">
      <div className="max-w-2xl w-full">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-block px-4 py-1.5 bg-indigo-50 text-indigo-700 rounded-full text-xs font-bold tracking-widest uppercase mb-4">
            AI-Powered Analysis
          </div>
          <h1 className="text-4xl md:text-5xl font-black text-slate-900 mb-4 tracking-tight">
            Genre <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-violet-600">Detector</span>
          </h1>
          <p className="text-slate-500 text-lg">
            Upload an MP3 track to identify its musical genre using our CNN model.
          </p>
        </header>

        {/* Upload Section */}
        <section className="bg-white rounded-3xl p-8 shadow-sm border border-slate-100 mb-8">
          <UploadArea onFileSelect={setFile} selectedFile={file} />

          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className={`w-full mt-8 py-4 px-6 rounded-xl font-bold text-lg transition-all duration-300 flex items-center justify-center gap-2
              ${!file || loading
                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                : 'bg-indigo-600 text-white hover:bg-indigo-700 hover:shadow-lg active:scale-95 shadow-indigo-200'
              }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Analyzing...
              </>
            ) : (
              'Start Analysis'
            )}
          </button>

          {error && (
            <div className="mt-6 p-4 bg-rose-50 border border-rose-100 rounded-xl flex items-start gap-3">
              <svg className="w-5 h-5 text-rose-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="text-sm font-bold text-rose-600">Error occurred</p>
                <p className="text-xs text-rose-500">{error}</p>
              </div>
            </div>
          )}
        </section>

        {/* Loading Skeleton / Animation */}
        {loading && (
          <div className="w-full mt-8 animate-pulse">
            <div className="bg-white rounded-2xl p-8 border border-slate-100">
              <div className="h-6 w-32 bg-slate-100 rounded-full mx-auto mb-4" />
              <div className="h-12 w-48 bg-slate-100 rounded-xl mx-auto mb-8" />
              <div className="space-y-4">
                {[1, 2, 3, 4].map(i => (
                  <div key={i} className="h-3 bg-slate-50 rounded-full w-full" />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <ResultChart
            predictedGenre={result.genre}
            confidence={result.confidence}
            allScores={result.all_scores}
          />
        )}

        {/* Footer info */}
        <footer className="mt-16 text-center">
          <p className="text-slate-400 text-sm">
            Powered by PyTorch & Next.js • CNN Music Genre Classification
          </p>
        </footer>
      </div>
    </main>
  );
}
