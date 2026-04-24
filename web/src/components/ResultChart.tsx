'use client';

import React from 'react';

interface ResultChartProps {
    confidences: Record<string, number>;
    predictedGenre: string;
}

export default function ResultChart({ confidences, predictedGenre }: ResultChartProps) {
    // Sort genres by confidence descending
    const sortedGenres = Object.entries(confidences).sort(([, a], [, b]) => b - a);

    return (
        <div className="w-full mt-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-8 border border-slate-100">
                <div className="text-center mb-8">
                    <p className="text-sm uppercase tracking-widest text-slate-400 font-semibold mb-2">Predicted Genre</p>
                    <h2 className="text-5xl font-black text-indigo-600 drop-shadow-sm">
                        {predictedGenre}
                    </h2>
                </div>

                <div className="space-y-4">
                    {sortedGenres.map(([genre, score]) => (
                        <div key={genre} className="group">
                            <div className="flex justify-between items-center mb-1.5">
                                <span className={`text-sm font-bold ${genre === predictedGenre ? 'text-indigo-600' : 'text-slate-600'}`}>
                                    {genre}
                                </span>
                                <span className="text-xs font-mono text-slate-400">
                                    {(score * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                <div
                                    className={`h-full rounded-full transition-all duration-1000 ease-out fill-mode-forwards
                    ${genre === predictedGenre
                                            ? 'bg-gradient-to-r from-indigo-500 to-violet-500'
                                            : 'bg-slate-300'}`}
                                    style={{
                                        width: `${score * 100}%`,
                                        transitionDelay: '200ms'
                                    }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
