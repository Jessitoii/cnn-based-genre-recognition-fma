'use client';

import React, { useState, useCallback } from 'react';

interface UploadAreaProps {
    onFileSelect: (file: File | null) => void;
    selectedFile: File | null;
}

export default function UploadArea({ onFileSelect, selectedFile }: UploadAreaProps) {
    const [isDragging, setIsDragging] = useState(false);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type === 'audio/mpeg' || file.name.endsWith('.mp3')) {
                onFileSelect(file);
            } else {
                alert('Please upload an MP3 file.');
            }
        }
    }, [onFileSelect]);

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            onFileSelect(e.target.files[0]);
        }
    }, [onFileSelect]);

    const clearFile = (e: React.MouseEvent) => {
        e.stopPropagation();
        onFileSelect(null);
    };

    return (
        <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById('fileInput')?.click()}
            className={`relative w-full p-12 border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer flex flex-col items-center justify-center gap-4
        ${isDragging
                    ? 'border-indigo-500 bg-indigo-50 scale-[1.02]'
                    : selectedFile
                        ? 'border-emerald-500 bg-emerald-50'
                        : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'
                }`}
        >
            <input
                id="fileInput"
                type="file"
                accept=".mp3,audio/mpeg"
                onChange={handleFileInput}
                className="hidden"
            />

            <div className={`p-4 rounded-full ${selectedFile ? 'bg-emerald-100' : 'bg-indigo-100'}`}>
                {selectedFile ? (
                    <svg className="w-8 h-8 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                ) : (
                    <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                )}
            </div>

            <div className="text-center">
                {selectedFile ? (
                    <div className="flex flex-col items-center">
                        <p className="text-lg font-medium text-slate-800 truncate max-w-xs">{selectedFile.name}</p>
                        <p className="text-sm text-slate-500">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                        <button
                            onClick={clearFile}
                            className="mt-4 text-sm text-rose-500 hover:text-rose-600 font-medium transition-colors"
                        >
                            Remove file
                        </button>
                    </div>
                ) : (
                    <>
                        <p className="text-lg font-medium text-slate-700">
                            {isDragging ? 'Drop it here!' : 'Click or drag & drop to upload'}
                        </p>
                        <p className="text-sm text-slate-500 mt-1">MP3 audio files only</p>
                    </>
                )}
            </div>
        </div>
    );
}
