"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";

interface UploadZoneProps {
  onUpload: (files: File[]) => Promise<void>;
  isLoading: boolean;
}

export function UploadZone({ onUpload, isLoading }: UploadZoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => { if (accepted.length) onUpload(accepted); },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "text/markdown": [".md", ".markdown"],
      "text/plain": [".txt"],
    },
    disabled: isLoading,
    multiple: true,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
        isDragActive
          ? "border-[#7c3aed] bg-[#1e1e3e]/50"
          : "border-[#2e2e4e] hover:border-[#7c3aed]/50"
      } ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <input {...getInputProps()} />
      <div className="text-2xl mb-2 text-[#4b5563]">
        {isLoading ? "⟳" : isDragActive ? "↓" : "⬆"}
      </div>
      <p className="text-xs text-[#6b7280]">
        {isLoading
          ? "Ingesting documents..."
          : isDragActive
          ? "Drop files here"
          : "Drop PDFs, Markdown, or .txt files"}
      </p>
      <p className="text-xs text-[#374151] mt-1">or click to browse</p>
    </div>
  );
}

export default UploadZone;
