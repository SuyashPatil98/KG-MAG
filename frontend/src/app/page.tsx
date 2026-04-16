"use client";

import { useState, useRef } from "react";
import ArticlePreview from "@/components/ArticlePreview";
import QAReportCard from "@/components/QAReportCard";
import UploadZone from "@/components/UploadZone";
import KBStatus from "@/components/KBStatus";
import { generateArticle, ingestDocuments } from "@/lib/api";
import type { GenerateResponse } from "@/lib/types";

export default function HomePage() {
  const [topic, setTopic] = useState("");
  const [audience, setAudience] = useState("general tech readers");
  const [tone, setTone] = useState("informative and engaging");
  const [maxSections, setMaxSections] = useState(5);
  const [generateImages, setGenerateImages] = useState(true);
  const [runQA, setRunQA] = useState(true);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string[]>([]);

  const addProgress = (msg: string) =>
    setProgress((p) => [...p, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const handleIngest = async (files: File[]) => {
    setIsIngesting(true);
    setError(null);
    addProgress(`Uploading ${files.length} document(s)...`);
    try {
      const res = await ingestDocuments(files);
      addProgress(
        `✓ Ingested ${res.documents_processed} docs, ${res.chunks_created} chunks in ${res.duration_seconds}s`
      );
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsIngesting(false);
    }
  };

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    setIsGenerating(true);
    setError(null);
    setResult(null);
    setProgress([]);
    addProgress("Starting article generation pipeline...");
    addProgress("→ PlannerAgent: crafting outline...");

    try {
      const res = await generateArticle({
        topic,
        target_audience: audience,
        tone,
        max_sections: maxSections,
        generate_images: generateImages,
        run_qa: runQA,
      });
      addProgress("→ RetrieverAgent: fetching relevant chunks...");
      addProgress("→ WriterAgent: drafting sections...");
      if (runQA) addProgress("→ CriticAgent: running QA checks...");
      addProgress(`✓ Article generated in ${res.duration_seconds}s`);
      setResult(res);
    } catch (e: any) {
      setError(e.message ?? "Generation failed");
      addProgress(`✗ Error: ${e.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-[#e8e6f0]" style={{ fontFamily: "'IBM Plex Mono', 'Courier New', monospace" }}>
      {/* Header */}
      <header className="border-b border-[#1e1e2e] bg-[#0d0d1a]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-[#7c3aed] rounded flex items-center justify-center text-white font-bold text-sm">K</div>
            <span className="font-bold text-lg tracking-tight text-[#c4b5fd]">KG-MAG</span>
            <span className="text-xs text-[#6b7280] hidden sm:block">Knowledge-Grounded Article Generator</span>
          </div>
          <KBStatus />
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* Left Panel — Controls */}
        <aside className="lg:col-span-1 space-y-6">

          {/* Upload */}
          <section className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-5">
            <h2 className="text-sm font-bold text-[#a78bfa] uppercase tracking-widest mb-4">1. Knowledge Base</h2>
            <UploadZone onUpload={handleIngest} isLoading={isIngesting} />
          </section>

          {/* Generate form */}
          <section className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-5 space-y-4">
            <h2 className="text-sm font-bold text-[#a78bfa] uppercase tracking-widest">2. Generate Article</h2>

            <div>
              <label className="block text-xs text-[#9ca3af] mb-1">Topic *</label>
              <textarea
                className="w-full bg-[#0d0d1a] border border-[#2e2e4e] rounded px-3 py-2 text-sm text-[#e8e6f0] placeholder-[#4b5563] resize-none focus:outline-none focus:border-[#7c3aed] transition-colors"
                rows={3}
                placeholder="e.g. The impact of transformer architectures on modern NLP..."
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-xs text-[#9ca3af] mb-1">Target Audience</label>
              <input
                className="w-full bg-[#0d0d1a] border border-[#2e2e4e] rounded px-3 py-2 text-sm text-[#e8e6f0] focus:outline-none focus:border-[#7c3aed] transition-colors"
                value={audience}
                onChange={(e) => setAudience(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-xs text-[#9ca3af] mb-1">Tone</label>
              <select
                className="w-full bg-[#0d0d1a] border border-[#2e2e4e] rounded px-3 py-2 text-sm text-[#e8e6f0] focus:outline-none focus:border-[#7c3aed]"
                value={tone}
                onChange={(e) => setTone(e.target.value)}
              >
                <option>informative and engaging</option>
                <option>technical and precise</option>
                <option>accessible and friendly</option>
                <option>analytical and critical</option>
                <option>inspirational and visionary</option>
              </select>
            </div>

            <div>
              <label className="block text-xs text-[#9ca3af] mb-1">Max Sections: {maxSections}</label>
              <input
                type="range" min={2} max={8} step={1}
                className="w-full accent-[#7c3aed]"
                value={maxSections}
                onChange={(e) => setMaxSections(Number(e.target.value))}
              />
            </div>

            <div className="flex gap-4">
              <label className="flex items-center gap-2 text-xs text-[#9ca3af] cursor-pointer">
                <input type="checkbox" checked={generateImages} onChange={(e) => setGenerateImages(e.target.checked)} className="accent-[#7c3aed]" />
                Generate images
              </label>
              <label className="flex items-center gap-2 text-xs text-[#9ca3af] cursor-pointer">
                <input type="checkbox" checked={runQA} onChange={(e) => setRunQA(e.target.checked)} className="accent-[#7c3aed]" />
                Run QA checks
              </label>
            </div>

            <button
              onClick={handleGenerate}
              disabled={isGenerating || !topic.trim()}
              className="w-full bg-[#7c3aed] hover:bg-[#6d28d9] disabled:bg-[#2e2e4e] disabled:text-[#4b5563] text-white font-bold py-2.5 px-4 rounded text-sm transition-colors flex items-center justify-center gap-2"
            >
              {isGenerating ? (
                <>
                  <span className="animate-spin">⟳</span>
                  Generating...
                </>
              ) : (
                "→ Generate Article"
              )}
            </button>
          </section>

          {/* Progress Log */}
          {progress.length > 0 && (
            <section className="bg-[#0d0d1a] border border-[#1e1e2e] rounded-lg p-4">
              <h3 className="text-xs font-bold text-[#6b7280] uppercase tracking-widest mb-3">Pipeline Log</h3>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {progress.map((msg, i) => (
                  <p key={i} className="text-xs text-[#9ca3af] font-mono">{msg}</p>
                ))}
              </div>
            </section>
          )}

          {error && (
            <div className="bg-red-950/50 border border-red-800 rounded-lg p-4">
              <p className="text-xs text-red-400">✗ {error}</p>
            </div>
          )}
        </aside>

        {/* Right Panel — Output */}
        <section className="lg:col-span-2 space-y-6">
          {!result && !isGenerating && (
            <div className="border border-dashed border-[#1e1e2e] rounded-lg p-16 text-center">
              <div className="text-4xl mb-4 opacity-30">✦</div>
              <p className="text-[#4b5563] text-sm">Upload documents to build your knowledge base,<br/>then enter a topic to generate a grounded article.</p>
            </div>
          )}

          {isGenerating && (
            <div className="border border-[#1e1e2e] rounded-lg p-16 text-center">
              <div className="text-4xl mb-4 animate-pulse text-[#7c3aed]">◈</div>
              <p className="text-[#9ca3af] text-sm">Multi-agent pipeline running...</p>
              <p className="text-[#4b5563] text-xs mt-2">Planner → Retriever → Writer → Critic</p>
            </div>
          )}

          {result && (
            <>
              {result.qa_report && <QAReportCard report={result.qa_report} />}
              {result.article && <ArticlePreview article={result.article} />}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
