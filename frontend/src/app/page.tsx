"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import ArticlePreview from "@/components/ArticlePreview";
import QAReportCard from "@/components/QAReportCard";
import UploadZone from "@/components/UploadZone";
import KBStatus from "@/components/KBStatus";
import {
  deleteUploads,
  generateArticle,
  getDashboardMetrics,
  ingestDocuments,
  listUploads,
  rebuildCorpus,
  resetCorpus,
} from "@/lib/api";
import type {
  DashboardMetrics,
  GenerateResponse,
  GenerationRunLog,
  UploadListResponse,
} from "@/lib/types";

const SESSION_KEY = "kgmag.generator.session.v1";

type GeneratorSessionState = {
  topic: string;
  audience: string;
  tone: string;
  maxSections: number;
  generateImages: boolean;
  runQA: boolean;
  result: GenerateResponse | null;
};

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unexpected error";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function toNumber(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

function tokenCount(log: GenerationRunLog, key: string): number {
  return toNumber(log.token_usage?.[key]);
}

export default function HomePage() {
  const [topic, setTopic] = useState("");
  const [audience, setAudience] = useState("general tech readers");
  const [tone, setTone] = useState("informative and engaging");
  const [maxSections, setMaxSections] = useState(5);
  const [generateImages, setGenerateImages] = useState(true);
  const [runQA, setRunQA] = useState(true);
  const [sessionHydrated, setSessionHydrated] = useState(false);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string[]>([]);

  const [uploads, setUploads] = useState<UploadListResponse | null>(null);
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [selectedUploads, setSelectedUploads] = useState<
    Record<string, boolean>
  >({});
  const [includeArtifacts, setIncludeArtifacts] = useState(false);

  const [opsMessage, setOpsMessage] = useState<string | null>(null);
  const [opsError, setOpsError] = useState<string | null>(null);
  const [isOpsRefreshing, setIsOpsRefreshing] = useState(false);
  const [isDeletingUploads, setIsDeletingUploads] = useState(false);
  const [isResettingCorpus, setIsResettingCorpus] = useState(false);
  const [isRebuildingCorpus, setIsRebuildingCorpus] = useState(false);

  const selectedUploadNames = useMemo(
    () =>
      Object.entries(selectedUploads)
        .filter(([, checked]) => checked)
        .map(([name]) => name),
    [selectedUploads],
  );

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(SESSION_KEY);
      if (!raw) {
        setSessionHydrated(true);
        return;
      }

      const parsed = JSON.parse(raw) as Partial<GeneratorSessionState>;
      if (typeof parsed.topic === "string") setTopic(parsed.topic);
      if (typeof parsed.audience === "string") setAudience(parsed.audience);
      if (typeof parsed.tone === "string") setTone(parsed.tone);
      if (typeof parsed.maxSections === "number")
        setMaxSections(parsed.maxSections);
      if (typeof parsed.generateImages === "boolean")
        setGenerateImages(parsed.generateImages);
      if (typeof parsed.runQA === "boolean") setRunQA(parsed.runQA);
      if (parsed.result && typeof parsed.result === "object") {
        setResult(parsed.result as GenerateResponse);
      }
    } catch {
      // Ignore parse failures and keep defaults.
    } finally {
      setSessionHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!sessionHydrated) return;
    const snapshot: GeneratorSessionState = {
      topic,
      audience,
      tone,
      maxSections,
      generateImages,
      runQA,
      result,
    };
    try {
      window.localStorage.setItem(SESSION_KEY, JSON.stringify(snapshot));
    } catch {
      // Ignore quota errors; generator still functions with in-memory state.
    }
  }, [
    topic,
    audience,
    tone,
    maxSections,
    generateImages,
    runQA,
    result,
    sessionHydrated,
  ]);

  const addProgress = (msg: string) =>
    setProgress((p) => [...p, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const loadOpsData = useCallback(async () => {
    setIsOpsRefreshing(true);
    setOpsError(null);
    try {
      const [uploadList, dashMetrics] = await Promise.all([
        listUploads(),
        getDashboardMetrics(),
      ]);
      setUploads(uploadList);
      setMetrics(dashMetrics);

      const validNames = new Set(uploadList.files.map((f) => f.stored_name));
      setSelectedUploads((prev) => {
        const next: Record<string, boolean> = {};
        for (const [name, checked] of Object.entries(prev)) {
          if (checked && validNames.has(name)) next[name] = true;
        }
        return next;
      });
    } catch (e: unknown) {
      setOpsError(getErrorMessage(e));
    } finally {
      setIsOpsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadOpsData();
    const id = setInterval(loadOpsData, 15000);
    return () => clearInterval(id);
  }, [loadOpsData]);

  const handleIngest = async (files: File[]) => {
    setIsIngesting(true);
    setError(null);
    addProgress(`Uploading ${files.length} document(s)...`);
    try {
      const res = await ingestDocuments(files);
      addProgress(
        `✓ Ingested ${res.documents_processed} docs, ${res.chunks_created} chunks in ${res.duration_seconds}s`,
      );
      await loadOpsData();
    } catch (e: unknown) {
      setError(getErrorMessage(e));
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
      await loadOpsData();
    } catch (e: unknown) {
      const message = getErrorMessage(e);
      setError(message || "Generation failed");
      addProgress(`✗ Error: ${message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const toggleUpload = (storedName: string) => {
    setSelectedUploads((prev) => ({
      ...prev,
      [storedName]: !prev[storedName],
    }));
  };

  const toggleAllUploads = () => {
    const files = uploads?.files ?? [];
    const shouldSelectAll = selectedUploadNames.length !== files.length;
    const next: Record<string, boolean> = {};
    for (const file of files) {
      next[file.stored_name] = shouldSelectAll;
    }
    setSelectedUploads(next);
  };

  const handleDeleteUploads = async () => {
    if (selectedUploadNames.length === 0) return;
    if (
      !window.confirm(
        `Delete ${selectedUploadNames.length} selected upload(s) and rebuild index?`,
      )
    ) {
      return;
    }

    setIsDeletingUploads(true);
    setOpsError(null);
    setOpsMessage(null);
    try {
      const res = await deleteUploads(selectedUploadNames);
      setOpsMessage(
        `Deleted ${res.deleted.length} upload(s). Rebuilt corpus with ${res.rebuild_documents_processed} docs / ${res.rebuild_chunks_indexed} chunks.`,
      );
      setSelectedUploads({});
      await loadOpsData();
    } catch (e: unknown) {
      setOpsError(getErrorMessage(e));
    } finally {
      setIsDeletingUploads(false);
    }
  };

  const handleRebuildCorpus = async () => {
    setIsRebuildingCorpus(true);
    setOpsError(null);
    setOpsMessage(null);
    try {
      const res = await rebuildCorpus();
      setOpsMessage(
        `Corpus ${res.status}. Indexed ${res.documents_processed} documents / ${res.chunks_indexed} chunks.`,
      );
      await loadOpsData();
    } catch (e: unknown) {
      setOpsError(getErrorMessage(e));
    } finally {
      setIsRebuildingCorpus(false);
    }
  };

  const handleResetCorpus = async () => {
    if (
      !window.confirm("Reset corpus now? This clears index and uploaded files.")
    ) {
      return;
    }

    setIsResettingCorpus(true);
    setOpsError(null);
    setOpsMessage(null);
    try {
      const res = await resetCorpus({
        delete_uploads: true,
        delete_artifacts: includeArtifacts,
      });
      setOpsMessage(
        `Corpus reset complete. Removed ${res.uploads_removed} uploads and ${res.artifacts_removed} artifacts.`,
      );
      setSelectedUploads({});
      await loadOpsData();
    } catch (e: unknown) {
      setOpsError(getErrorMessage(e));
    } finally {
      setIsResettingCorpus(false);
    }
  };

  const recentRuns = metrics?.recent_runs ?? [];

  return (
    <div
      className="min-h-screen bg-[#0a0a0f] text-[#e8e6f0]"
      style={{ fontFamily: "'IBM Plex Mono', 'Courier New', monospace" }}
    >
      {/* Header */}
      <header className="border-b border-[#1e1e2e] bg-[#0d0d1a]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-[#7c3aed] rounded flex items-center justify-center text-white font-bold text-sm">
              K
            </div>
            <span className="font-bold text-lg tracking-tight text-[#c4b5fd]">
              KG-MAG
            </span>
            <span className="text-xs text-[#6b7280] hidden sm:block">
              Knowledge-Grounded Article Generator
            </span>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/dashboard"
              className="text-xs border border-[#2e2e4e] hover:border-[#7c3aed] px-3 py-1.5 rounded text-[#9ca3af]"
            >
              Dashboard
            </Link>
            <KBStatus />
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Panel — Controls */}
        <aside className="lg:col-span-1 space-y-6">
          {/* Upload */}
          <section className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-5">
            <h2 className="text-sm font-bold text-[#a78bfa] uppercase tracking-widest mb-4">
              1. Knowledge Base
            </h2>
            <UploadZone onUpload={handleIngest} isLoading={isIngesting} />

            <div className="mt-4 pt-4 border-t border-[#1e1e2e] space-y-3">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs text-[#9ca3af] uppercase tracking-widest">
                  Uploaded Documents
                </p>
                <button
                  onClick={loadOpsData}
                  disabled={isOpsRefreshing}
                  className="text-[11px] border border-[#2e2e4e] hover:border-[#7c3aed] disabled:bg-[#2e2e4e] px-2 py-1 rounded"
                >
                  {isOpsRefreshing ? "Refreshing..." : "Refresh"}
                </button>
              </div>

              <div className="max-h-48 overflow-y-auto rounded border border-[#1e1e2e]">
                {(uploads?.files ?? []).slice(0, 8).map((file) => (
                  <label
                    key={file.stored_name}
                    className="flex items-center justify-between gap-2 px-3 py-2 border-b border-[#1a1a2a] text-xs cursor-pointer"
                  >
                    <span className="flex items-center gap-2 min-w-0">
                      <input
                        type="checkbox"
                        checked={!!selectedUploads[file.stored_name]}
                        onChange={() => toggleUpload(file.stored_name)}
                        className="accent-[#7c3aed]"
                      />
                      <span className="truncate text-[#d1d5db]">
                        {file.display_name}
                      </span>
                    </span>
                    <span className="text-[#6b7280] shrink-0">
                      {formatBytes(file.size_bytes)} · {file.chunk_count}
                    </span>
                  </label>
                ))}
                {(uploads?.files.length ?? 0) === 0 && (
                  <p className="text-xs text-[#4b5563] px-3 py-3">
                    No uploaded files yet.
                  </p>
                )}
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  onClick={toggleAllUploads}
                  className="text-[11px] border border-[#2e2e4e] hover:border-[#7c3aed] px-2 py-1 rounded"
                >
                  {selectedUploadNames.length === (uploads?.files.length ?? 0)
                    ? "Unselect All"
                    : "Select All"}
                </button>
                <button
                  onClick={handleDeleteUploads}
                  disabled={
                    selectedUploadNames.length === 0 || isDeletingUploads
                  }
                  className="text-[11px] bg-[#7c3aed] hover:bg-[#6d28d9] disabled:bg-[#2e2e4e] px-2 py-1 rounded font-bold"
                >
                  {isDeletingUploads
                    ? "Deleting..."
                    : `Delete (${selectedUploadNames.length})`}
                </button>
                <button
                  onClick={handleRebuildCorpus}
                  disabled={isRebuildingCorpus}
                  className="text-[11px] border border-[#2e2e4e] hover:border-[#7c3aed] disabled:bg-[#2e2e4e] px-2 py-1 rounded"
                >
                  {isRebuildingCorpus ? "Rebuilding..." : "Rebuild Index"}
                </button>
              </div>

              <div className="flex items-center justify-between gap-2">
                <label className="flex items-center gap-2 text-[11px] text-[#9ca3af]">
                  <input
                    type="checkbox"
                    checked={includeArtifacts}
                    onChange={(e) => setIncludeArtifacts(e.target.checked)}
                    className="accent-[#7c3aed]"
                  />
                  Delete artifacts too
                </label>
                <button
                  onClick={handleResetCorpus}
                  disabled={isResettingCorpus}
                  className="text-[11px] bg-red-900/70 hover:bg-red-800 disabled:bg-[#2e2e4e] px-2 py-1 rounded font-bold"
                >
                  {isResettingCorpus ? "Resetting..." : "Reset Corpus"}
                </button>
              </div>

              {opsMessage && (
                <p className="text-[11px] text-green-300 bg-green-950/30 border border-green-900 px-2 py-1 rounded">
                  {opsMessage}
                </p>
              )}
              {opsError && (
                <p className="text-[11px] text-red-300 bg-red-950/30 border border-red-900 px-2 py-1 rounded">
                  {opsError}
                </p>
              )}
            </div>
          </section>

          {/* Generate form */}
          <section className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-5 space-y-4">
            <h2 className="text-sm font-bold text-[#a78bfa] uppercase tracking-widest">
              2. Generate Article
            </h2>

            <div>
              <label className="block text-xs text-[#9ca3af] mb-1">
                Topic *
              </label>
              <textarea
                className="w-full bg-[#0d0d1a] border border-[#2e2e4e] rounded px-3 py-2 text-sm text-[#e8e6f0] placeholder-[#4b5563] resize-none focus:outline-none focus:border-[#7c3aed] transition-colors"
                rows={3}
                placeholder="e.g. The impact of transformer architectures on modern NLP..."
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-xs text-[#9ca3af] mb-1">
                Target Audience
              </label>
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
              <label className="block text-xs text-[#9ca3af] mb-1">
                Max Sections: {maxSections}
              </label>
              <input
                type="range"
                min={2}
                max={8}
                step={1}
                className="w-full accent-[#7c3aed]"
                value={maxSections}
                onChange={(e) => setMaxSections(Number(e.target.value))}
              />
            </div>

            <div className="flex gap-4">
              <label className="flex items-center gap-2 text-xs text-[#9ca3af] cursor-pointer">
                <input
                  type="checkbox"
                  checked={generateImages}
                  onChange={(e) => setGenerateImages(e.target.checked)}
                  className="accent-[#7c3aed]"
                />
                Generate images
              </label>
              <label className="flex items-center gap-2 text-xs text-[#9ca3af] cursor-pointer">
                <input
                  type="checkbox"
                  checked={runQA}
                  onChange={(e) => setRunQA(e.target.checked)}
                  className="accent-[#7c3aed]"
                />
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
              <h3 className="text-xs font-bold text-[#6b7280] uppercase tracking-widest mb-3">
                Pipeline Log
              </h3>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {progress.map((msg, i) => (
                  <p key={i} className="text-xs text-[#9ca3af] font-mono">
                    {msg}
                  </p>
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
          <section className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-5">
            <h2 className="text-xs font-bold text-[#a78bfa] uppercase tracking-widest mb-3">
              Runtime QA And Generation Logs
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mb-4">
              <div className="bg-[#0d0d1a] border border-[#1e1e2e] rounded p-2">
                <p className="text-[10px] text-[#6b7280] uppercase">Runs</p>
                <p className="text-sm font-bold text-[#c4b5fd]">
                  {metrics?.total_runs ?? 0}
                </p>
              </div>
              <div className="bg-[#0d0d1a] border border-[#1e1e2e] rounded p-2">
                <p className="text-[10px] text-[#6b7280] uppercase">
                  QA Passed
                </p>
                <p className="text-sm font-bold text-[#86efac]">
                  {metrics?.qa_passed_runs ?? 0}
                </p>
              </div>
              <div className="bg-[#0d0d1a] border border-[#1e1e2e] rounded p-2">
                <p className="text-[10px] text-[#6b7280] uppercase">
                  QA Failed
                </p>
                <p className="text-sm font-bold text-[#fca5a5]">
                  {metrics?.qa_failed_runs ?? 0}
                </p>
              </div>
              <div className="bg-[#0d0d1a] border border-[#1e1e2e] rounded p-2">
                <p className="text-[10px] text-[#6b7280] uppercase">
                  Input Tokens
                </p>
                <p className="text-sm font-bold text-[#c4b5fd]">
                  {(metrics?.total_input_tokens ?? 0).toLocaleString()}
                </p>
              </div>
              <div className="bg-[#0d0d1a] border border-[#1e1e2e] rounded p-2">
                <p className="text-[10px] text-[#6b7280] uppercase">
                  Output Tokens
                </p>
                <p className="text-sm font-bold text-[#c4b5fd]">
                  {(metrics?.total_output_tokens ?? 0).toLocaleString()}
                </p>
              </div>
              <div className="bg-[#0d0d1a] border border-[#1e1e2e] rounded p-2">
                <p className="text-[10px] text-[#6b7280] uppercase">
                  Avg Duration
                </p>
                <p className="text-sm font-bold text-[#c4b5fd]">
                  {(metrics?.avg_duration_seconds ?? 0).toFixed(2)}s
                </p>
              </div>
            </div>

            <div className="space-y-2 max-h-56 overflow-y-auto">
              {recentRuns.slice(0, 6).map((run) => (
                <div
                  key={run.run_id}
                  className="bg-[#0d0d1a] border border-[#1e1e2e] rounded p-3"
                >
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                    <span
                      className={`px-2 py-0.5 rounded ${
                        run.status === "completed"
                          ? "bg-green-950 text-green-400"
                          : "bg-red-950 text-red-400"
                      }`}
                    >
                      {run.status.toUpperCase()}
                    </span>
                    <span className="text-[#d1d5db] font-semibold">
                      {run.topic}
                    </span>
                    <span className="text-[#9ca3af]">
                      {run.duration_seconds.toFixed(2)}s
                    </span>
                    <span className="text-[#9ca3af]">
                      In:{" "}
                      {tokenCount(run, "total_input_tokens").toLocaleString()}
                    </span>
                    <span className="text-[#9ca3af]">
                      Out:{" "}
                      {tokenCount(run, "total_output_tokens").toLocaleString()}
                    </span>
                    <span className="text-[#9ca3af]">
                      Images: {run.image_generated}/{run.image_attempted}
                    </span>
                    {run.run_qa && (
                      <span
                        className={`px-2 py-0.5 rounded ${
                          run.qa_passed
                            ? "bg-green-950 text-green-400"
                            : run.qa_passed === false
                              ? "bg-red-950 text-red-400"
                              : "bg-[#1e1e2e] text-[#9ca3af]"
                        }`}
                      >
                        QA{" "}
                        {run.qa_passed
                          ? "PASSED"
                          : run.qa_passed === false
                            ? "FAILED"
                            : "N/A"}
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-[11px] text-[#6b7280] font-mono flex flex-wrap gap-x-3 gap-y-1">
                    {run.qa_overall_confidence !== null &&
                      run.qa_overall_confidence !== undefined && (
                        <span>
                          qa_conf:{" "}
                          {(run.qa_overall_confidence * 100).toFixed(1)}%
                        </span>
                      )}
                    {run.qa_grounding_score !== null &&
                      run.qa_grounding_score !== undefined && (
                        <span>
                          grounding: {(run.qa_grounding_score * 100).toFixed(1)}
                          %
                        </span>
                      )}
                    {run.qa_warning_count > 0 && (
                      <span>qa_warnings: {run.qa_warning_count}</span>
                    )}
                  </div>
                </div>
              ))}
              {recentRuns.length === 0 && (
                <p className="text-xs text-[#4b5563]">
                  No generation runs yet.
                </p>
              )}
            </div>
          </section>

          {!result && !isGenerating && (
            <div className="border border-dashed border-[#1e1e2e] rounded-lg p-16 text-center">
              <div className="text-4xl mb-4 opacity-30">✦</div>
              <p className="text-[#4b5563] text-sm">
                Upload documents to build your knowledge base,
                <br />
                then enter a topic to generate a grounded article.
              </p>
            </div>
          )}

          {isGenerating && (
            <div className="border border-[#1e1e2e] rounded-lg p-16 text-center">
              <div className="text-4xl mb-4 animate-pulse text-[#7c3aed]">
                ◈
              </div>
              <p className="text-[#9ca3af] text-sm">
                Multi-agent pipeline running...
              </p>
              <p className="text-[#4b5563] text-xs mt-2">
                Planner → Retriever → Writer → Critic
              </p>
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
