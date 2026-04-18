"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  deleteUploads,
  getDashboardMetrics,
  getKBStatus,
  listUploads,
  rebuildCorpus,
  resetCorpus,
} from "@/lib/api";
import type {
  DashboardMetrics,
  GenerationRunLog,
  KBStatus,
  UploadListResponse,
} from "@/lib/types";

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

export default function DashboardPage() {
  const [kbStatus, setKbStatus] = useState<KBStatus | null>(null);
  const [uploads, setUploads] = useState<UploadListResponse | null>(null);
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);

  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [includeArtifacts, setIncludeArtifacts] = useState(false);

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isRebuilding, setIsRebuilding] = useState(false);

  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedNames = useMemo(
    () =>
      Object.entries(selected)
        .filter(([, checked]) => checked)
        .map(([name]) => name),
    [selected],
  );

  const loadDashboard = useCallback(async () => {
    setIsRefreshing(true);
    setError(null);
    try {
      const [kb, uploadList, dashMetrics] = await Promise.all([
        getKBStatus(),
        listUploads(),
        getDashboardMetrics(),
      ]);
      setKbStatus(kb);
      setUploads(uploadList);
      setMetrics(dashMetrics);

      // Keep only selections that still exist.
      const validNames = new Set(uploadList.files.map((f) => f.stored_name));
      setSelected((prev) => {
        const next: Record<string, boolean> = {};
        for (const [name, checked] of Object.entries(prev)) {
          if (checked && validNames.has(name)) next[name] = true;
        }
        return next;
      });
    } catch (e: unknown) {
      setError(getErrorMessage(e));
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
    const id = setInterval(loadDashboard, 15000);
    return () => clearInterval(id);
  }, [loadDashboard]);

  const toggleRow = (storedName: string) => {
    setSelected((prev) => ({ ...prev, [storedName]: !prev[storedName] }));
  };

  const toggleAll = () => {
    const files = uploads?.files ?? [];
    const shouldSelectAll = selectedNames.length !== files.length;
    const next: Record<string, boolean> = {};
    for (const file of files) {
      next[file.stored_name] = shouldSelectAll;
    }
    setSelected(next);
  };

  const handleDeleteSelected = async () => {
    if (selectedNames.length === 0) return;
    if (
      !window.confirm(
        `Delete ${selectedNames.length} selected upload(s) and rebuild index?`,
      )
    ) {
      return;
    }

    setIsDeleting(true);
    setMessage(null);
    setError(null);
    try {
      const res = await deleteUploads(selectedNames);
      setMessage(
        `Deleted ${res.deleted.length} upload(s). Rebuilt corpus with ${res.rebuild_documents_processed} docs / ${res.rebuild_chunks_indexed} chunks.`,
      );
      setSelected({});
      await loadDashboard();
    } catch (e: unknown) {
      setError(getErrorMessage(e));
    } finally {
      setIsDeleting(false);
    }
  };

  const handleResetCorpus = async () => {
    if (
      !window.confirm("Reset corpus now? This clears index and uploaded files.")
    ) {
      return;
    }

    setIsResetting(true);
    setMessage(null);
    setError(null);
    try {
      const res = await resetCorpus({
        delete_uploads: true,
        delete_artifacts: includeArtifacts,
      });
      setMessage(
        `Corpus reset complete. Removed ${res.uploads_removed} uploads and ${res.artifacts_removed} artifacts.`,
      );
      setSelected({});
      await loadDashboard();
    } catch (e: unknown) {
      setError(getErrorMessage(e));
    } finally {
      setIsResetting(false);
    }
  };

  const handleRebuildCorpus = async () => {
    setIsRebuilding(true);
    setMessage(null);
    setError(null);
    try {
      const res = await rebuildCorpus();
      setMessage(
        `Corpus ${res.status}. Indexed ${res.documents_processed} documents / ${res.chunks_indexed} chunks.`,
      );
      await loadDashboard();
    } catch (e: unknown) {
      setError(getErrorMessage(e));
    } finally {
      setIsRebuilding(false);
    }
  };

  const recentRuns = metrics?.recent_runs ?? [];

  return (
    <div
      className="min-h-screen bg-[#0a0a0f] text-[#e8e6f0]"
      style={{ fontFamily: "var(--font-mono)" }}
    >
      <header className="border-b border-[#1e1e2e] bg-[#0d0d1a]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div>
            <p className="text-xs text-[#6b7280] uppercase tracking-widest">
              KG-MAG
            </p>
            <h1 className="text-lg font-bold text-[#c4b5fd]">
              Operations Dashboard
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href="/"
              className="text-xs border border-[#2e2e4e] hover:border-[#7c3aed] px-3 py-1.5 rounded text-[#9ca3af]"
            >
              Back To Generator
            </Link>
            <button
              onClick={loadDashboard}
              disabled={isRefreshing}
              className="text-xs bg-[#7c3aed] hover:bg-[#6d28d9] disabled:bg-[#2e2e4e] px-3 py-1.5 rounded font-bold"
            >
              {isRefreshing ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-8 gap-3">
          <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-3">
            <p className="text-[10px] text-[#6b7280] uppercase">Documents</p>
            <p className="text-lg font-bold text-[#c4b5fd]">
              {kbStatus?.total_documents ?? 0}
            </p>
          </div>
          <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-3">
            <p className="text-[10px] text-[#6b7280] uppercase">Chunks</p>
            <p className="text-lg font-bold text-[#c4b5fd]">
              {(kbStatus?.total_chunks ?? 0).toLocaleString()}
            </p>
          </div>
          <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-3">
            <p className="text-[10px] text-[#6b7280] uppercase">
              Uploaded Files
            </p>
            <p className="text-lg font-bold text-[#c4b5fd]">
              {uploads?.total_files ?? 0}
            </p>
          </div>
          <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-3">
            <p className="text-[10px] text-[#6b7280] uppercase">
              Generation Runs
            </p>
            <p className="text-lg font-bold text-[#c4b5fd]">
              {metrics?.total_runs ?? 0}
            </p>
          </div>
          <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-3">
            <p className="text-[10px] text-[#6b7280] uppercase">Input Tokens</p>
            <p className="text-lg font-bold text-[#c4b5fd]">
              {(metrics?.total_input_tokens ?? 0).toLocaleString()}
            </p>
          </div>
          <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-3">
            <p className="text-[10px] text-[#6b7280] uppercase">
              Output Tokens
            </p>
            <p className="text-lg font-bold text-[#c4b5fd]">
              {(metrics?.total_output_tokens ?? 0).toLocaleString()}
            </p>
          </div>
          <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-3">
            <p className="text-[10px] text-[#6b7280] uppercase">QA Passed</p>
            <p className="text-lg font-bold text-[#86efac]">
              {(metrics?.qa_passed_runs ?? 0).toLocaleString()}
            </p>
          </div>
          <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-3">
            <p className="text-[10px] text-[#6b7280] uppercase">QA Failed</p>
            <p className="text-lg font-bold text-[#fca5a5]">
              {(metrics?.qa_failed_runs ?? 0).toLocaleString()}
            </p>
          </div>
        </section>

        {message && (
          <div className="bg-green-950/40 border border-green-900 text-green-300 text-xs px-3 py-2 rounded">
            {message}
          </div>
        )}
        {error && (
          <div className="bg-red-950/40 border border-red-900 text-red-300 text-xs px-3 py-2 rounded">
            {error}
          </div>
        )}

        <section className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-5">
          <div className="flex items-center justify-between gap-3 mb-4">
            <h2 className="text-xs font-bold text-[#a78bfa] uppercase tracking-widest">
              Upload Management
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={toggleAll}
                className="text-xs border border-[#2e2e4e] hover:border-[#7c3aed] px-2 py-1 rounded"
              >
                {selectedNames.length === (uploads?.files.length ?? 0)
                  ? "Unselect All"
                  : "Select All"}
              </button>
              <button
                onClick={handleDeleteSelected}
                disabled={selectedNames.length === 0 || isDeleting}
                className="text-xs bg-[#7c3aed] hover:bg-[#6d28d9] disabled:bg-[#2e2e4e] px-3 py-1 rounded font-bold"
              >
                {isDeleting
                  ? "Deleting..."
                  : `Delete Selected (${selectedNames.length})`}
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-[#6b7280] border-b border-[#1e1e2e]">
                  <th className="py-2 w-10">Pick</th>
                  <th className="py-2">File</th>
                  <th className="py-2">Size</th>
                  <th className="py-2">Indexed Chunks</th>
                  <th className="py-2">Uploaded</th>
                </tr>
              </thead>
              <tbody>
                {(uploads?.files ?? []).map((file) => (
                  <tr
                    key={file.stored_name}
                    className="border-b border-[#1a1a2a]"
                  >
                    <td className="py-2">
                      <input
                        type="checkbox"
                        checked={!!selected[file.stored_name]}
                        onChange={() => toggleRow(file.stored_name)}
                        className="accent-[#7c3aed]"
                      />
                    </td>
                    <td className="py-2 text-[#d1d5db]">{file.display_name}</td>
                    <td className="py-2 text-[#9ca3af]">
                      {formatBytes(file.size_bytes)}
                    </td>
                    <td className="py-2 text-[#9ca3af]">{file.chunk_count}</td>
                    <td className="py-2 text-[#9ca3af]">
                      {new Date(file.uploaded_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
                {(uploads?.files.length ?? 0) === 0 && (
                  <tr>
                    <td colSpan={5} className="py-6 text-center text-[#4b5563]">
                      No uploaded files found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="mt-4 pt-4 border-t border-[#1e1e2e] flex items-center justify-between gap-3">
            <label className="flex items-center gap-2 text-xs text-[#9ca3af]">
              <input
                type="checkbox"
                checked={includeArtifacts}
                onChange={(e) => setIncludeArtifacts(e.target.checked)}
                className="accent-[#7c3aed]"
              />
              Delete artifacts (generated images) too
            </label>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRebuildCorpus}
                disabled={isRebuilding}
                className="text-xs border border-[#2e2e4e] hover:border-[#7c3aed] disabled:bg-[#2e2e4e] px-3 py-1.5 rounded font-bold"
              >
                {isRebuilding ? "Rebuilding..." : "Rebuild From Uploads"}
              </button>
              <button
                onClick={handleResetCorpus}
                disabled={isResetting}
                className="text-xs bg-red-900/70 hover:bg-red-800 disabled:bg-[#2e2e4e] px-3 py-1.5 rounded font-bold"
              >
                {isResetting ? "Resetting..." : "Reset Corpus"}
              </button>
            </div>
          </div>
        </section>

        <section className="bg-[#12121f] border border-[#1e1e2e] rounded-lg p-5">
          <h2 className="text-xs font-bold text-[#a78bfa] uppercase tracking-widest mb-4">
            Generation Logs (Content + Images)
          </h2>

          <div className="space-y-3">
            {recentRuns.map((run) => {
              const inputTokens = tokenCount(run, "total_input_tokens");
              const outputTokens = tokenCount(run, "total_output_tokens");
              return (
                <div
                  key={run.run_id}
                  className="bg-[#0d0d1a] border border-[#1e1e2e] rounded p-3"
                >
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
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
                    <span className="text-[#6b7280]">
                      {new Date(run.started_at).toLocaleString()}
                    </span>
                    <span className="text-[#9ca3af]">
                      {run.duration_seconds.toFixed(2)}s
                    </span>
                    <span className="text-[#9ca3af]">
                      In: {inputTokens.toLocaleString()}
                    </span>
                    <span className="text-[#9ca3af]">
                      Out: {outputTokens.toLocaleString()}
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

                  <div className="mt-2 text-[11px] text-[#6b7280] font-mono flex flex-wrap gap-x-3 gap-y-1">
                    {Object.entries(run.stage_timings || {}).map(
                      ([stage, seconds]) => (
                        <span key={stage}>
                          {stage}: {seconds.toFixed(2)}s
                        </span>
                      ),
                    )}
                    {run.run_qa &&
                      run.qa_overall_confidence !== null &&
                      run.qa_overall_confidence !== undefined && (
                        <span>
                          qa_conf:{" "}
                          {(run.qa_overall_confidence * 100).toFixed(1)}%
                        </span>
                      )}
                    {run.run_qa &&
                      run.qa_grounding_score !== null &&
                      run.qa_grounding_score !== undefined && (
                        <span>
                          grounding: {(run.qa_grounding_score * 100).toFixed(1)}
                          %
                        </span>
                      )}
                    {run.run_qa && run.qa_warning_count > 0 && (
                      <span>qa_warnings: {run.qa_warning_count}</span>
                    )}
                  </div>

                  {run.error && (
                    <p className="mt-2 text-xs text-red-400 bg-red-950/30 px-2 py-1 rounded">
                      {run.error}
                    </p>
                  )}
                </div>
              );
            })}
            {recentRuns.length === 0 && (
              <p className="text-xs text-[#4b5563]">
                No generation runs recorded yet.
              </p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
