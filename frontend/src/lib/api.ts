/**
 * KG-MAG — Frontend API Client
 * All backend communication goes through this module.
 * Requests go through a same-origin proxy route so backend secrets
 * stay server-side and are never exposed to browsers.
 */

import type {
  DashboardMetrics,
  DeleteUploadsResponse,
  GenerateRequest,
  GenerateResponse,
  GenerationRunLog,
  GeneratedArticle,
  IngestResponse,
  KBStatus,
  RebuildCorpusResponse,
  ResetCorpusResponse,
  UploadListResponse,
} from "./types";

const PROXY_BASE = (
  process.env.NEXT_PUBLIC_API_PROXY_BASE ?? "/api/proxy"
).replace(/\/+$/, "");

async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  const res = await fetch(`${PROXY_BASE}${normalizedPath}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const json = await res.json();
      detail = json.detail ?? detail;
    } catch {}
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

// ── Knowledge Base ─────────────────────────────────────────────────────────────

export async function getKBStatus(): Promise<KBStatus> {
  return apiFetch<KBStatus>("/api/kb/status");
}

export async function ingestDocuments(files: File[]): Promise<IngestResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.name);
  }
  return apiFetch<IngestResponse>("/api/ingest", {
    method: "POST",
    body: form,
  });
}

export async function clearKnowledgeBase(): Promise<void> {
  await apiFetch("/api/kb/clear", { method: "DELETE" });
}

export async function listUploads(): Promise<UploadListResponse> {
  return apiFetch<UploadListResponse>("/api/uploads");
}

export async function deleteUploads(
  storedNames: string[],
): Promise<DeleteUploadsResponse> {
  return apiFetch<DeleteUploadsResponse>("/api/uploads/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stored_names: storedNames }),
  });
}

export async function resetCorpus(options?: {
  delete_uploads?: boolean;
  delete_artifacts?: boolean;
}): Promise<ResetCorpusResponse> {
  return apiFetch<ResetCorpusResponse>("/api/kb/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      delete_uploads: options?.delete_uploads ?? true,
      delete_artifacts: options?.delete_artifacts ?? false,
    }),
  });
}

export async function rebuildCorpus(): Promise<RebuildCorpusResponse> {
  return apiFetch<RebuildCorpusResponse>("/api/kb/rebuild", {
    method: "POST",
  });
}

// ── Generation ─────────────────────────────────────────────────────────────────

export async function generateArticle(
  request: GenerateRequest,
): Promise<GenerateResponse> {
  return apiFetch<GenerateResponse>("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function getArticle(articleId: string): Promise<GeneratedArticle> {
  return apiFetch<GeneratedArticle>(`/api/article/${articleId}`);
}

export async function listArticles(): Promise<
  Array<{ article_id: string; title: string; topic: string }>
> {
  return apiFetch("/api/articles");
}

export async function getDashboardMetrics(): Promise<DashboardMetrics> {
  return apiFetch<DashboardMetrics>("/api/dashboard/metrics");
}

export async function getDashboardLogs(): Promise<GenerationRunLog[]> {
  return apiFetch<GenerationRunLog[]>("/api/dashboard/logs");
}

// ── Health ─────────────────────────────────────────────────────────────────────

export async function healthCheck(): Promise<{ status: string }> {
  return apiFetch("/health");
}
