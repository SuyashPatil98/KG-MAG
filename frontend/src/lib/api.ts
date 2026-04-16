/**
 * KG-MAG — Frontend API Client
 * All backend communication goes through this module.
 * API base URL is configured via NEXT_PUBLIC_API_URL env var.
 */

import type {
  GenerateRequest,
  GenerateResponse,
  IngestResponse,
  KBStatus,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8080";

const API_KEY = process.env.NEXT_PUBLIC_BACKEND_API_KEY ?? "";

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  // Add auth if configured
  if (API_KEY) {
    headers["Authorization"] = `Bearer ${API_KEY}`;
  }

  const res = await fetch(`${BASE_URL}${path}`, {
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

// ── Generation ─────────────────────────────────────────────────────────────────

export async function generateArticle(
  request: GenerateRequest
): Promise<GenerateResponse> {
  return apiFetch<GenerateResponse>("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function getArticle(articleId: string): Promise<any> {
  return apiFetch(`/api/article/${articleId}`);
}

export async function listArticles(): Promise<
  Array<{ article_id: string; title: string; topic: string }>
> {
  return apiFetch("/api/articles");
}

// ── Health ─────────────────────────────────────────────────────────────────────

export async function healthCheck(): Promise<{ status: string }> {
  return apiFetch("/health");
}
