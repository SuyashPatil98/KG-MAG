import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const FALLBACK_BACKEND_URL = "http://127.0.0.1:8080";
const SAFE_FILENAME_RE = /^[A-Za-z0-9._-]+$/;
const ALLOWED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif"]);

function getBackendBaseUrl(): string {
  const value = process.env.BACKEND_INTERNAL_URL ?? FALLBACK_BACKEND_URL;
  return value.replace(/\/+$/, "");
}

function getBackendApiKey(): string {
  return (process.env.BACKEND_API_KEY ?? "").trim();
}

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ filename: string }> },
): Promise<NextResponse> {
  const { filename = "" } = await params;
  if (!filename || !SAFE_FILENAME_RE.test(filename)) {
    return NextResponse.json(
      { detail: "Invalid artifact name" },
      { status: 400 },
    );
  }

  const extension = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXTENSIONS.has(extension)) {
    return NextResponse.json({ detail: "Artifact not found" }, { status: 404 });
  }

  const backendBase = getBackendBaseUrl();
  const targetUrl = `${backendBase}/artifacts/${encodeURIComponent(filename)}`;

  const headers = new Headers();
  const backendApiKey = getBackendApiKey();
  if (backendApiKey) {
    headers.set("authorization", `Bearer ${backendApiKey}`);
  }

  try {
    const upstream = await fetch(targetUrl, {
      method: "GET",
      headers,
      redirect: "follow",
    });

    const responseHeaders = new Headers(upstream.headers);
    responseHeaders.delete("content-encoding");
    responseHeaders.delete("transfer-encoding");

    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown proxy error";
    return NextResponse.json(
      { detail: `Artifact proxy failed: ${message}` },
      { status: 502 },
    );
  }
}
