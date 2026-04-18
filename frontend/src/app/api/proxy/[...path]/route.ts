import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const FALLBACK_BACKEND_URL = "http://127.0.0.1:8080";
const ALLOWED_METHODS = new Set([
  "GET",
  "POST",
  "PUT",
  "PATCH",
  "DELETE",
  "OPTIONS",
  "HEAD",
]);

type RouteContext = {
  params: {
    path?: string[];
  };
};

function getBackendBaseUrl(): string {
  const value = process.env.BACKEND_INTERNAL_URL ?? FALLBACK_BACKEND_URL;
  return value.replace(/\/+$/, "");
}

function getBackendApiKey(): string {
  return (process.env.BACKEND_API_KEY ?? "").trim();
}

function buildTargetUrl(request: NextRequest, pathParts: string[]): string {
  const backendBase = getBackendBaseUrl();
  const joinedPath = pathParts.join("/");
  const search = request.nextUrl.search ?? "";
  return `${backendBase}/${joinedPath}${search}`;
}

async function proxyRequest(
  request: NextRequest,
  context: RouteContext,
): Promise<NextResponse> {
  if (!ALLOWED_METHODS.has(request.method)) {
    return NextResponse.json(
      { detail: `Method ${request.method} not allowed` },
      { status: 405 },
    );
  }

  const pathParts = context.params.path ?? [];
  if (pathParts.length === 0) {
    return NextResponse.json(
      { detail: "Missing upstream API path" },
      { status: 400 },
    );
  }

  const targetUrl = buildTargetUrl(request, pathParts);
  const forwardHeaders = new Headers(request.headers);
  forwardHeaders.delete("host");
  forwardHeaders.delete("content-length");
  forwardHeaders.delete("connection");

  const backendApiKey = getBackendApiKey();
  if (backendApiKey) {
    forwardHeaders.set("authorization", `Bearer ${backendApiKey}`);
  }

  let body: BodyInit | undefined;
  if (request.method !== "GET" && request.method !== "HEAD") {
    const contentType = request.headers.get("content-type") ?? "";
    if (contentType.includes("multipart/form-data")) {
      body = await request.formData();
      // Let fetch set multipart boundaries for forwarded FormData.
      forwardHeaders.delete("content-type");
    } else {
      const rawBody = await request.arrayBuffer();
      if (rawBody.byteLength > 0) {
        body = rawBody;
      }
    }
  }

  try {
    const upstream = await fetch(targetUrl, {
      method: request.method,
      headers: forwardHeaders,
      body,
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
      { detail: `Proxy request failed: ${message}` },
      { status: 502 },
    );
  }
}

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function PUT(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function OPTIONS(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function HEAD(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}
