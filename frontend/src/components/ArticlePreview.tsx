"use client";

import { useState } from "react";
import type { GeneratedArticle } from "@/lib/types";

interface Props {
  article: GeneratedArticle;
}

export default function ArticlePreview({ article }: Props) {
  const [activeTab, setActiveTab] = useState<"preview" | "markdown" | "json">(
    "preview",
  );

  const toNumber = (value: unknown): number => {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
      const n = Number(value);
      return Number.isFinite(n) ? n : 0;
    }
    return 0;
  };

  const resolveImageUrl = (url: string | null | undefined) => {
    if (!url) return null;
    if (url.startsWith("http")) return url;
    if (url.startsWith("/")) return url;
    return `/${url}`;
  };

  const slugify = (value: string) =>
    value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/(^-|-$)/g, "") || "image";

  const buildMarkdown = () => {
    let md = `# ${article.title}\n\n`;
    md += `*${article.subtitle}*\n\n`;
    md += `---\n\n`;
    for (const section of article.sections) {
      md += `## ${section.heading}\n\n`;
      const sectionImageUrl = resolveImageUrl(section.image_url);
      if (sectionImageUrl) {
        md += `![${section.heading}](${sectionImageUrl})\n\n`;
      }
      md += `${section.content}\n\n`;
    }
    md += `## Conclusion\n\n${article.conclusion}\n\n`;
    md += `---\n\n**Tags:** ${article.tags.join(", ")}\n`;
    md += `**Keywords:** ${article.seo_keywords.join(", ")}\n`;
    return md;
  };

  const handleExport = () => {
    const blob = new Blob([buildMarkdown()], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${article.title.replace(/\s+/g, "-").toLowerCase()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadImage = async (
    imageUrl: string | null | undefined,
    label: string,
  ) => {
    const resolved = resolveImageUrl(imageUrl);
    if (!resolved) return;

    const base = `${slugify(article.title)}-${slugify(label)}`;

    try {
      const response = await fetch(resolved);
      if (!response.ok) throw new Error("Failed to fetch image");

      const blob = await response.blob();
      const extension =
        blob.type === "image/jpeg"
          ? "jpg"
          : blob.type === "image/webp"
            ? "webp"
            : "png";

      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = `${base}.${extension}`;
      a.click();
      URL.revokeObjectURL(objectUrl);
      return;
    } catch {
      const a = document.createElement("a");
      a.href = resolved;
      a.download = `${base}.png`;
      a.click();
    }
  };

  const tokenUsage = article.token_usage as Record<string, unknown>;
  const totalTokens =
    toNumber(tokenUsage?.total_input_tokens) +
    toNumber(tokenUsage?.total_output_tokens);

  return (
    <div className="bg-[#12121f] border border-[#1e1e2e] rounded-lg overflow-hidden">
      {/* Article header */}
      <div className="bg-[#0d0d1a] px-6 py-5 border-b border-[#1e1e2e]">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-[#e8e6f0] leading-tight">
              {article.title}
            </h1>
            <p className="text-sm text-[#a78bfa] mt-1 italic">
              {article.subtitle}
            </p>
            <div className="flex flex-wrap gap-2 mt-3">
              {article.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs bg-[#1e1e3e] text-[#818cf8] px-2 py-0.5 rounded-full"
                >
                  #{tag}
                </span>
              ))}
            </div>
          </div>
          <div className="flex gap-2 shrink-0">
            <button
              onClick={handleExport}
              className="text-xs bg-[#1e1e2e] hover:bg-[#2e2e4e] text-[#9ca3af] px-3 py-1.5 rounded transition-colors"
            >
              ↓ Export .md
            </button>
          </div>
        </div>
        {/* Meta stats */}
        <div className="flex gap-4 mt-4 text-xs text-[#4b5563]">
          <span>{article.sections.length} sections</span>
          <span>{totalTokens.toLocaleString()} tokens used</span>
          <span>
            Model: {article.model_used.split("-").slice(0, 2).join("-")}
          </span>
          <span>{Object.keys(article.citations_map).length} sources cited</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-[#1e1e2e]">
        {(["preview", "markdown", "json"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-5 py-2.5 text-xs font-bold uppercase tracking-widest transition-colors ${
              activeTab === tab
                ? "text-[#a78bfa] border-b-2 border-[#7c3aed]"
                : "text-[#4b5563] hover:text-[#9ca3af]"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="p-6 max-h-[70vh] overflow-y-auto">
        {activeTab === "preview" && (
          <article className="prose prose-invert prose-sm max-w-none">
            {article.sections.map((section, i) => (
              <div key={i} className="mb-8">
                {section.image_url && (
                  <figure className="mb-4 rounded-md border border-[#1e1e2e] bg-[#0d0d1a] p-3">
                    <div className="w-full aspect-video overflow-hidden rounded bg-[#090914]">
                      <img
                        src={resolveImageUrl(section.image_url) ?? ""}
                        alt={section.heading}
                        className="w-full h-full object-contain"
                        loading="lazy"
                        decoding="async"
                      />
                    </div>
                    <div className="mt-2 flex items-center justify-between gap-3">
                      <figcaption className="text-xs text-[#6b7280]">
                        Figure: visual explanation for {section.heading}
                      </figcaption>
                      <button
                        onClick={() => {
                          void handleDownloadImage(
                            section.image_url,
                            section.heading,
                          );
                        }}
                        className="text-[11px] border border-[#2e2e4e] hover:border-[#7c3aed] px-2 py-1 rounded text-[#9ca3af]"
                      >
                        Download image
                      </button>
                    </div>
                  </figure>
                )}
                <h2 className="text-base font-bold text-[#c4b5fd] mb-3 font-mono uppercase tracking-wide text-xs">
                  {section.heading}
                </h2>
                <div className="text-sm text-[#d1d5db] leading-relaxed whitespace-pre-line">
                  {section.content}
                </div>
                {section.citations.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-[#1e1e2e]">
                    <p className="text-xs text-[#4b5563]">
                      Sources:{" "}
                      {section.citations.map((id) => (
                        <span
                          key={id}
                          className="text-[#6b7280] mr-2 font-mono"
                        >
                          [{id.slice(0, 8)}…]
                        </span>
                      ))}
                    </p>
                  </div>
                )}
              </div>
            ))}
            <div className="mt-6 pt-6 border-t border-[#1e1e2e]">
              <h2 className="text-xs font-bold text-[#c4b5fd] mb-3 font-mono uppercase tracking-wide">
                Conclusion
              </h2>
              <p className="text-sm text-[#d1d5db] leading-relaxed">
                {article.conclusion}
              </p>
            </div>
          </article>
        )}

        {activeTab === "markdown" && (
          <pre className="text-xs text-[#9ca3af] font-mono whitespace-pre-wrap leading-relaxed">
            {buildMarkdown()}
          </pre>
        )}

        {activeTab === "json" && (
          <pre className="text-xs text-[#9ca3af] font-mono whitespace-pre-wrap leading-relaxed">
            {JSON.stringify(article, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
