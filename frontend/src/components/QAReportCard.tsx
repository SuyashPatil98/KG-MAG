"use client";

import type { QAReport } from "@/lib/types";

interface Props {
  report: QAReport;
}

function ScoreBar({ label, score, max = 1, color }: { label: string; score: number; max?: number; color: string }) {
  const pct = Math.round((score / max) * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-[#9ca3af]">{label}</span>
        <span className="font-bold font-mono" style={{ color }}>{max === 100 ? score.toFixed(1) : (score * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-[#1e1e2e] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default function QAReportCard({ report }: Props) {
  const confidence = Math.round(report.overall_confidence * 100);

  return (
    <div className={`bg-[#12121f] border rounded-lg p-5 ${report.passed ? "border-green-900/50" : "border-red-900/50"}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-bold text-[#a78bfa] uppercase tracking-widest">QA Report</h3>
        <div className={`flex items-center gap-2 text-xs font-bold px-3 py-1 rounded-full ${report.passed ? "bg-green-950 text-green-400" : "bg-red-950 text-red-400"}`}>
          {report.passed ? "✓ PASSED" : "✗ FAILED"}
          <span className="opacity-70">({confidence}% confidence)</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <ScoreBar label="Grounding" score={report.grounding_score} color="#818cf8" />
        <ScoreBar label="Coverage" score={report.coverage_score} color="#a78bfa" />
        <ScoreBar label="Consistency" score={report.consistency_score} color="#6ee7b7" />
        <ScoreBar label="Readability (Flesch)" score={report.readability_score} max={100} color="#fbbf24" />
      </div>

      {report.warnings.length > 0 && (
        <div className="mt-4 space-y-1">
          {report.warnings.map((w, i) => (
            <p key={i} className="text-xs text-yellow-500 bg-yellow-950/30 px-3 py-1.5 rounded">
              ⚠ {w}
            </p>
          ))}
        </div>
      )}

      <div className="mt-3 text-xs text-[#4b5563]">
        {report.grounding_details.length} statements verified · {report.grounding_details.filter(d => d.is_grounded).length} grounded
      </div>
    </div>
  );
}
