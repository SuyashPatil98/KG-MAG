"use client";

import { useEffect, useState } from "react";
import { getKBStatus } from "@/lib/api";
import type { KBStatus as KBStatusType } from "@/lib/types";

export default function KBStatus() {
  const [status, setStatus] = useState<KBStatusType | null>(null);

  useEffect(() => {
    const poll = () => getKBStatus().then(setStatus).catch(() => {});
    poll();
    const id = setInterval(poll, 15000);
    return () => clearInterval(id);
  }, []);

  if (!status) return null;

  return (
    <div className="flex items-center gap-3 text-xs text-[#6b7280]">
      <div className={`w-2 h-2 rounded-full ${status.index_built ? "bg-green-500" : "bg-yellow-500"}`} />
      <span className="hidden sm:block">
        {status.total_chunks.toLocaleString()} chunks · {status.total_documents} docs
      </span>
      <span className="text-[#374151]">{status.vector_db.toUpperCase()}</span>
    </div>
  );
}
