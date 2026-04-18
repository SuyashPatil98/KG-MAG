// KG-MAG — Frontend TypeScript Types
// Mirror of backend/core/models.py Pydantic schemas

export interface ArticleSection {
  heading: string;
  content: string;
  citations: string[];
  image_url?: string | null;
  image_prompt?: string | null;
}

export interface GeneratedArticle {
  article_id: string;
  topic: string;
  title: string;
  subtitle: string;
  header_image_url?: string | null;
  sections: ArticleSection[];
  conclusion: string;
  citations_map: Record<string, DocumentChunk>;
  seo_keywords: string[];
  tags: string[];
  generated_at: string;
  model_used: string;
  token_usage: TokenUsageSummary | Record<string, unknown>;
}

export interface TokenBreakdownEntry {
  tag: string;
  input_tokens: number;
  output_tokens: number;
  elapsed_s: number;
}

export interface TokenUsageSummary {
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  calls: number;
  breakdown: TokenBreakdownEntry[];
}

export interface DocumentChunk {
  chunk_id: string;
  source_id: string;
  filename: string;
  text: string;
  chunk_index: number;
  heading?: string | null;
  page_number?: number | null;
  token_count?: number | null;
}

export interface GroundingResult {
  sentence: string;
  is_grounded: boolean;
  supporting_chunk_ids: string[];
  confidence: number;
}

export interface QAReport {
  article_id: string;
  grounding_score: number;
  readability_score: number;
  coverage_score: number;
  consistency_score: number;
  overall_confidence: number;
  grounding_details: GroundingResult[];
  warnings: string[];
  passed: boolean;
}

export interface GenerateRequest {
  topic: string;
  target_audience?: string;
  tone?: string;
  generate_images?: boolean;
  run_qa?: boolean;
  max_sections?: number;
}

export interface GenerateResponse {
  article_id: string;
  status: string;
  article?: GeneratedArticle | null;
  qa_report?: QAReport | null;
  duration_seconds: number;
}

export interface IngestResponse {
  job_id: string;
  status: string;
  chunks_created: number;
  documents_processed: number;
  duration_seconds: number;
}

export interface KBStatus {
  total_documents: number;
  total_chunks: number;
  index_built: boolean;
  vector_db: string;
  embedding_model: string;
  last_updated?: string | null;
}

export interface UploadedFileInfo {
  stored_name: string;
  display_name: string;
  size_bytes: number;
  uploaded_at: string;
  chunk_count: number;
  indexed: boolean;
}

export interface UploadListResponse {
  total_files: number;
  total_size_bytes: number;
  files: UploadedFileInfo[];
}

export interface DeleteUploadsResponse {
  deleted: string[];
  not_found: string[];
  rebuild_documents_processed: number;
  rebuild_chunks_indexed: number;
}

export interface ResetCorpusResponse {
  status: string;
  uploads_removed: number;
  artifacts_removed: number;
}

export interface RebuildCorpusResponse {
  status: string;
  documents_processed: number;
  chunks_indexed: number;
}

export interface GenerationRunLog {
  run_id: string;
  topic: string;
  status: string;
  started_at: string;
  duration_seconds: number;
  generate_images: boolean;
  run_qa: boolean;
  stage_timings: Record<string, number>;
  token_usage: Record<string, unknown>;
  image_attempted: number;
  image_generated: number;
  image_failed: number;
  qa_passed?: boolean | null;
  qa_overall_confidence?: number | null;
  qa_grounding_score?: number | null;
  qa_readability_score?: number | null;
  qa_warning_count: number;
  error?: string | null;
}

export interface DashboardMetrics {
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  qa_enabled_runs: number;
  qa_passed_runs: number;
  qa_failed_runs: number;
  avg_duration_seconds: number;
  total_input_tokens: number;
  total_output_tokens: number;
  recent_runs: GenerationRunLog[];
}
