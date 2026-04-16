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
  token_usage: Record<string, number>;
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
