// Browser-side mirror of the backend response schemas.
//
// These match the Pydantic models in `src/sec_generative_search/api/schemas.py`
// by field name; we deliberately do NOT generate them from OpenAPI because
// the OpenAPI surface is auth-gated and we want types under git review.
// When you change a backend schema, mirror it here in the same commit.

export interface StatusResponse {
  version: string;
  deployment_profile: string;
  embedding_provider: string;
  embedding_model: string;
  storage_filings: number;
  is_admin: boolean;
  persist_provider_credentials: boolean;
}

export interface EdgarIdentityRegisterResponse {
  registered: boolean;
}

export interface FilingSchema {
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  chunk_count: number;
  ingested_at: string;
}

export interface FilingListResponse {
  filings: FilingSchema[];
  total: number;
}

export interface IngestTaskResponse {
  task_id: string;
  status: string;
  websocket_url: string;
}

export interface IngestResultSchema {
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  segment_count: number;
  chunk_count: number;
  duration_seconds: number;
}

export interface TaskProgressSchema {
  current_ticker: string | null;
  current_form_type: string | null;
  step_label: string;
  step_index: number;
  step_total: number;
  filings_done: number;
  filings_total: number;
  filings_skipped: number;
  filings_failed: number;
}

export interface TaskStatusResponse {
  task_id: string;
  status: string;
  tickers: string[];
  form_types: string[];
  progress: TaskProgressSchema;
  results: IngestResultSchema[];
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface TaskListResponse {
  tasks: TaskStatusResponse[];
  total: number;
}
