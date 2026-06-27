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

// ---------------------------------------------------------------------------
// Providers — catalogue + key validation
// ---------------------------------------------------------------------------

export interface ProviderInfo {
  name: string;
  surface: string;
  supports_upstream_routing: boolean;
}

export interface ProviderListResponse {
  providers: ProviderInfo[];
  total: number;
}

// `pricing_tier` is the lower-case `PricingTier` value the backend
// catalogue is the single source of truth for: "free" | "low" |
// "standard" | "high" | "premium" | "unknown". Typed as a string (not a
// union) so a backend tier rename never breaks the build silently — the
// ModelPicker tooltip (14.6.bis) maps the known values and falls back to
// "unknown" for anything else.
export interface ModelPricing {
  model: string;
  pricing_tier: string;
}

// `GET /api/providers/{provider}/models`. `models` is empty and
// `supports_arbitrary_models` is true for OpenRouter (free-text slugs,
// treated as UNKNOWN pricing).
export interface ProviderModelsResponse {
  provider: string;
  surface: string;
  supports_arbitrary_models: boolean;
  models: ModelPricing[];
  total: number;
}

export interface ProviderValidateResponse {
  valid: boolean;
  provider: string;
  surface: string;
}

// `POST /api/providers/catalogue/refresh` (admin only). Content-free lift
// of the backend `CatalogueRefreshReport`: the source key + public metadata
// URL that was fetched and the aggregate counts written into the additive
// overlay. The backend deliberately omits the overlay filesystem path — no
// model slug, cost figure, or credential ever reaches this surface.
export interface CatalogueRefreshResponse {
  source: string;
  source_url: string;
  provider_count: number;
  model_count: number;
}

// ---------------------------------------------------------------------------
// RAG — plan / query / stream
// ---------------------------------------------------------------------------

export type AnswerMode =
  | "concise"
  | "analytical"
  | "extractive"
  | "comparative";

export interface QueryPlanSchema {
  raw_query: string;
  detected_language: string;
  query_en: string;
  tickers: string[];
  form_types: string[];
  date_range: [string, string] | null;
  intent: string;
  suggested_answer_mode: AnswerMode;
}

export interface RagPlanResponse {
  plan: QueryPlanSchema;
  provider: string;
  model: string;
}

export interface TokenUsageSchema {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface CitationSchema {
  chunk_id: string;
  ticker: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  section_path: string;
  text_span: string;
  similarity: number;
  display_index: number;
}

/**
 * One prior chat turn carried back to the backend in the chat-mode
 * request body. Mirrors `ConversationTurnSchema` in
 * `src/sec_generative_search/api/schemas.py`. Only the user query and
 * model answer survive — retrieved chunks and citations from prior
 * turns are intentionally not on the wire (load-bearing privacy
 * invariant, enforced at the API route lift).
 */
export interface ConversationTurnSchema {
  query: string;
  answer: string;
}

export interface RagStreamFinalPayload {
  answer: string;
  provider: string;
  model: string;
  prompt_version: string;
  token_usage: TokenUsageSchema;
  latency_seconds: number;
  streamed: boolean;
  refused: boolean;
}

/**
 * OpenRouter upstream-provider routing hints — wire shape of the
 * `OpenRouterRoutingHintsSchema` Pydantic model.
 *
 * Only forwarded to the backend on `POST /api/rag/{query,stream}`; only
 * `OpenRouterProvider` consumes it. Supplying hints against any other
 * provider yields HTTP 400 `invalid_flag_combination` at the backend —
 * the SPA's `ModelPicker` hides the routing UI for providers whose
 * `supports_upstream_routing` capability is `false`, so the failure
 * surface should only fire if a future provider drops the capability
 * without the UI being updated.
 *
 * Fields mirror the dataclass exactly. The v1 UI only surfaces
 * `order` + `allow_fallbacks` (matching the CLI's
 * `--openrouter-provider` / `--openrouter-fallbacks` surface); the
 * remaining fields are typed so future UI additions don't need a
 * schema change.
 */
export interface OpenRouterRoutingHintsSchema {
  order?: string[];
  allow_fallbacks?: boolean | null;
  only?: string[];
  ignore?: string[];
  require_parameters?: boolean | null;
  data_collection?: "allow" | "deny" | null;
}

// ---------------------------------------------------------------------------
// User-tier authentication
// ---------------------------------------------------------------------------

/**
 * Response shape of `GET /api/auth/login-params?username=…`.
 * The backend returns the same shape (real or deterministic decoy) for
 * unknown usernames so the wire never enumerates them.
 *
 * `salt_m` is base64url over 16 bytes (22–24 chars).
 */
export interface LoginParamsResponse {
  salt_m: string;
  kdf_algo: string;
  pbkdf2_iterations: number;
}

/** Body for `POST /api/auth/login`. `auth_proof` is 32 base64url bytes. */
export interface LoginRequestBody {
  username: string;
  auth_proof: string;
}

/**
 * Response shape of a successful `POST /api/auth/login`. Both
 * `ciphertext_vault` and `vault_iv` are base64url; the vault decrypts
 * client-side under the KEK derived from the user password.
 */
export interface LoginResponseBody {
  user_id: number;
  username: string;
  ciphertext_vault: string;
  vault_iv: string;
}

/** Body for `POST /api/auth/enrol`. */
export interface EnrolmentCompleteRequestBody {
  token: string;
  salt_m: string;
  auth_proof: string;
  ciphertext_vault: string;
  vault_iv: string;
  kdf_algo: string;
  pbkdf2_iterations: number;
}

/** Response shape of a successful `POST /api/auth/enrol`. */
export interface EnrolmentCompleteResponseBody {
  enrolled: boolean;
  user_id: number;
  username: string;
}

/** Body for `POST /api/auth/password`. */
export interface PasswordChangeRequestBody {
  auth_proof_old: string;
  auth_proof_new: string;
  salt_m: string;
  ciphertext_vault: string;
  vault_iv: string;
  kdf_algo: string;
  pbkdf2_iterations: number;
}

/** Response shape of a successful `POST /api/auth/password`. */
export interface PasswordChangeResponseBody {
  rotated: boolean;
}

/** Body for `POST /api/auth/vault` — re-upload an updated ciphertext. */
export interface VaultUpdateRequestBody {
  ciphertext_vault: string;
  vault_iv: string;
}

/** Response shape of a successful `POST /api/auth/vault`. */
export interface VaultUpdateResponseBody {
  updated: boolean;
}

/** Response shape of `DELETE /api/auth/session`. */
export interface AuthSignOutResponseBody {
  cleared: boolean;
}

/** Body for `POST /api/admin/users`. */
export interface AdminUserCreateRequestBody {
  username: string;
}

/** Response shape of a successful `POST /api/admin/users`. */
export interface AdminUserCreateResponseBody {
  username: string;
  enrolment_token: string;
  expires_at: number;
  enrol_url: string;
}

/** Response shape of `DELETE /api/admin/users/{id}`. */
export interface AdminUserDeleteResponseBody {
  deleted: boolean;
  user_id: number;
}

/** Response shape of `POST /api/admin/users/{id}/unlock`. */
export interface AdminUserUnlockResponseBody {
  unlocked: boolean;
  user_id: number;
}

