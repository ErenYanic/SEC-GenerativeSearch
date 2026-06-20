# SEC-GenerativeSearch

A security-first Retrieval-Augmented Generation (RAG) system for SEC filings (10-K, 10-Q, 8-K, and their amended variants). Grounded financial answers with inline citations, a user-choice LLM provider, and a privacy-preserving on-device embedder by default.

---

## Features

- **Grounded answers with citations** — every claim is tied to a specific chunk from the source filing.
- **User-choice LLM provider** — bring your own key for OpenAI, Anthropic, Gemini, Mistral, DeepSeek, Grok, Qwen, Kimi, MiniMax, MiMo, Z.ai, or OpenRouter.
- **Privacy-preserving by default** — the default embedder (`google/embeddinggemma-300m`) runs on-device; no query or filing chunk leaves your machine during retrieval.
- **Three deployment profiles** — Local (single user), Team (shared server), Cloud (GCP Cloud Run, internet-facing).
- **Full web UI** — Next.js 16 + React 19 SPA with per-request CSP nonce, Trusted Types, and a per-user encrypted vault (server never sees your password or KEK).
- **Security-first architecture** — two-tier API key, per-session EDGAR identity, SQLCipher at rest, sliding-TTL rate limits, and 2 300+ tests with a dedicated `@pytest.mark.security` suite.

---

## Architecture

```text
Fetch (edgartools) → Parse (doc2dict) → Chunk (sentence splitter)
        ↓                    ↓                    ↓
  FilingIdentifier      list[Segment]        list[Chunk]
        ↓
Embed (Provider)  →  Store (ChromaDB + SQLite) → Retrieve → Generate (Provider)
        ↓                      ↓                     ↓           ↓
 np.ndarray(768)      sec_filings collection   RetrievalResult  GenerationResult
                      + metadata registry      + Citation[]     + Citation[]
```

---

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended package manager)
- Node 22 LTS + pnpm 11.2.2 (frontend only)
- `libsqlcipher-dev` system package (optional, for the `[encryption]` extra)

---

## Quick Start — Local Development (Scenario A)

### 1. Clone and install

```bash
git clone https://github.com/ErenYanic/SEC-GenerativeSearch.git
cd SEC-GenerativeSearch
uv pip install -e ".[dev,test,local-embeddings]"
```

### 2. Configure

Create a `.env` file (or export the variables):

```env
EDGAR_IDENTITY_NAME=Your Name
EDGAR_IDENTITY_EMAIL=your@email.com
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
DB_DEPLOYMENT_PROFILE=local
```

The embedder defaults to the on-device `google/embeddinggemma-300m` model — no query or filing chunk is sent to a third party. To use a hosted embedder instead, set `EMBEDDING_PROVIDER=openai` (or `gemini`/`mistral`/`qwen`) and provide the corresponding API key; note that a hosted embedder sends every query and filing chunk to that third party.

### 3. Start the API

```bash
uvicorn sec_generative_search.api.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

### 4. Start the frontend (optional)

```bash
cd frontend
corepack enable && corepack prepare --activate   # one-off
pnpm install
pnpm dev                                          # webpack dev server on :3000
```

Open `http://localhost:3000`. The SPA will prompt for your API and admin keys via `WelcomeGate`.

---

## CLI Reference

The `sec-rag` entry point exposes all operator workflows:

```bash
# Ingest filings
sec-rag ingest add AAPL --form 10-K --count 3
sec-rag ingest batch tickers.txt --form 10-Q

# Manage the corpus
sec-rag manage list
sec-rag manage status
sec-rag manage remove AAPL
sec-rag manage clear -y          # bypasses API_DEMO_MODE

# Semantic search
sec-rag search "capital expenditure trends" --ticker AAPL

# RAG query (non-interactive)
sec-rag rag query "What were Apple's main risk factors in 2024?"

# RAG chat (interactive REPL)
sec-rag rag chat

# Provider management
sec-rag provider list
sec-rag provider validate anthropic
sec-rag provider set anthropic    # prompts for key; never via --flag

# Evaluation
sec-rag evaluate retrieval --cases tests/fixtures/retrieval_eval_cases.json

# Operator utilities
sec-rag manage reindex            # re-embed the collection under a new model
sec-rag backup create             # tarball SQLite + ChromaDB
```

Add `--output json` to `search`, `rag query`, `manage list`, `manage status`, `provider list`, `provider validate`, and `evaluate retrieval` for machine-readable output.

---

## Deployment Scenarios

### Scenario A — Local (default)

Auth is disabled; OpenAPI is exposed. SQLCipher is optional. No user accounts — EDGAR identity set via env.

```bash
docker build -f deploy/Dockerfile.api -t sec-gs-api:local .
docker run --rm -p 8000:8000 \
  -e EDGAR_IDENTITY_NAME="Your Name" \
  -e EDGAR_IDENTITY_EMAIL="your@email.com" \
  -v sec_gs_data:/app/data \
  sec-gs-api:local
```

### Scenario B — Team (Compose + nginx)

SQLCipher required. Per-user encrypted vault. EDGAR identity required per session (`API_EDGAR_SESSION_REQUIRED=true`). API key gates all routes.

```bash
# Prepare secrets and TLS material (examples)
mkdir -p deploy/secrets deploy/certs
printf '%s' "$(openssl rand -hex 32)" > deploy/secrets/db_encryption_key
printf '%s' "$(openssl rand -hex 32)" > deploy/secrets/api_auth_pepper
# drop fullchain.pem + privkey.pem into deploy/certs

# Create deploy/.env with your API keys, encryption key path, and deployment profile
docker compose -f deploy/docker-compose.yml up -d --build
```

**Important:** the `api` service must run with exactly one replica — the in-process `TaskManager` mints task IDs in memory.

### Scenario C — Cloud (GCP Cloud Run)

Internet-facing. Browser reaches only the **frontend** (public ingress, GFE-managed TLS). The API service is `internal`-ingress — unreachable from the public internet. Deploy via CI-gated keyless Workload Identity Federation:

```bash
# Apply manifests (substitute PROJECT_ID and REGION)
gcloud run services replace deploy/cloud/api-service.yaml      --region REGION
gcloud run services replace deploy/cloud/frontend-service.yaml --region REGION
gcloud run jobs     replace deploy/cloud/demo-reset-job.yaml   --region REGION
```

Push to `main` triggers the CI-gated `deploy.yml` workflow automatically after all checks pass.

---

## LLM Providers

| Provider   | Chat default                  | Embedding default            |
| ---------- | ----------------------------- | ---------------------------- |
| OpenAI     | `gpt-5.4-mini`                | `text-embedding-3-small`     |
| Anthropic  | `claude-haiku-4-5`            | —                            |
| Gemini     | `gemini-3-flash-preview`      | `text-embedding-004`         |
| Mistral    | `mistral-small-latest`        | `mistral-embed`              |
| DeepSeek   | `deepseek-chat`               | —                            |
| Grok       | `grok-4-1-fast-non-reasoning` | —                            |
| Qwen       | `qwen-turbo`                  | `text-embedding-v3`          |
| Kimi       | `moonshot-v1-32k`             | —                            |
| MiniMax    | `MiniMax-M2.7-highspeed`      | —                            |
| MiMo       | `MiMo-V2-Flash`               | —                            |
| Z.ai       | `glm-4.5-air`                 | —                            |
| OpenRouter | `openai/gpt-5.4-mini`         | —                            |
| Local      | —                             | `google/embeddinggemma-300m` |

Provider keys are never stored in URL params or shell flags. Set them via environment variable, the `sec-rag provider set` prompt, or the SPA Provider Settings page (encrypted in the per-user vault).

---

## Security Design

| Control                                                                  | What it closes                                                                           |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Two-tier API key (`API_KEY` + `API_ADMIN_KEY`)                           | Destructive routes require both keys; a leaked read key cannot probe the delete surface  |
| SQLCipher at rest (`[encryption]` extra)                                 | Database file theft without the key yields no usable data                                |
| Per-user vault (PBKDF2 → HKDF → AES-GCM)                                 | Server never holds your password or KEK; provider keys are encrypted client-side         |
| `sanitize_retrieved_context()` + `<UNTRUSTED_FILING_CONTEXT>` delimiters | Defence-in-depth against prompt injection via filing content                             |
| Per-request CSP nonce + Trusted Types                                    | XSS-to-key-exfiltration; `localStorage`/`sessionStorage` banned project-wide             |
| `LOG_REDACT_QUERIES=true`                                                | Query text and tickers hashed before emission                                            |
| Per-session EDGAR identity (`API_EDGAR_SESSION_REQUIRED=true`)           | Prevents shared-identity rate-limit collapse in multi-tenant deployments                 |
| Content-free correlation IDs and metric labels                           | Ticker, query, and `session_id` never reach a log aggregator or metrics cardinality axis |
| `@pytest.mark.security` test suite (340+ tests)                          | Auth boundaries, citation integrity, prompt-privacy, deployment-artefact lockers         |

---

## Testing

```bash
# Default suite (load tests deselected)
.venv/bin/python -m pytest

# Opt-in load / throughput suite
.venv/bin/python -m pytest -m load

# Lint and format checks
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .

# Frontend
cd frontend
pnpm test
pnpm lint
pnpm typecheck
pnpm build
pnpm audit:ci
```

---

## Optional Extras

| Extra                | Installs                         | When to use                                            |
| -------------------- | -------------------------------- | ------------------------------------------------------ |
| `[encryption]`       | `pysqlcipher3`                   | SQLCipher at rest (Scenarios B/C; requires system lib) |
| `[local-embeddings]` | `sentence-transformers`, `torch` | On-device embedder (default for Scenario A/B/C)        |
| `[metrics]`          | `prometheus-client`              | `GET /api/metrics` OpenMetrics exposition              |

```bash
uv pip install -e ".[dev,test,encryption,local-embeddings,metrics]"
```

---

## Licensing & Usage

**SEC-GenerativeSearch** is licensed under the **Business Source License 1.1 (BSL 1.1)**.

### What you CAN do (Free & Open)

You are completely free to download, modify, and use this project for:

- **Personal and Academic projects.**
- **Non-commercial research.**
- **Temporary Internal Evaluation.** (including running the full stack locally to assess fit)

### What requires a Commercial License

You may not use this software for commercial purposes without an explicit commercial license. This includes:

- **Production Deployment:** Integrating the software into your company's daily operations or active data pipelines.
- **Managed Services (SaaS):** Offering the software to third parties as a hosted service, API, or search platform.

**Future open-source conversion:** every version automatically transitions to the **Apache License 2.0** four years after its specific release date.

For commercial licensing enquiries, contact me directly.
