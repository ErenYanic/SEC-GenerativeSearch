"""Configuration management using Pydantic Settings v2.

Hierarchical settings pattern: each domain area is a separate nested
``BaseSettings`` subclass with its own ``env_prefix``.  The root
``Settings`` class composes them all and provides a singleton accessor.

Environment variable mapping (examples):
    EDGAR_IDENTITY_NAME     -> settings.edgar.identity_name
    EMBEDDING_MODEL_NAME    -> settings.embedding.model_name
    LLM_DEFAULT_PROVIDER    -> settings.llm.default_provider
    PROVIDER_TIMEOUT         -> settings.provider.timeout
    RAG_CONTEXT_TOKEN_BUDGET -> settings.rag.context_token_budget
    DB_ENCRYPTION_KEY       -> settings.database.encryption_key
    API_KEY                 -> settings.api.key
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ BEFORE nested BaseSettings classes are
# instantiated as default values in the Settings class body.  Without
# this, EdgarSettings() (which has required fields and no defaults)
# fails because it only searches os.environ — it has no env_file of
# its own.
load_dotenv()


class EdgarSettings(BaseSettings):
    """SEC EDGAR API credentials.

    In web deployments (Scenarios B/C) where ``EDGAR_SESSION_REQUIRED=true``,
    ``identity_name`` and ``identity_email`` may be unset — each user provides
    their own credentials per session via HTTP headers.  The CLI still requires
    them.

    EDGAR rate limiting is handled by edgartools internally (``pyrate_limiter``
    token bucket at 9 req/s by default, configurable via the
    ``EDGAR_RATE_LIMIT_PER_SEC`` env var that edgartools reads directly).
    """

    identity_name: str | None = None
    identity_email: str | None = None

    model_config = SettingsConfigDict(env_prefix="EDGAR_")


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_name: str = "google/embeddinggemma-300m"
    device: str = "auto"  # "cuda", "cpu", or "auto"
    batch_size: int = 32
    idle_timeout_minutes: int = 0  # 0 = disabled; auto-unload model after idle

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")


class ChunkingSettings(BaseSettings):
    """Text chunking configuration."""

    token_limit: int = 500
    tolerance: int = 50

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")


def resolve_encryption_key_from_values(key: str | None, key_file: str | None) -> str | None:
    """Resolve an encryption key from a direct value or file path.

    Enforces mutual exclusion between the two sources and validates the
    file when ``key_file`` is used. Returns the resolved key string, or
    ``None`` if neither source is set.

    Used by both ``DatabaseSettings`` (Pydantic validation) and
    ``MetadataRegistry`` (runtime resolution without re-instantiating
    settings).
    """
    if key and key_file:
        raise ValueError(
            "DB_ENCRYPTION_KEY and DB_ENCRYPTION_KEY_FILE are mutually exclusive. Set only one."
        )

    if key:
        return key

    if key_file:
        key_path = Path(key_file)
        if not key_path.exists():
            raise ValueError(f"DB_ENCRYPTION_KEY_FILE '{key_file}' does not exist.")
        if not key_path.is_file():
            raise ValueError(f"DB_ENCRYPTION_KEY_FILE '{key_file}' is not a file.")
        key_content = key_path.read_text().strip()
        if not key_content:
            raise ValueError(f"DB_ENCRYPTION_KEY_FILE '{key_file}' is empty.")
        return key_content

    return None


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    chroma_path: str = "./data/chroma_db"
    metadata_db_path: str = "./data/metadata.sqlite"
    max_filings: int = 2500

    # SQLCipher encryption key; unset = plain sqlite3 (local dev).
    encryption_key: str | None = None

    # Path to a file containing the SQLCipher encryption key (e.g. Docker
    # secrets at ``/run/secrets/db_encryption_key``). Mutually exclusive with
    # ``encryption_key``. Preferred in production — file contents are not
    # visible in ``/proc/<pid>/environ``.
    encryption_key_file: str | None = None

    # Task history privacy settings.
    task_history_retention_days: int = 0  # 0 = keep indefinitely
    task_history_persist_tickers: bool = False

    model_config = SettingsConfigDict(env_prefix="DB_")

    @model_validator(mode="after")
    def _resolve_encryption_key(self) -> "DatabaseSettings":
        """Resolve ``encryption_key`` from ``encryption_key_file`` if set.

        Delegates to :func:`resolve_encryption_key_from_values` for the
        actual validation and file reading. See that function's docstring
        for the mutual-exclusion and file-validation rules.
        """
        self.encryption_key = resolve_encryption_key_from_values(
            self.encryption_key, self.encryption_key_file
        )
        return self

    @model_validator(mode="after")
    def _validate_paths(self) -> "DatabaseSettings":
        """Validate that database paths resolve within the working directory.

        Prevents path traversal attacks where an attacker controls environment
        variables (e.g. ``DB_METADATA_DB_PATH=../../sensitive/data.sqlite``)
        to write files outside the project directory.

        Checks:
        - Resolved path must be relative to ``Path.cwd()``.
        - No symlinks in any lexical parent directory.  The walk is
          intentionally done over the *lexical* (non-resolved) path — a
          post-``resolve()`` walk never sees symlinks because they have
          already been followed, which would silently pass a symlink
          whose target happens to live inside cwd.
        """
        base_dir = Path.cwd().resolve()
        for field_name in ("chroma_path", "metadata_db_path"):
            raw_value = getattr(self, field_name)
            lexical = Path(raw_value).absolute()
            resolved = Path(raw_value).resolve()

            # Check the resolved path stays within the working directory.
            if not resolved.is_relative_to(base_dir):
                raise ValueError(
                    f"Database path '{field_name}' resolves to "
                    f"'{resolved}' which is outside the project "
                    f"directory '{base_dir}'. Use a relative path "
                    f"within the project directory."
                )

            # Walk up the lexical path and refuse if any existing parent
            # is a symlink.  Stops at cwd once it is reached.
            check = lexical
            while check != check.parent:  # stop at filesystem root
                if check.exists() and check.is_symlink():
                    raise ValueError(
                        f"Database path '{field_name}' contains a "
                        f"symlink at '{check}'. Symlinks are not "
                        f"permitted in database paths for security."
                    )
                if check == base_dir:
                    break
                check = check.parent

        return self


class LLMSettings(BaseSettings):
    """LLM model selection and generation defaults.

    These control which model is used and how it generates responses.
    Provider-level network policy (timeouts, retries) lives in
    ``ProviderSettings``; this class covers model behaviour.
    """

    default_provider: str = "openai"  # provider key registered in ProviderRegistry
    default_model: str | None = None  # None = use the provider's own default
    temperature: float = 0.1  # low temperature for factual SEC analysis
    max_output_tokens: int = 2048
    streaming: bool = True  # prefer streaming responses by default

    model_config = SettingsConfigDict(env_prefix="LLM_")


class ProviderSettings(BaseSettings):
    """Provider-level network and resilience policy.

    Applies to all external LLM/embedding API calls (OpenAI, Anthropic,
    Gemini, etc.).  Per-provider overrides will be supported in the
    provider registry; these are the global defaults.
    """

    timeout: int = 60  # seconds per API call
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # exponential backoff base (seconds)
    circuit_breaker_threshold: int = 5  # consecutive failures before circuit opens
    circuit_breaker_reset: int = 60  # seconds before half-open retry
    cost_tracking_enabled: bool = True  # track token usage and estimated cost

    model_config = SettingsConfigDict(env_prefix="PROVIDER_")


class RAGSettings(BaseSettings):
    """RAG orchestration configuration.

    Controls how retrieval results are assembled into a prompt context
    and how the generation pipeline behaves.  ``SearchSettings`` covers
    the vector-search layer; this class covers the generation layer
    on top of it.
    """

    context_token_budget: int = 6000  # max tokens allocated to retrieved context
    citation_mode: str = "inline"  # "inline" or "footnote"
    default_answer_mode: str = "concise"  # "concise", "analytical", "extractive", "comparative"
    refusal_enabled: bool = True  # refuse when context is insufficient
    chunk_overlap_tokens: int = 50  # overlap between adjacent chunks for context continuity
    chat_history_enabled: bool = False  # session-scoped conversation memory (off by default)
    chat_history_max_turns: int = 10  # max turns retained in session memory

    model_config = SettingsConfigDict(env_prefix="RAG_")


class SearchSettings(BaseSettings):
    """Vector search configuration (retrieval layer)."""

    top_k: int = 5
    min_similarity: float = 0.0

    model_config = SettingsConfigDict(env_prefix="SEARCH_")


class LoggingSettings(BaseSettings):
    """Logging configuration for optional file logging."""

    # Optional file logging (in addition to stdout).
    # Env vars: LOG_FILE_PATH, LOG_FILE_MAX_BYTES, LOG_FILE_BACKUP_COUNT
    path: str | None = None  # unset = stdout only
    max_bytes: int = 10_485_760  # 10 MB
    backup_count: int = 3

    model_config = SettingsConfigDict(env_prefix="LOG_FILE_")


class HuggingFaceSettings(BaseSettings):
    """Hugging Face configuration."""

    token: str | None = None

    model_config = SettingsConfigDict(env_prefix="HUGGING_FACE_")


class ApiSettings(BaseSettings):
    """API server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
    key: str | None = None  # API key; None = auth disabled (local dev)

    # Rate limiting (requests per minute; 0 = disabled)
    rate_limit_search: int = 60
    rate_limit_ingest: int = 10
    rate_limit_delete: int = 30
    rate_limit_general: int = 120

    # Admin key for destructive operations; unset = unrestricted (Scenario A).
    admin_key: str | None = None

    # Per-session EDGAR credentials requirement.
    edgar_session_required: bool = False

    # Demo mode — FIFO eviction, nightly reset banner, "clear all" disabled.
    demo_mode: bool = False
    demo_eviction_buffer: int = 500

    # Task queue size (maximum concurrent + pending ingest tasks).
    max_task_queue_size: int = 5

    # Abuse prevention caps (0 = unlimited/disabled).
    max_tickers_per_request: int = 0
    max_filings_per_request: int = 0
    ingest_cooldown_seconds: int = 0
    max_task_duration_minutes: int = 0

    @field_validator("key", "admin_key", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: str | None) -> str | None:
        return v or None

    model_config = SettingsConfigDict(env_prefix="API_")


class Settings(BaseSettings):
    """Root settings class combining all sections."""

    edgar: EdgarSettings = EdgarSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    provider: ProviderSettings = ProviderSettings()
    rag: RAGSettings = RAGSettings()
    search: SearchSettings = SearchSettings()
    log_file: LoggingSettings = LoggingSettings()
    hugging_face: HuggingFaceSettings = HuggingFaceSettings()
    api: ApiSettings = ApiSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore prefixed env vars handled by nested classes
    )


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the global Settings instance (singleton pattern)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    """Reload settings from environment (mainly for testing)."""
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance
