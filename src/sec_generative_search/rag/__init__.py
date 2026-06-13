"""RAG orchestration package.

Public surface — these are the names the API layer, CLI, and tests
import. Internal helpers live behind underscored names in the
respective modules.
"""

from sec_generative_search.rag.citations import (
    ExtractedAnswer,
    extract_citations,
    extract_from_inline_markers,
    extract_from_json_envelope,
)
from sec_generative_search.rag.context import (
    BudgetAllocation,
    ContextBudget,
    build_context_block,
    render_history_block,
)
from sec_generative_search.rag.evaluation import (
    AnswerEvaluationReport,
    AnswerFn,
    CitationScore,
    FaithfulnessScore,
    LatencyStats,
    evaluate_answers,
    score_citations,
    score_faithfulness,
)
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import (
    RAGOrchestrator,
    StreamEvent,
)
from sec_generative_search.rag.prompts import (
    ACTIVE_PROMPT_VERSION,
    PromptTemplate,
    get_template,
)
from sec_generative_search.rag.query_understanding import (
    QUERY_PLAN_JSON_SCHEMA,
    QueryPlan,
    parse_query_plan,
    understand_query,
)

__all__ = [
    "ACTIVE_PROMPT_VERSION",
    "QUERY_PLAN_JSON_SCHEMA",
    "AnswerEvaluationReport",
    "AnswerFn",
    "AnswerMode",
    "BudgetAllocation",
    "CitationScore",
    "ContextBudget",
    "ExtractedAnswer",
    "FaithfulnessScore",
    "LatencyStats",
    "PromptTemplate",
    "QueryPlan",
    "RAGOrchestrator",
    "StreamEvent",
    "build_context_block",
    "evaluate_answers",
    "extract_citations",
    "extract_from_inline_markers",
    "extract_from_json_envelope",
    "get_template",
    "parse_query_plan",
    "render_history_block",
    "score_citations",
    "score_faithfulness",
    "understand_query",
]
