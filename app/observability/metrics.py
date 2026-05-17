from prometheus_client import Counter, Histogram, Gauge
import time

AGENT_RUN_COUNTER = Counter(
    "agent_runs_total",
    "Total agent node executions",
    ["agent", "status"]          # status: success | error
)

AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Agent node execution time",
    ["agent"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60]
)

TOKEN_COUNTER = Counter(
    "llm_tokens_total",
    "Total LLM tokens used",
    ["agent", "token_type"]      # token_type: input | output
)

QUALITY_SCORE = Gauge(
    "run_quality_score",
    "Latest quality score from critic"
)

RETRY_COUNT = Histogram(
    "run_retry_count",
    "Number of critic retries per run",
    buckets=[0, 1, 2, 3]
)

WORKFLOW_DURATION = Histogram(
    "workflow_duration_seconds",
    "End-to-end workflow completion time",
    buckets=[5, 10, 30, 60, 120, 300]
)
