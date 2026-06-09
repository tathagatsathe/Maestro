from prometheus_client import Counter, Gauge, Histogram

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

QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Number of jobs waiting in the Redis queue",
)

WORKER_ACTIVE_JOBS = Gauge(
    "worker_active_jobs",
    "Number of jobs currently being processed by workers",
)

JOB_DURATION = Histogram(
    "job_duration_seconds",
    "Worker job processing time",
    buckets=[5, 10, 30, 60, 120, 300, 600]
)

SEARCH_FALLBACK_TOTAL = Counter(
    "search_fallback_total",
    "Search fallbacks when primary source is unavailable",
    ["source"],
)

RUN_COST_USD = Histogram(
    "run_estimated_cost_usd",
    "Estimated LLM cost per completed run in USD",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

EVAL_PASS_RATE = Gauge(
    "eval_pass_rate",
    "Fraction of golden eval tasks that passed the latest run",
)
