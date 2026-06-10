CREATE SCHEMA IF NOT EXISTS benchmark_metrics;

CREATE TABLE IF NOT EXISTS benchmark_metrics.musique_completeness (
    task_id text NOT NULL,
    arm_name text NOT NULL,
    run_batch text NOT NULL,
    hops integer,
    evidence_recall double precision,
    evidence_precision double precision,
    complete_evidence_coverage double precision,
    retrieved_count double precision,
    answer_f1 double precision,
    exact_match double precision,
    model_calls integer NOT NULL DEFAULT 0,
    input_tokens bigint NOT NULL DEFAULT 0,
    output_tokens bigint NOT NULL DEFAULT 0,
    total_tokens bigint NOT NULL DEFAULT 0,
    latency_ms bigint NOT NULL DEFAULT 0,
    stop_reason text,
    error text,
    model_name text,
    synced_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (task_id, arm_name, run_batch)
);

CREATE INDEX IF NOT EXISTS idx_musique_completeness_arm
    ON benchmark_metrics.musique_completeness (arm_name);
CREATE INDEX IF NOT EXISTS idx_musique_completeness_hops
    ON benchmark_metrics.musique_completeness (hops);
