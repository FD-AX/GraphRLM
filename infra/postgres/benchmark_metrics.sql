CREATE SCHEMA IF NOT EXISTS benchmark_metrics;

CREATE TABLE IF NOT EXISTS benchmark_metrics.benchmark_runs (
    run_id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    experiment_id text NOT NULL,
    run_fingerprint text NOT NULL,
    artifact_path text,
    artifact_hash text,
    benchmark_id text NOT NULL,
    dataset_name text,
    dataset_origin text NOT NULL,
    dataset_subset text,
    dataset_config text,
    dataset_split text,
    requested_revision text,
    resolved_revision text,
    task_id text NOT NULL,
    task_group text,
    task_name text,
    answer_type text,
    benchmark_context_len integer,
    measured_context_tokens integer,
    tokenizer_id text,
    arm_name text NOT NULL,
    execution_path_status text NOT NULL,
    research_eligible boolean NOT NULL DEFAULT false,
    graph_source text,
    graph_builder_class text,
    encoder_class text,
    semantic_graph_used boolean NOT NULL DEFAULT false,
    legacy_hash_frontier_used boolean NOT NULL DEFAULT false,
    root_provider text,
    root_model text,
    root_reasoning_effort text,
    worker_provider text,
    worker_model text,
    recursive_model text,
    prompt_id text,
    prediction text,
    gold_answer jsonb,
    original_score double precision,
    original_scorer_backend text,
    active_score double precision,
    active_scorer_backend text,
    score_revision_id uuid,
    model_rerun_for_score boolean NOT NULL DEFAULT false,
    model_calls integer NOT NULL DEFAULT 0,
    root_calls integer NOT NULL DEFAULT 0,
    worker_calls integer NOT NULL DEFAULT 0,
    recursive_calls integer NOT NULL DEFAULT 0,
    input_tokens bigint NOT NULL DEFAULT 0,
    output_tokens bigint NOT NULL DEFAULT 0,
    total_tokens bigint NOT NULL DEFAULT 0,
    root_tokens bigint NOT NULL DEFAULT 0,
    worker_tokens bigint NOT NULL DEFAULT 0,
    recursive_tokens bigint NOT NULL DEFAULT 0,
    latency_ms bigint,
    graph_build_latency_ms bigint,
    online_query_latency_ms bigint,
    context_amplification double precision,
    root_input_amplification double precision,
    tokens_per_correct_answer double precision,
    artifact_valid boolean NOT NULL DEFAULT false,
    invalid_reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
    fallback_used boolean NOT NULL DEFAULT false,
    response_ids_complete boolean NOT NULL DEFAULT false,
    trace_token_match boolean NOT NULL DEFAULT false,
    labelled_context_leakage boolean NOT NULL DEFAULT false,
    stop_reason text,
    graph_build_id text,
    graph_reused boolean,
    runtime_provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
    graph_metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (experiment_id, run_fingerprint, task_id, arm_name)
);

CREATE TABLE IF NOT EXISTS benchmark_metrics.benchmark_model_calls (
    call_id text PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES benchmark_metrics.benchmark_runs(run_id) ON DELETE CASCADE,
    call_index integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    provider text NOT NULL,
    model_name text NOT NULL,
    model_role text NOT NULL,
    purpose text,
    reasoning_effort text,
    request_timestamp timestamptz,
    response_timestamp timestamptz,
    latency_ms bigint,
    input_tokens bigint NOT NULL DEFAULT 0,
    output_tokens bigint NOT NULL DEFAULT 0,
    total_tokens bigint NOT NULL DEFAULT 0,
    response_id text,
    response_id_present boolean NOT NULL DEFAULT false,
    fallback_used boolean NOT NULL DEFAULT false,
    batch_start integer,
    batch_size integer,
    records_returned integer,
    trace_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (run_id, call_index)
);

CREATE TABLE IF NOT EXISTS benchmark_metrics.benchmark_score_revisions (
    score_revision_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES benchmark_metrics.benchmark_runs(run_id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    previous_backend text,
    previous_score double precision,
    scorer_backend text NOT NULL,
    scorer_version text,
    score_name text NOT NULL,
    score_value double precision NOT NULL,
    reason text NOT NULL,
    model_rerun boolean NOT NULL DEFAULT false,
    is_official_score boolean NOT NULL DEFAULT false,
    scorer_metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS benchmark_metrics.benchmark_graph_metrics (
    run_id uuid PRIMARY KEY REFERENCES benchmark_metrics.benchmark_runs(run_id) ON DELETE CASCADE,
    semantic_document_count integer,
    entity_snapshot_count integer,
    pair_snapshot_count integer,
    observation_count integer,
    aggregate_projection_count integer,
    embedding_count integer,
    embedding_dim integer,
    initial_frontier_size integer,
    mean_frontier_size double precision,
    max_frontier_size integer,
    visited_node_count integer,
    controller_steps integer,
    presented_evidence_count integer,
    duplicate_evidence_count integer,
    worker_label_coverage double precision,
    graph_build_tokens bigint,
    graph_build_model_calls integer,
    controller_payload_tokens_initial integer,
    controller_payload_tokens_max integer,
    repeated_context_tokens integer,
    graph_selected_context_tokens integer,
    full_context_tokens integer,
    context_reduction_ratio double precision,
    graph_answer text,
    rlm_answer text,
    agreement_status text,
    verifier_calls integer,
    extra_metrics jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS benchmark_metrics.benchmark_graph_forensics (
    run_id uuid PRIMARY KEY REFERENCES benchmark_metrics.benchmark_runs(run_id) ON DELETE CASCADE,
    query_semantics text NOT NULL DEFAULT 'unknown',
    execution_mode text NOT NULL DEFAULT 'unknown',
    graph_query_executed boolean NOT NULL DEFAULT false,
    graph_repository_called boolean NOT NULL DEFAULT false,
    uses_local_projection_list boolean NOT NULL DEFAULT false,
    graph_contributed boolean NOT NULL DEFAULT false,
    graph_contribution_outcome text NOT NULL DEFAULT 'neutral',
    graph_contribution_reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
    path_length integer NOT NULL DEFAULT 0,
    graph_query_reason text,
    graph_query_dsl text,
    required_path_found boolean NOT NULL DEFAULT false,
    required_entities_found boolean NOT NULL DEFAULT false,
    required_evidence_set_complete boolean NOT NULL DEFAULT false,
    answer_derived_from_graph_path boolean NOT NULL DEFAULT false,
    graph_rows_returned integer NOT NULL DEFAULT 0,
    matched_node_count integer NOT NULL DEFAULT 0,
    matched_edge_count integer NOT NULL DEFAULT 0,
    evidence_id_count integer NOT NULL DEFAULT 0,
    projection_fidelity double precision,
    semantic_documents_per_source_record double precision,
    prompted_documents_per_unique_source_record double precision,
    cold_total_tokens bigint NOT NULL DEFAULT 0,
    ingestion_tokens bigint NOT NULL DEFAULT 0,
    warm_query_tokens bigint NOT NULL DEFAULT 0,
    forensic_metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS benchmark_metrics.benchmark_factlens_audits (
    run_id uuid PRIMARY KEY REFERENCES benchmark_metrics.benchmark_runs(run_id) ON DELETE CASCADE,
    mode text NOT NULL,
    claim_id text NOT NULL,
    subclaims_total integer NOT NULL DEFAULT 0,
    subclaims_with_evidence integer NOT NULL DEFAULT 0,
    final_verdict_accuracy double precision NOT NULL DEFAULT 0,
    subclaim_recall double precision NOT NULL DEFAULT 0,
    all_required_subclaims_verified boolean NOT NULL DEFAULT false,
    unsupported_subclaim_rate double precision NOT NULL DEFAULT 0,
    unsupported_verdict_rate double precision NOT NULL DEFAULT 0,
    shared_graph_facts integer NOT NULL DEFAULT 0,
    reused_evidence_count integer NOT NULL DEFAULT 0,
    shared_evidence_reuse integer NOT NULL DEFAULT 0,
    cross_subclaim_edges integer NOT NULL DEFAULT 0,
    contradictions_found integer NOT NULL DEFAULT 0,
    complete_evidence_coverage boolean NOT NULL DEFAULT false,
    required_path_found boolean NOT NULL DEFAULT false,
    answer_derived_from_graph_path boolean NOT NULL DEFAULT false,
    tokens_per_verified_subclaim double precision,
    tokens_per_fully_verified_claim double precision,
    fully_verified_claims_per_10k_tokens double precision NOT NULL DEFAULT 0,
    graph_contributed boolean NOT NULL DEFAULT false,
    graph_contribution_outcome text NOT NULL DEFAULT 'neutral',
    evidence_span_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    graph_fact_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    graph_edge_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    audit_metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE benchmark_metrics.benchmark_factlens_audits
    ADD COLUMN IF NOT EXISTS final_verdict_accuracy double precision NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS subclaim_recall double precision NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS all_required_subclaims_verified boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS unsupported_subclaim_rate double precision NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS unsupported_verdict_rate double precision NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS shared_evidence_reuse integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS required_path_found boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS answer_derived_from_graph_path boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS tokens_per_verified_subclaim double precision,
    ADD COLUMN IF NOT EXISTS tokens_per_fully_verified_claim double precision,
    ADD COLUMN IF NOT EXISTS fully_verified_claims_per_10k_tokens double precision NOT NULL DEFAULT 0;

ALTER TABLE benchmark_metrics.benchmark_graph_forensics
    ADD COLUMN IF NOT EXISTS query_semantics text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS execution_mode text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS graph_contributed boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS graph_contribution_outcome text NOT NULL DEFAULT 'neutral',
    ADD COLUMN IF NOT EXISTS graph_contribution_reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS path_length integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS graph_query_reason text,
    ADD COLUMN IF NOT EXISTS graph_query_dsl text,
    ADD COLUMN IF NOT EXISTS required_path_found boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS required_entities_found boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS required_evidence_set_complete boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS answer_derived_from_graph_path boolean NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created_at
    ON benchmark_metrics.benchmark_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_experiment
    ON benchmark_metrics.benchmark_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_task_arm
    ON benchmark_metrics.benchmark_runs(task_id, arm_name);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_bucket
    ON benchmark_metrics.benchmark_runs(benchmark_id, dataset_subset, benchmark_context_len);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_research
    ON benchmark_metrics.benchmark_runs(execution_path_status, artifact_valid)
    WHERE artifact_valid = true;
CREATE INDEX IF NOT EXISTS idx_model_calls_run
    ON benchmark_metrics.benchmark_model_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_model_calls_role
    ON benchmark_metrics.benchmark_model_calls(model_role, model_name);
CREATE INDEX IF NOT EXISTS idx_runs_metadata_gin
    ON benchmark_metrics.benchmark_runs USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_runs_graph_metrics_gin
    ON benchmark_metrics.benchmark_runs USING gin(graph_metrics);
CREATE INDEX IF NOT EXISTS idx_graph_forensics_query
    ON benchmark_metrics.benchmark_graph_forensics(graph_query_executed, uses_local_projection_list);
CREATE INDEX IF NOT EXISTS idx_graph_forensics_execution_mode
    ON benchmark_metrics.benchmark_graph_forensics(execution_mode);
CREATE INDEX IF NOT EXISTS idx_graph_forensics_query_semantics
    ON benchmark_metrics.benchmark_graph_forensics(query_semantics);
CREATE INDEX IF NOT EXISTS idx_factlens_audits_mode
    ON benchmark_metrics.benchmark_factlens_audits(mode);
CREATE INDEX IF NOT EXISTS idx_factlens_audits_outcome
    ON benchmark_metrics.benchmark_factlens_audits(graph_contribution_outcome);

DROP VIEW IF EXISTS benchmark_metrics.v_graph_forensic_summary;
DROP VIEW IF EXISTS benchmark_metrics.v_execution_mode_summary;
DROP VIEW IF EXISTS benchmark_metrics.v_factlens_audit_summary;
DROP VIEW IF EXISTS benchmark_metrics.v_model_role_summary;
DROP VIEW IF EXISTS benchmark_metrics.v_arm_summary;
DROP VIEW IF EXISTS benchmark_metrics.v_research_runs;

CREATE OR REPLACE VIEW benchmark_metrics.v_research_runs AS
SELECT *
FROM benchmark_metrics.benchmark_runs
WHERE artifact_valid = true
  AND labelled_context_leakage = false
  AND trace_token_match = true
  AND fallback_used = false
  AND execution_path_status = 'research';

CREATE OR REPLACE VIEW benchmark_metrics.v_arm_summary AS
SELECT
    experiment_id,
    benchmark_id,
    dataset_subset,
    benchmark_context_len,
    arm_name,
    count(*) AS runs_total,
    avg(active_score) AS mean_score,
    sum(total_tokens) AS total_tokens,
    avg(total_tokens) AS mean_tokens,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY total_tokens) AS p50_tokens,
    avg(latency_ms) AS mean_latency_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
    avg(context_amplification) AS mean_amplification,
    avg(context_amplification) AS mean_context_amplification,
    sum(total_tokens)::double precision / NULLIF(sum(active_score), 0) AS tokens_per_correct
FROM benchmark_metrics.v_research_runs
GROUP BY experiment_id, benchmark_id, dataset_subset, benchmark_context_len, arm_name;

CREATE OR REPLACE VIEW benchmark_metrics.v_model_role_summary AS
SELECT
    r.experiment_id,
    r.arm_name,
    c.model_role,
    c.model_name,
    count(*) AS calls,
    sum(c.input_tokens) AS input_tokens,
    sum(c.output_tokens) AS output_tokens,
    sum(c.total_tokens) AS total_tokens,
    avg(c.latency_ms) AS mean_latency_ms
FROM benchmark_metrics.v_research_runs r
JOIN benchmark_metrics.benchmark_model_calls c ON c.run_id = r.run_id
GROUP BY r.experiment_id, r.arm_name, c.model_role, c.model_name;

CREATE OR REPLACE VIEW benchmark_metrics.v_graph_forensic_summary AS
SELECT
    r.experiment_id,
    r.arm_name,
    f.query_semantics,
    f.execution_mode,
    count(*) AS runs_total,
    sum(CASE WHEN f.graph_query_executed THEN 1 ELSE 0 END) AS graph_query_runs,
    sum(CASE WHEN f.uses_local_projection_list THEN 1 ELSE 0 END) AS local_projection_runs,
    sum(CASE WHEN f.graph_contributed THEN 1 ELSE 0 END) AS graph_contributed_runs,
    sum(CASE WHEN f.graph_contribution_outcome = 'useful' THEN 1 ELSE 0 END) AS graph_useful_runs,
    sum(CASE WHEN f.graph_contribution_outcome = 'insufficient' THEN 1 ELSE 0 END) AS graph_insufficient_runs,
    avg(f.projection_fidelity) AS mean_projection_fidelity,
    avg(f.semantic_documents_per_source_record) AS mean_semantic_documents_per_record,
    sum(f.cold_total_tokens) AS cold_total_tokens,
    sum(f.warm_query_tokens) AS warm_query_tokens
FROM benchmark_metrics.benchmark_runs r
JOIN benchmark_metrics.benchmark_graph_forensics f ON f.run_id = r.run_id
GROUP BY r.experiment_id, r.arm_name, f.query_semantics, f.execution_mode;

CREATE OR REPLACE VIEW benchmark_metrics.v_execution_mode_summary AS
SELECT
    r.experiment_id,
    f.query_semantics,
    f.execution_mode,
    count(*) AS runs_total,
    sum(r.active_score) AS score_sum,
    avg(r.active_score) AS mean_score,
    sum(r.total_tokens) AS total_tokens,
    avg(r.total_tokens) AS mean_tokens,
    sum(CASE WHEN f.graph_query_executed THEN 1 ELSE 0 END) AS graph_query_runs,
    sum(CASE WHEN f.uses_local_projection_list THEN 1 ELSE 0 END) AS local_projection_runs,
    sum(CASE WHEN f.graph_contributed THEN 1 ELSE 0 END) AS graph_contributed_runs,
    sum(CASE WHEN f.graph_contribution_outcome = 'useful' THEN 1 ELSE 0 END) AS graph_useful_runs,
    sum(CASE WHEN f.graph_contribution_outcome = 'insufficient' THEN 1 ELSE 0 END) AS graph_insufficient_runs,
    sum(f.warm_query_tokens) AS warm_query_tokens
FROM benchmark_metrics.benchmark_runs r
JOIN benchmark_metrics.benchmark_graph_forensics f ON f.run_id = r.run_id
WHERE r.artifact_valid = true
  AND r.labelled_context_leakage = false
  AND r.trace_token_match = true
  AND r.fallback_used = false
GROUP BY r.experiment_id, f.query_semantics, f.execution_mode;

CREATE OR REPLACE VIEW benchmark_metrics.v_factlens_audit_summary AS
SELECT
    r.experiment_id,
    a.mode,
    count(*) AS runs_total,
    avg(r.active_score) AS mean_score,
    avg(a.final_verdict_accuracy) AS final_verdict_accuracy,
    avg(a.subclaim_recall) AS subclaim_recall,
    avg(CASE WHEN a.complete_evidence_coverage THEN 1.0 ELSE 0.0 END) AS complete_evidence_coverage_rate,
    avg(a.unsupported_subclaim_rate) AS unsupported_subclaim_rate,
    avg(a.unsupported_verdict_rate) AS unsupported_verdict_rate,
    sum(r.total_tokens) AS total_tokens,
    sum(a.fully_verified_claims_per_10k_tokens) AS fully_verified_claims_per_10k_tokens,
    avg(a.subclaims_total) AS mean_subclaims_total,
    avg(a.subclaims_with_evidence) AS mean_subclaims_with_evidence,
    sum(a.shared_graph_facts) AS shared_graph_facts,
    sum(a.reused_evidence_count) AS reused_evidence_count,
    sum(a.shared_evidence_reuse) AS shared_evidence_reuse,
    sum(a.cross_subclaim_edges) AS cross_subclaim_edges,
    sum(a.contradictions_found) AS contradictions_found,
    sum(CASE WHEN a.complete_evidence_coverage THEN 1 ELSE 0 END) AS complete_evidence_coverage_runs,
    sum(CASE WHEN a.required_path_found THEN 1 ELSE 0 END) AS required_path_found_runs,
    sum(CASE WHEN a.answer_derived_from_graph_path THEN 1 ELSE 0 END) AS answer_derived_from_graph_path_runs,
    sum(CASE WHEN a.graph_contributed THEN 1 ELSE 0 END) AS graph_contributed_runs,
    sum(CASE WHEN a.graph_contribution_outcome = 'useful' THEN 1 ELSE 0 END) AS graph_useful_runs,
    sum(CASE WHEN a.graph_contribution_outcome = 'insufficient' THEN 1 ELSE 0 END) AS graph_insufficient_runs
FROM benchmark_metrics.benchmark_runs r
JOIN benchmark_metrics.benchmark_factlens_audits a ON a.run_id = r.run_id
WHERE r.artifact_valid = true
  AND r.labelled_context_leakage = false
  AND r.trace_token_match = true
  AND r.fallback_used = false
GROUP BY r.experiment_id, a.mode;
