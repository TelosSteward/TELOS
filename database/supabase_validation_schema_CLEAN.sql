-- Validation Telemetric Sessions
CREATE TABLE validation_telemetric_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_study_name TEXT NOT NULL,
    telemetric_signature TEXT NOT NULL,
    key_history_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    model_used TEXT,
    total_turns INTEGER DEFAULT 0,
    ollama_version TEXT,
    signature_algorithm TEXT DEFAULT 'HMAC-SHA256-telemetric',
    entropy_sources_count INTEGER DEFAULT 8,
    telos_version TEXT DEFAULT '1.0.0',
    dataset_source TEXT,
    pa_configuration JSONB,
    basin_constant REAL DEFAULT 1.0,
    constraint_tolerance REAL DEFAULT 0.05,
    avg_fidelity REAL,
    intervention_count INTEGER DEFAULT 0,
    drift_detection_count INTEGER DEFAULT 0
);

-- Validation Sessions
CREATE TABLE validation_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES validation_telemetric_sessions(session_id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    fidelity_score REAL,
    distance_from_pa REAL,
    baseline_fidelity REAL,
    telos_fidelity REAL,
    fidelity_delta REAL,
    intervention_triggered BOOLEAN DEFAULT FALSE,
    intervention_type TEXT,
    drift_detected BOOLEAN DEFAULT FALSE,
    governance_mode TEXT,
    turn_telemetric_signature TEXT NOT NULL,
    entropy_signature TEXT,
    key_rotation_number INTEGER,
    delta_t_ms INTEGER,
    embedding_distance REAL,
    user_message_length INTEGER,
    assistant_response_length INTEGER,
    is_counterfactual_branch BOOLEAN DEFAULT FALSE,
    counterfactual_of_session UUID,
    divergence_point INTEGER,
    CONSTRAINT unique_validation_turn UNIQUE (session_id, turn_number)
);

-- Counterfactual Comparisons
CREATE TABLE validation_counterfactual_comparisons (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    original_session_id UUID NOT NULL REFERENCES validation_telemetric_sessions(session_id),
    counterfactual_session_id UUID NOT NULL REFERENCES validation_telemetric_sessions(session_id),
    divergence_turn INTEGER NOT NULL,
    comparison_type TEXT,
    original_avg_fidelity REAL,
    counterfactual_avg_fidelity REAL,
    fidelity_improvement_pct REAL,
    p_value REAL,
    effect_size REAL,
    CONSTRAINT unique_counterfactual_pair UNIQUE (original_session_id, counterfactual_session_id)
);

-- Indexes
CREATE INDEX idx_validation_session ON validation_sessions(session_id);
CREATE INDEX idx_validation_turn_number ON validation_sessions(session_id, turn_number);
CREATE INDEX idx_validation_study ON validation_telemetric_sessions(validation_study_name);
CREATE INDEX idx_validation_telemetric_sig ON validation_sessions(turn_telemetric_signature);
CREATE INDEX idx_validation_created ON validation_telemetric_sessions(created_at);
CREATE INDEX idx_validation_governance_mode ON validation_sessions(governance_mode);
CREATE INDEX idx_counterfactual_original ON validation_counterfactual_comparisons(original_session_id);

-- IP Verification View
CREATE VIEW validation_ip_proofs AS
SELECT
    s.session_id,
    s.validation_study_name,
    s.telemetric_signature as session_signature,
    s.created_at,
    s.completed_at,
    s.total_turns,
    s.model_used,
    s.avg_fidelity,
    COUNT(t.id) as signed_turns,
    ARRAY_AGG(t.turn_telemetric_signature ORDER BY t.turn_number) as signature_chain,
    s.key_history_hash,
    s.signature_algorithm,
    s.entropy_sources_count,
    s.dataset_source,
    s.pa_configuration
FROM validation_telemetric_sessions s
LEFT JOIN validation_sessions t ON s.session_id = t.session_id
GROUP BY s.session_id, s.validation_study_name, s.telemetric_signature, s.created_at,
         s.completed_at, s.total_turns, s.model_used, s.avg_fidelity, s.key_history_hash,
         s.signature_algorithm, s.entropy_sources_count, s.dataset_source, s.pa_configuration;

-- Baseline Comparison View
CREATE VIEW validation_baseline_comparison AS
SELECT
    s.validation_study_name,
    t.governance_mode,
    COUNT(*) as turn_count,
    AVG(t.fidelity_score) as avg_fidelity,
    STDDEV(t.fidelity_score) as fidelity_stddev,
    MIN(t.fidelity_score) as min_fidelity,
    MAX(t.fidelity_score) as max_fidelity,
    SUM(CASE WHEN t.intervention_triggered THEN 1 ELSE 0 END) as intervention_count,
    SUM(CASE WHEN t.drift_detected THEN 1 ELSE 0 END) as drift_count
FROM validation_telemetric_sessions s
JOIN validation_sessions t ON s.session_id = t.session_id
WHERE t.governance_mode IS NOT NULL
GROUP BY s.validation_study_name, t.governance_mode;

-- Counterfactual Summary View
CREATE VIEW validation_counterfactual_summary AS
SELECT
    cc.id as comparison_id,
    cc.divergence_turn,
    cc.comparison_type,
    orig_s.validation_study_name,
    orig_s.model_used,
    cc.original_avg_fidelity,
    cc.counterfactual_avg_fidelity,
    cc.fidelity_improvement_pct,
    cc.p_value,
    cc.effect_size,
    (SELECT COUNT(*) FROM validation_sessions WHERE session_id = cc.original_session_id) as original_turns,
    (SELECT COUNT(*) FROM validation_sessions WHERE session_id = cc.counterfactual_session_id) as counterfactual_turns
FROM validation_counterfactual_comparisons cc
JOIN validation_telemetric_sessions orig_s ON cc.original_session_id = orig_s.session_id
JOIN validation_telemetric_sessions cf_s ON cc.counterfactual_session_id = cf_s.session_id;

-- Statistics Function
CREATE OR REPLACE FUNCTION calculate_validation_statistics(study_name_param TEXT)
RETURNS TABLE (
    governance_mode TEXT,
    avg_fidelity REAL,
    fidelity_improvement_vs_stateless_pct REAL,
    intervention_rate REAL,
    drift_rate REAL
) AS $$
DECLARE
    stateless_fidelity REAL;
BEGIN
    SELECT AVG(t.fidelity_score) INTO stateless_fidelity
    FROM validation_telemetric_sessions s
    JOIN validation_sessions t ON s.session_id = t.session_id
    WHERE s.validation_study_name = study_name_param
      AND t.governance_mode = 'stateless';

    RETURN QUERY
    SELECT
        t.governance_mode,
        AVG(t.fidelity_score)::REAL as avg_fidelity,
        (CASE
            WHEN stateless_fidelity > 0
            THEN ((AVG(t.fidelity_score) - stateless_fidelity) / stateless_fidelity * 100)::REAL
            ELSE 0::REAL
         END) as fidelity_improvement_vs_stateless_pct,
        (SUM(CASE WHEN t.intervention_triggered THEN 1 ELSE 0 END)::REAL / COUNT(*)::REAL) as intervention_rate,
        (SUM(CASE WHEN t.drift_detected THEN 1 ELSE 0 END)::REAL / COUNT(*)::REAL) as drift_rate
    FROM validation_telemetric_sessions s
    JOIN validation_sessions t ON s.session_id = t.session_id
    WHERE s.validation_study_name = study_name_param
    GROUP BY t.governance_mode;
END;
$$ LANGUAGE plpgsql;

-- Auto-update Session Stats Trigger
CREATE OR REPLACE FUNCTION update_validation_session_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE validation_telemetric_sessions
    SET
        total_turns = (SELECT COUNT(*) FROM validation_sessions WHERE session_id = NEW.session_id),
        avg_fidelity = (SELECT AVG(fidelity_score) FROM validation_sessions WHERE session_id = NEW.session_id),
        intervention_count = (SELECT COUNT(*) FROM validation_sessions WHERE session_id = NEW.session_id AND intervention_triggered = TRUE),
        drift_detection_count = (SELECT COUNT(*) FROM validation_sessions WHERE session_id = NEW.session_id AND drift_detected = TRUE)
    WHERE session_id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_validation_stats
AFTER INSERT ON validation_sessions
FOR EACH ROW
EXECUTE FUNCTION update_validation_session_stats();
