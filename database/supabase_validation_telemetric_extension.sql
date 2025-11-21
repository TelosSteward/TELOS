-- ============================================================
-- TELEMETRIC SIGNATURE EXTENSION FOR VALIDATION DATA
-- ============================================================
--
-- This schema extension adds tables for storing cryptographically
-- signed validation data with telemetric signatures for IP protection.
--
-- Purpose:
-- - Store validation sessions with session-level signatures
-- - Store individual turns with per-turn signatures
-- - Enable third-party verification of IP claims
-- - Support counterfactual analysis with full session data
--
-- Note: Full conversation content is allowed here since this is
--       public ShareGPT/research data, not private user data.
-- ============================================================

-- Table 1: Validation Telemetric Sessions (Session-level signatures)
CREATE TABLE IF NOT EXISTS validation_telemetric_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_study_name TEXT NOT NULL,  -- e.g., "dual_pa_comparison", "sharegpt_250"

    -- Telemetric signature fields
    telemetric_signature TEXT NOT NULL,   -- Session-level cryptographic signature
    key_history_hash TEXT NOT NULL,       -- For verification

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Validation metadata
    model_used TEXT,                      -- e.g., "mistral:latest" or "mistral:7b-instruct-q4_0"
    total_turns INTEGER DEFAULT 0,
    ollama_version TEXT,

    -- IP protection metadata
    signature_algorithm TEXT DEFAULT 'HMAC-SHA256-telemetric',
    entropy_sources_count INTEGER DEFAULT 8,
    telos_version TEXT DEFAULT '1.0.0',

    -- Research context
    dataset_source TEXT,                  -- e.g., "ShareGPT", "WildChat", "LMSYS"
    pa_configuration JSONB,               -- PA settings used

    -- Study results
    avg_fidelity REAL,
    intervention_count INTEGER DEFAULT 0,
    drift_detection_count INTEGER DEFAULT 0
);

-- Table 2: Validation Sessions (Full conversation data with per-turn signatures)
CREATE TABLE IF NOT EXISTS validation_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES validation_telemetric_sessions(session_id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Full conversation data (allowed - public ShareGPT data)
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,

    -- TELOS governance metrics
    fidelity_score REAL,
    distance_from_pa REAL,
    baseline_fidelity REAL,
    telos_fidelity REAL,
    fidelity_delta REAL,

    -- Intervention data
    intervention_triggered BOOLEAN DEFAULT FALSE,
    intervention_type TEXT,
    drift_detected BOOLEAN DEFAULT FALSE,
    governance_mode TEXT,                 -- e.g., "stateless", "prompt_only", "telos"

    -- Telemetric signature for THIS turn
    turn_telemetric_signature TEXT NOT NULL,
    entropy_signature TEXT,
    key_rotation_number INTEGER,

    -- Delta telemetry used for signature generation
    delta_t_ms INTEGER,
    embedding_distance REAL,
    user_message_length INTEGER,
    assistant_response_length INTEGER,

    -- Counterfactual branch data (if applicable)
    is_counterfactual_branch BOOLEAN DEFAULT FALSE,
    counterfactual_of_session UUID,       -- Reference to original session
    divergence_point INTEGER,             -- Turn number where branches diverged

    CONSTRAINT unique_validation_turn UNIQUE (session_id, turn_number)
);

-- Table 3: Counterfactual Comparisons
CREATE TABLE IF NOT EXISTS validation_counterfactual_comparisons (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Sessions being compared
    original_session_id UUID NOT NULL REFERENCES validation_telemetric_sessions(session_id),
    counterfactual_session_id UUID NOT NULL REFERENCES validation_telemetric_sessions(session_id),

    -- Comparison metadata
    divergence_turn INTEGER NOT NULL,
    comparison_type TEXT,                 -- e.g., "with_intervention_vs_without"

    -- Aggregated results
    original_avg_fidelity REAL,
    counterfactual_avg_fidelity REAL,
    fidelity_improvement_pct REAL,

    -- Statistical significance
    p_value REAL,
    effect_size REAL,

    CONSTRAINT unique_counterfactual_pair UNIQUE (original_session_id, counterfactual_session_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_validation_session ON validation_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_validation_turn_number ON validation_sessions(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_validation_study ON validation_telemetric_sessions(validation_study_name);
CREATE INDEX IF NOT EXISTS idx_validation_telemetric_sig ON validation_sessions(turn_telemetric_signature);
CREATE INDEX IF NOT EXISTS idx_validation_created ON validation_telemetric_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_validation_governance_mode ON validation_sessions(governance_mode);
CREATE INDEX IF NOT EXISTS idx_counterfactual_original ON validation_counterfactual_comparisons(original_session_id);

-- View: IP Verification for Validation Data
CREATE OR REPLACE VIEW validation_ip_proofs AS
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
    ARRAY_AGG(
        t.turn_telemetric_signature
        ORDER BY t.turn_number
    ) as signature_chain,
    -- Verification metadata
    s.key_history_hash,
    s.signature_algorithm,
    s.entropy_sources_count,
    s.dataset_source,
    s.pa_configuration
FROM validation_telemetric_sessions s
LEFT JOIN validation_sessions t ON s.session_id = t.session_id
GROUP BY
    s.session_id,
    s.validation_study_name,
    s.telemetric_signature,
    s.created_at,
    s.completed_at,
    s.total_turns,
    s.model_used,
    s.avg_fidelity,
    s.key_history_hash,
    s.signature_algorithm,
    s.entropy_sources_count,
    s.dataset_source,
    s.pa_configuration;

-- View: Baseline Comparison Analysis
CREATE OR REPLACE VIEW validation_baseline_comparison AS
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

-- View: Counterfactual Analysis Summary
CREATE OR REPLACE VIEW validation_counterfactual_summary AS
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
    -- Original branch stats
    (SELECT COUNT(*) FROM validation_sessions
     WHERE session_id = cc.original_session_id) as original_turns,
    -- Counterfactual branch stats
    (SELECT COUNT(*) FROM validation_sessions
     WHERE session_id = cc.counterfactual_session_id) as counterfactual_turns
FROM validation_counterfactual_comparisons cc
JOIN validation_telemetric_sessions orig_s ON cc.original_session_id = orig_s.session_id
JOIN validation_telemetric_sessions cf_s ON cc.counterfactual_session_id = cf_s.session_id;

-- Enable Row Level Security
ALTER TABLE validation_telemetric_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE validation_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE validation_counterfactual_comparisons ENABLE ROW LEVEL SECURITY;

-- Policy: Research team read access (authenticated users)
CREATE POLICY IF NOT EXISTS "Research read validation telemetric"
ON validation_telemetric_sessions FOR SELECT
USING (true);  -- Public research data, readable by anyone

CREATE POLICY IF NOT EXISTS "Research read validation sessions"
ON validation_sessions FOR SELECT
USING (true);  -- Public research data, readable by anyone

CREATE POLICY IF NOT EXISTS "Research read counterfactual"
ON validation_counterfactual_comparisons FOR SELECT
USING (true);

-- Policy: App can insert validation data (service role)
CREATE POLICY IF NOT EXISTS "App insert validation telemetric"
ON validation_telemetric_sessions FOR INSERT
WITH CHECK (true);  -- Service role can insert

CREATE POLICY IF NOT EXISTS "App insert validation sessions"
ON validation_sessions FOR INSERT
WITH CHECK (true);

CREATE POLICY IF NOT EXISTS "App insert counterfactual"
ON validation_counterfactual_comparisons FOR INSERT
WITH CHECK (true);

-- Policy: App can update validation data (service role only)
CREATE POLICY IF NOT EXISTS "App update validation telemetric"
ON validation_telemetric_sessions FOR UPDATE
USING (true);

-- Function: Calculate validation study statistics
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
    -- Get stateless baseline fidelity
    SELECT AVG(t.fidelity_score) INTO stateless_fidelity
    FROM validation_telemetric_sessions s
    JOIN validation_sessions t ON s.session_id = t.session_id
    WHERE s.validation_study_name = study_name_param
      AND t.governance_mode = 'stateless';

    -- Return comparison statistics
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

-- Trigger: Update session statistics on turn insert
CREATE OR REPLACE FUNCTION update_validation_session_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE validation_telemetric_sessions
    SET
        total_turns = (
            SELECT COUNT(*)
            FROM validation_sessions
            WHERE session_id = NEW.session_id
        ),
        avg_fidelity = (
            SELECT AVG(fidelity_score)
            FROM validation_sessions
            WHERE session_id = NEW.session_id
        ),
        intervention_count = (
            SELECT COUNT(*)
            FROM validation_sessions
            WHERE session_id = NEW.session_id AND intervention_triggered = TRUE
        ),
        drift_detection_count = (
            SELECT COUNT(*)
            FROM validation_sessions
            WHERE session_id = NEW.session_id AND drift_detected = TRUE
        )
    WHERE session_id = NEW.session_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_validation_stats
AFTER INSERT ON validation_sessions
FOR EACH ROW
EXECUTE FUNCTION update_validation_session_stats();

-- ============================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================

COMMENT ON TABLE validation_telemetric_sessions IS
'Stores session-level telemetric signatures for IP protection. Each session represents a complete validation run with cryptographic proof.';

COMMENT ON TABLE validation_sessions IS
'Stores individual turns with full conversation content and per-turn telemetric signatures. Allowed to store full content since this is public research data (ShareGPT, etc).';

COMMENT ON TABLE validation_counterfactual_comparisons IS
'Stores counterfactual branch comparisons showing what happens WITH vs WITHOUT TELOS governance. Critical for demonstrating effectiveness.';

COMMENT ON VIEW validation_ip_proofs IS
'Provides complete IP verification data including signature chains for patent/legal documentation.';

COMMENT ON VIEW validation_baseline_comparison IS
'Aggregates statistics across different governance modes for baseline comparison studies.';

COMMENT ON VIEW validation_counterfactual_summary IS
'Summarizes counterfactual analysis results showing governance impact.';

-- ============================================================
-- END SCHEMA EXTENSION
-- ============================================================

-- Quick verification query
SELECT
    'Schema extension created successfully' as status,
    COUNT(*) FILTER (WHERE table_name = 'validation_telemetric_sessions') as telemetric_sessions_table,
    COUNT(*) FILTER (WHERE table_name = 'validation_sessions') as validation_sessions_table,
    COUNT(*) FILTER (WHERE table_name = 'validation_counterfactual_comparisons') as counterfactual_table
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('validation_telemetric_sessions', 'validation_sessions', 'validation_counterfactual_comparisons');
