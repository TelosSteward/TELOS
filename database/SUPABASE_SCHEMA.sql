-- ============================================================
-- TELOS Observatory - Supabase Database Schema
-- Delta-Only Storage (NO Conversation Content)
-- ============================================================
--
-- PRIVACY ARCHITECTURE:
-- - Stores ONLY governance metrics (mathematical measurements)
-- - NEVER stores user messages or AI responses
-- - NEVER stores conversation content
-- - Enables research data collection while preserving privacy
--
-- ============================================================

-- Table 1: Governance Deltas (Per-Turn Metrics)
-- ============================================================
-- Stores mathematical measurements for each conversation turn
-- NO conversation content whatsoever
-- ============================================================

CREATE TABLE governance_deltas (
    -- Primary identification
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    turn_number INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Governance Metrics (Mathematical Only - No Content)
    fidelity_score REAL NOT NULL CHECK (fidelity_score >= 0.0 AND fidelity_score <= 1.0),
    distance_from_pa REAL NOT NULL CHECK (distance_from_pa >= 0.0),
    delta_from_previous REAL,  -- Change from previous turn

    -- Intervention Data
    intervention_triggered BOOLEAN DEFAULT FALSE,
    intervention_type VARCHAR(50),  -- e.g., 'monitor', 'correct', 'intervene', 'escalate'
    intervention_reason TEXT,  -- Brief reason (NO content quotes)

    -- Statistical Process Control Metrics
    spc_lcl REAL,  -- Lower Control Limit
    spc_ucl REAL,  -- Upper Control Limit
    spc_out_of_control BOOLEAN DEFAULT FALSE,

    -- Primacy Attractor Distance Components
    purpose_alignment REAL,
    scope_alignment REAL,
    boundary_alignment REAL,

    -- Session Context (NO content)
    mode VARCHAR(20) CHECK (mode IN ('demo', 'beta', 'open')),
    model_used VARCHAR(100),  -- Which LLM model was used

    -- Indexes for fast queries
    CONSTRAINT unique_turn UNIQUE (session_id, turn_number)
);

-- Indexes for performance
CREATE INDEX idx_session_id ON governance_deltas(session_id);
CREATE INDEX idx_created_at ON governance_deltas(created_at);
CREATE INDEX idx_fidelity_score ON governance_deltas(fidelity_score);
CREATE INDEX idx_intervention ON governance_deltas(intervention_triggered);

-- ============================================================
-- Table 2: Session Summaries (Aggregated Metrics)
-- ============================================================
-- Stores session-level aggregated governance data
-- NO conversation content
-- ============================================================

CREATE TABLE session_summaries (
    -- Primary identification
    session_id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Session Metadata (NO content)
    mode VARCHAR(20) CHECK (mode IN ('demo', 'beta', 'open')),
    total_turns INTEGER DEFAULT 0,
    session_duration_seconds INTEGER,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Aggregated Governance Metrics
    avg_fidelity_score REAL,
    min_fidelity_score REAL,
    max_fidelity_score REAL,
    std_dev_fidelity REAL,

    -- Intervention Summary
    total_interventions INTEGER DEFAULT 0,
    intervention_rate REAL,  -- interventions per turn

    -- Quality Metrics
    total_out_of_control INTEGER DEFAULT 0,
    quality_score REAL,  -- Overall session quality

    -- Primacy Attractor Summary (NO content - just averages)
    avg_purpose_alignment REAL,
    avg_scope_alignment REAL,
    avg_boundary_alignment REAL,

    -- Research Consent
    beta_consent_given BOOLEAN DEFAULT FALSE,
    beta_consent_timestamp TIMESTAMP WITH TIME ZONE,
    consent_version VARCHAR(10),

    -- Cryptographic Signature (Future)
    session_signature TEXT,  -- Cryptographic proof of session authenticity
    signature_algorithm VARCHAR(50)
);

-- Indexes
CREATE INDEX idx_session_mode ON session_summaries(mode);
CREATE INDEX idx_session_created ON session_summaries(created_at);
CREATE INDEX idx_avg_fidelity ON session_summaries(avg_fidelity_score);
CREATE INDEX idx_beta_consent ON session_summaries(beta_consent_given);

-- ============================================================
-- Table 3: Beta Consent Log (Audit Trail)
-- ============================================================
-- Immutable log of consent events
-- ============================================================

CREATE TABLE beta_consent_log (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    consent_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    consent_statement TEXT NOT NULL,
    consent_version VARCHAR(10) NOT NULL,
    ip_address INET,  -- Optional: for audit
    user_agent TEXT,  -- Optional: for audit

    -- Indexes
    CONSTRAINT unique_session_consent UNIQUE (session_id)
);

CREATE INDEX idx_consent_timestamp ON beta_consent_log(consent_timestamp);
CREATE INDEX idx_consent_version ON beta_consent_log(consent_version);

-- ============================================================
-- Table 4: Primacy Attractor Configurations
-- ============================================================
-- Stores PA configurations used in sessions (NO conversation content)
-- Helps understand which PA settings produced which governance outcomes
-- ============================================================

CREATE TABLE primacy_attractor_configs (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- PA Configuration (Structure only - NO actual content)
    purpose_elements INTEGER,  -- Count of purpose statements
    scope_elements INTEGER,    -- Count of scope items
    boundary_elements INTEGER,  -- Count of boundaries

    -- PA Parameters
    constraint_tolerance REAL,
    privacy_level REAL,
    task_priority REAL,

    -- PA Embedding Stats (NO actual embeddings stored)
    pa_embedding_dimensions INTEGER,
    pa_centroid_magnitude REAL,
    basin_radius REAL,

    -- Mode
    mode VARCHAR(20),

    CONSTRAINT unique_session_pa UNIQUE (session_id)
);

CREATE INDEX idx_pa_session ON primacy_attractor_configs(session_id);

-- ============================================================
-- Row Level Security (RLS) Policies
-- ============================================================
-- Ensure only authorized access to delta data
-- ============================================================

-- Enable RLS
ALTER TABLE governance_deltas ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE beta_consent_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE primacy_attractor_configs ENABLE ROW LEVEL SECURITY;

-- Policy: Research team can read all deltas
CREATE POLICY "Research team read access"
ON governance_deltas FOR SELECT
USING (auth.role() = 'authenticated');

-- Policy: App can insert deltas
CREATE POLICY "App insert deltas"
ON governance_deltas FOR INSERT
WITH CHECK (auth.role() = 'service_role');

-- Similar policies for other tables
CREATE POLICY "Research read summaries"
ON session_summaries FOR SELECT
USING (auth.role() = 'authenticated');

CREATE POLICY "App update summaries"
ON session_summaries FOR INSERT
WITH CHECK (auth.role() = 'service_role');

-- ============================================================
-- Useful Views for Research Analysis
-- ============================================================

-- View: High-quality sessions only
CREATE VIEW high_quality_sessions AS
SELECT
    s.*,
    COUNT(g.id) as total_delta_records
FROM session_summaries s
LEFT JOIN governance_deltas g ON s.session_id = g.session_id
WHERE s.avg_fidelity_score >= 0.8
AND s.beta_consent_given = TRUE
GROUP BY s.session_id;

-- View: Intervention analysis
CREATE VIEW intervention_analysis AS
SELECT
    intervention_type,
    COUNT(*) as count,
    AVG(fidelity_score) as avg_fidelity_at_intervention,
    AVG(distance_from_pa) as avg_distance_at_intervention
FROM governance_deltas
WHERE intervention_triggered = TRUE
GROUP BY intervention_type;

-- View: Session quality distribution
CREATE VIEW quality_distribution AS
SELECT
    mode,
    CASE
        WHEN avg_fidelity_score >= 0.9 THEN 'excellent'
        WHEN avg_fidelity_score >= 0.8 THEN 'good'
        WHEN avg_fidelity_score >= 0.6 THEN 'acceptable'
        ELSE 'poor'
    END as quality_tier,
    COUNT(*) as session_count,
    AVG(total_interventions) as avg_interventions
FROM session_summaries
WHERE beta_consent_given = TRUE
GROUP BY mode, quality_tier;

-- ============================================================
-- Functions for Automatic Session Summary Updates
-- ============================================================

-- Function: Update session summary when new delta is added
CREATE OR REPLACE FUNCTION update_session_summary()
RETURNS TRIGGER AS $$
BEGIN
    -- Update session summaries table with new delta data
    INSERT INTO session_summaries (
        session_id,
        mode,
        total_turns,
        avg_fidelity_score,
        min_fidelity_score,
        max_fidelity_score,
        total_interventions
    )
    VALUES (
        NEW.session_id,
        NEW.mode,
        1,
        NEW.fidelity_score,
        NEW.fidelity_score,
        NEW.fidelity_score,
        CASE WHEN NEW.intervention_triggered THEN 1 ELSE 0 END
    )
    ON CONFLICT (session_id) DO UPDATE SET
        total_turns = session_summaries.total_turns + 1,
        avg_fidelity_score = (
            (session_summaries.avg_fidelity_score * session_summaries.total_turns + NEW.fidelity_score)
            / (session_summaries.total_turns + 1)
        ),
        min_fidelity_score = LEAST(session_summaries.min_fidelity_score, NEW.fidelity_score),
        max_fidelity_score = GREATEST(session_summaries.max_fidelity_score, NEW.fidelity_score),
        total_interventions = session_summaries.total_interventions +
            CASE WHEN NEW.intervention_triggered THEN 1 ELSE 0 END,
        updated_at = NOW();

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-update session summary on new delta
CREATE TRIGGER trigger_update_session_summary
AFTER INSERT ON governance_deltas
FOR EACH ROW
EXECUTE FUNCTION update_session_summary();

-- ============================================================
-- Sample Queries for Research
-- ============================================================

-- Query 1: Average fidelity by mode
-- SELECT mode, AVG(avg_fidelity_score) as avg_fidelity
-- FROM session_summaries
-- WHERE beta_consent_given = TRUE
-- GROUP BY mode;

-- Query 2: Intervention effectiveness
-- SELECT
--     intervention_type,
--     AVG(fidelity_score) as avg_fidelity_after_intervention
-- FROM governance_deltas
-- WHERE intervention_triggered = TRUE
-- GROUP BY intervention_type;

-- Query 3: Session quality trends over time
-- SELECT
--     DATE(created_at) as date,
--     AVG(avg_fidelity_score) as daily_avg_fidelity,
--     COUNT(*) as session_count
-- FROM session_summaries
-- WHERE beta_consent_given = TRUE
-- GROUP BY DATE(created_at)
-- ORDER BY date;

-- ============================================================
-- END OF SCHEMA
-- ============================================================
