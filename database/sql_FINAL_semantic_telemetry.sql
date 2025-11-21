-- ============================================================================
-- TELOS SEMANTIC TELEMETRY - FINAL VERSION
-- Run this in Supabase SQL Editor
-- This adds lifecycle tracking + semantic intelligence for research
-- ============================================================================

-- STEP 1: Add lifecycle tracking columns
-- These track turn progression through the governance pipeline
ALTER TABLE governance_deltas
ADD COLUMN IF NOT EXISTS turn_status VARCHAR(50),
ADD COLUMN IF NOT EXISTS processing_stage VARCHAR(100),
ADD COLUMN IF NOT EXISTS stage_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS error_message TEXT,
ADD COLUMN IF NOT EXISTS processing_duration_ms INTEGER;

-- STEP 2: Add semantic intelligence columns
-- These provide CONTEXT about what's happening, not just numbers
ALTER TABLE governance_deltas
ADD COLUMN IF NOT EXISTS request_type VARCHAR(100),
ADD COLUMN IF NOT EXISTS request_complexity VARCHAR(50),
ADD COLUMN IF NOT EXISTS detected_topics TEXT[],
ADD COLUMN IF NOT EXISTS topic_shift_magnitude REAL,
ADD COLUMN IF NOT EXISTS semantic_drift_direction TEXT,
ADD COLUMN IF NOT EXISTS constraints_approached TEXT[],
ADD COLUMN IF NOT EXISTS constraint_violation_type VARCHAR(100);

-- STEP 3: Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_turn_status ON governance_deltas(turn_status);
CREATE INDEX IF NOT EXISTS idx_request_type ON governance_deltas(request_type);
CREATE INDEX IF NOT EXISTS idx_detected_topics ON governance_deltas USING GIN(detected_topics);

-- STEP 4: Add validation constraint (with safe error handling)
DO $$
BEGIN
    -- Drop old constraint if it exists
    ALTER TABLE governance_deltas DROP CONSTRAINT IF EXISTS valid_turn_status;

    -- Add new constraint
    ALTER TABLE governance_deltas ADD CONSTRAINT valid_turn_status
        CHECK (turn_status IN ('initiated', 'calculating_pa', 'evaluating', 'completed', 'failed', 'abandoned'));
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Constraint creation skipped (may already exist or conflict)';
END $$;

-- STEP 5: Update existing rows with default values
UPDATE governance_deltas
SET turn_status = 'completed',
    processing_stage = 'Legacy data - status unknown',
    request_type = 'unknown',
    request_complexity = 'unknown'
WHERE turn_status IS NULL;

-- STEP 6: Create semantic analysis view (full telemetry with context)
CREATE OR REPLACE VIEW semantic_telemetry_analysis AS
SELECT
    session_id,
    turn_number,
    turn_status,
    request_type,
    request_complexity,
    detected_topics,
    fidelity_score,
    distance_from_pa,
    purpose_alignment,
    scope_alignment,
    boundary_alignment,
    topic_shift_magnitude,
    semantic_drift_direction,
    constraints_approached,
    constraint_violation_type,
    intervention_triggered,
    intervention_type,
    intervention_reason,
    error_message,
    created_at
FROM governance_deltas
ORDER BY session_id, turn_number;

-- ============================================================================
-- INTELLIGENCE VIEWS - These answer research questions
-- ============================================================================

-- VIEW 1: What request types cause problems?
CREATE OR REPLACE VIEW request_type_performance AS
SELECT
    request_type,
    COUNT(*) as total_requests,
    AVG(fidelity_score) as avg_fidelity,
    COUNT(CASE WHEN intervention_triggered THEN 1 END) as interventions,
    ROUND(100.0 * COUNT(CASE WHEN intervention_triggered THEN 1 END) / COUNT(*), 2) as intervention_rate_pct,
    AVG(distance_from_pa) as avg_pa_distance,
    array_agg(DISTINCT constraint_violation_type) FILTER (WHERE constraint_violation_type IS NOT NULL) as common_violations
FROM governance_deltas
WHERE request_type IS NOT NULL
GROUP BY request_type
ORDER BY avg_fidelity ASC;

-- VIEW 2: Which PA components are weakest?
CREATE OR REPLACE VIEW pa_component_weakness AS
SELECT
    session_id,
    turn_number,
    CASE
        WHEN purpose_alignment IS NOT NULL AND scope_alignment IS NOT NULL AND boundary_alignment IS NOT NULL THEN
            CASE
                WHEN purpose_alignment < scope_alignment AND purpose_alignment < boundary_alignment THEN 'purpose'
                WHEN scope_alignment < boundary_alignment THEN 'scope'
                ELSE 'boundary'
            END
        ELSE 'unknown'
    END as weakest_component,
    purpose_alignment,
    scope_alignment,
    boundary_alignment,
    fidelity_score,
    intervention_triggered,
    detected_topics
FROM governance_deltas
WHERE purpose_alignment IS NOT NULL
   OR scope_alignment IS NOT NULL
   OR boundary_alignment IS NOT NULL;

-- VIEW 3: Topic-based drift patterns
CREATE OR REPLACE VIEW topic_drift_patterns AS
SELECT
    unnest(detected_topics) as topic,
    COUNT(*) as occurrence_count,
    AVG(fidelity_score) as avg_fidelity,
    AVG(topic_shift_magnitude) as avg_shift_magnitude,
    COUNT(CASE WHEN intervention_triggered THEN 1 END) as intervention_count
FROM governance_deltas
WHERE detected_topics IS NOT NULL AND array_length(detected_topics, 1) > 0
GROUP BY topic
ORDER BY intervention_count DESC, avg_fidelity ASC;

-- VIEW 4: Constraint boundary analysis
CREATE OR REPLACE VIEW constraint_boundary_analysis AS
SELECT
    unnest(constraints_approached) as constraint_name,
    COUNT(*) as times_approached,
    COUNT(CASE WHEN intervention_triggered THEN 1 END) as times_violated,
    AVG(fidelity_score) as avg_fidelity_when_approached,
    array_agg(DISTINCT request_type) as request_types_involved
FROM governance_deltas
WHERE constraints_approached IS NOT NULL AND array_length(constraints_approached, 1) > 0
GROUP BY constraint_name
ORDER BY times_violated DESC;

-- VIEW 5: Incomplete turns with semantic context
CREATE OR REPLACE VIEW incomplete_turns_semantic AS
SELECT
    session_id,
    turn_number,
    turn_status,
    processing_stage,
    request_type,
    detected_topics,
    error_message,
    created_at,
    stage_timestamp,
    EXTRACT(EPOCH FROM (NOW() - stage_timestamp)) / 60 as minutes_since_last_update
FROM governance_deltas
WHERE turn_status NOT IN ('completed')
ORDER BY stage_timestamp DESC;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- You now have:
-- - Turn lifecycle tracking (see what stage each turn is at)
-- - Semantic context (understand WHAT is being discussed)
-- - 5 intelligence views for pattern analysis
-- ============================================================================
