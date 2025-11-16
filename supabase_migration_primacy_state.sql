-- ============================================================================
-- Primacy State Migration Script
-- Adds columns for Primacy State metrics to governance_deltas table
--
-- This is a BACKWARD-COMPATIBLE migration:
-- - All new columns are nullable
-- - Existing queries continue to work
-- - Can be rolled back safely
--
-- Date: November 2025
-- Purpose: Enable Primacy State formalization tracking
-- ============================================================================

-- Add Primacy State columns to governance_deltas table
ALTER TABLE governance_deltas
ADD COLUMN IF NOT EXISTS primacy_state_score FLOAT8,
ADD COLUMN IF NOT EXISTS primacy_state_condition TEXT CHECK (
    primacy_state_condition IS NULL OR
    primacy_state_condition IN ('achieved', 'weakening', 'violated', 'collapsed')
),
ADD COLUMN IF NOT EXISTS user_pa_fidelity FLOAT8 CHECK (
    user_pa_fidelity IS NULL OR
    (user_pa_fidelity >= -1 AND user_pa_fidelity <= 1)
),
ADD COLUMN IF NOT EXISTS ai_pa_fidelity FLOAT8 CHECK (
    ai_pa_fidelity IS NULL OR
    (ai_pa_fidelity >= -1 AND ai_pa_fidelity <= 1)
),
ADD COLUMN IF NOT EXISTS pa_correlation FLOAT8 CHECK (
    pa_correlation IS NULL OR
    (pa_correlation >= -1 AND pa_correlation <= 1)
),
ADD COLUMN IF NOT EXISTS v_dual_energy FLOAT8,
ADD COLUMN IF NOT EXISTS delta_v_dual FLOAT8,
ADD COLUMN IF NOT EXISTS primacy_converging BOOLEAN;

-- Add comments to document columns
COMMENT ON COLUMN governance_deltas.primacy_state_score IS 'Primacy State score: PS = ρ_PA · (2·F_user·F_AI)/(F_user + F_AI), range [0,1]';
COMMENT ON COLUMN governance_deltas.primacy_state_condition IS 'PS condition: achieved (≥0.85), weakening (≥0.70), violated (≥0.50), collapsed (<0.50)';
COMMENT ON COLUMN governance_deltas.user_pa_fidelity IS 'User PA fidelity (F_user): conversation purpose alignment, range [-1,1]';
COMMENT ON COLUMN governance_deltas.ai_pa_fidelity IS 'AI PA fidelity (F_AI): AI behavior/role alignment, range [-1,1]';
COMMENT ON COLUMN governance_deltas.pa_correlation IS 'PA correlation (ρ_PA): attractor synchronization, range [-1,1]';
COMMENT ON COLUMN governance_deltas.v_dual_energy IS 'Dual potential energy: V_dual = α·||x-â_user||² + β·||x-â_AI||² + γ·||â_user-â_AI||²';
COMMENT ON COLUMN governance_deltas.delta_v_dual IS 'Energy change: ΔV_dual, negative indicates convergence';
COMMENT ON COLUMN governance_deltas.primacy_converging IS 'Whether system is converging to Primacy State (ΔV < 0)';

-- Add indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_governance_deltas_ps_score
  ON governance_deltas(primacy_state_score)
  WHERE primacy_state_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_governance_deltas_ps_condition
  ON governance_deltas(primacy_state_condition)
  WHERE primacy_state_condition IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_governance_deltas_pa_correlation
  ON governance_deltas(pa_correlation)
  WHERE pa_correlation IS NOT NULL;

-- Composite index for PS diagnostic queries
CREATE INDEX IF NOT EXISTS idx_governance_deltas_ps_diagnostics
  ON governance_deltas(session_id, turn_number, primacy_state_score, user_pa_fidelity, ai_pa_fidelity)
  WHERE primacy_state_score IS NOT NULL;

-- ============================================================================
-- Analytics Views (Optional - for easier reporting)
-- ============================================================================

-- View for PS summary by session
CREATE OR REPLACE VIEW primacy_state_session_summary AS
SELECT
    session_id,
    COUNT(*) as total_turns,
    COUNT(primacy_state_score) as ps_computed_turns,
    AVG(primacy_state_score) as avg_ps_score,
    AVG(user_pa_fidelity) as avg_user_fidelity,
    AVG(ai_pa_fidelity) as avg_ai_fidelity,
    AVG(pa_correlation) as avg_pa_correlation,
    SUM(CASE WHEN primacy_state_condition = 'achieved' THEN 1 ELSE 0 END) as achieved_count,
    SUM(CASE WHEN primacy_state_condition = 'violated' THEN 1 ELSE 0 END) as violated_count,
    SUM(CASE WHEN primacy_converging = true THEN 1 ELSE 0 END) as converging_count,
    SUM(CASE WHEN primacy_converging = false THEN 1 ELSE 0 END) as diverging_count,
    MIN(created_at) as session_start,
    MAX(created_at) as session_end
FROM governance_deltas
GROUP BY session_id;

COMMENT ON VIEW primacy_state_session_summary IS 'Aggregated Primacy State metrics by session for analytics';

-- View for PS trajectory (time series)
CREATE OR REPLACE VIEW primacy_state_trajectory AS
SELECT
    session_id,
    turn_number,
    primacy_state_score,
    primacy_state_condition,
    user_pa_fidelity,
    ai_pa_fidelity,
    pa_correlation,
    delta_v_dual,
    primacy_converging,
    created_at
FROM governance_deltas
WHERE primacy_state_score IS NOT NULL
ORDER BY session_id, turn_number;

COMMENT ON VIEW primacy_state_trajectory IS 'Time series view of Primacy State evolution within sessions';

-- ============================================================================
-- Rollback Script (If needed)
-- ============================================================================
-- To rollback this migration, run:
/*
-- Remove views
DROP VIEW IF EXISTS primacy_state_trajectory;
DROP VIEW IF EXISTS primacy_state_session_summary;

-- Remove indexes
DROP INDEX IF EXISTS idx_governance_deltas_ps_diagnostics;
DROP INDEX IF EXISTS idx_governance_deltas_pa_correlation;
DROP INDEX IF EXISTS idx_governance_deltas_ps_condition;
DROP INDEX IF EXISTS idx_governance_deltas_ps_score;

-- Remove columns
ALTER TABLE governance_deltas
DROP COLUMN IF EXISTS primacy_state_score,
DROP COLUMN IF EXISTS primacy_state_condition,
DROP COLUMN IF EXISTS user_pa_fidelity,
DROP COLUMN IF EXISTS ai_pa_fidelity,
DROP COLUMN IF EXISTS pa_correlation,
DROP COLUMN IF EXISTS v_dual_energy,
DROP COLUMN IF EXISTS delta_v_dual,
DROP COLUMN IF EXISTS primacy_converging;
*/

-- ============================================================================
-- Test Query (Run after migration to verify)
-- ============================================================================
-- SELECT
--     column_name,
--     data_type,
--     is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'governance_deltas'
--   AND column_name LIKE '%primacy%'
--    OR column_name LIKE '%pa_%'
--    OR column_name LIKE '%v_dual%'
-- ORDER BY ordinal_position;