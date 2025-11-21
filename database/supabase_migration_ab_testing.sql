-- ============================================================================
-- SUPABASE MIGRATION: Add A/B Testing Columns
-- ============================================================================
-- Purpose: Enable full A/B testing support for TELOS BETA mode
-- Privacy: All fields are governance metrics only - NO conversation content
-- Date: November 15, 2025
-- ============================================================================

-- Add A/B testing columns to governance_deltas table
ALTER TABLE governance_deltas
ADD COLUMN IF NOT EXISTS test_condition TEXT,
ADD COLUMN IF NOT EXISTS shown_response_source TEXT,
ADD COLUMN IF NOT EXISTS baseline_fidelity FLOAT8,
ADD COLUMN IF NOT EXISTS fidelity_delta FLOAT8;

-- Add index for efficient filtering of A/B test data
CREATE INDEX IF NOT EXISTS idx_governance_deltas_test_condition
ON governance_deltas(test_condition);

-- Add column documentation
COMMENT ON COLUMN governance_deltas.test_condition IS
  'A/B test condition: single_blind_baseline | single_blind_telos | head_to_head';

COMMENT ON COLUMN governance_deltas.shown_response_source IS
  'Which response was shown to user: baseline | telos (hidden from user in single-blind)';

COMMENT ON COLUMN governance_deltas.baseline_fidelity IS
  'Raw LLM fidelity score before TELOS governance (delta only - no content)';

COMMENT ON COLUMN governance_deltas.fidelity_delta IS
  'TELOS improvement delta: telos_fidelity - baseline_fidelity (positive = improvement)';

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================
-- Run this after migration to verify columns were added:

-- SELECT column_name, data_type, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'governance_deltas'
--   AND column_name IN ('test_condition', 'shown_response_source', 'baseline_fidelity', 'fidelity_delta')
-- ORDER BY column_name;

-- Expected output: 4 rows showing the new columns

-- ============================================================================
-- PRIVACY VALIDATION
-- ============================================================================
-- These columns store ONLY governance metrics:
--
-- test_condition:
--   - Values: "single_blind_baseline" | "single_blind_telos" | "head_to_head"
--   - NOT stored: which response was better, user preference
--
-- shown_response_source:
--   - Values: "baseline" | "telos"
--   - NOT stored: the actual response text
--
-- baseline_fidelity:
--   - Values: Float 0.0 - 1.0
--   - NOT stored: why it got that score, what the response said
--
-- fidelity_delta:
--   - Values: Float (can be negative, zero, or positive)
--   - NOT stored: how the responses differed
--
-- ✅ PRIVACY CLAIM MAINTAINED: Only deltas, no conversation content
-- ============================================================================
