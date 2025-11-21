-- Add turn lifecycle tracking columns to governance_deltas
-- Run this in Supabase SQL Editor

ALTER TABLE governance_deltas
ADD COLUMN IF NOT EXISTS turn_status VARCHAR(50),
ADD COLUMN IF NOT EXISTS processing_stage VARCHAR(100),
ADD COLUMN IF NOT EXISTS stage_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS error_message TEXT,
ADD COLUMN IF NOT EXISTS processing_duration_ms INTEGER;

-- Create index for querying by status
CREATE INDEX IF NOT EXISTS idx_turn_status ON governance_deltas(turn_status);

-- Add check constraint for valid statuses
ALTER TABLE governance_deltas
ADD CONSTRAINT IF NOT EXISTS valid_turn_status
CHECK (turn_status IN ('initiated', 'calculating_pa', 'evaluating', 'completed', 'failed', 'abandoned'));

-- Update existing rows to have 'completed' status (backward compatibility)
UPDATE governance_deltas
SET turn_status = 'completed', processing_stage = 'Legacy data - status unknown'
WHERE turn_status IS NULL;

-- Create a view for turn lifecycle analysis
CREATE OR REPLACE VIEW turn_lifecycle_analysis AS
SELECT
    session_id,
    turn_number,
    turn_status,
    processing_stage,
    created_at as turn_started,
    stage_timestamp as last_update,
    EXTRACT(EPOCH FROM (stage_timestamp - created_at)) * 1000 as elapsed_ms,
    fidelity_score,
    distance_from_pa,
    intervention_triggered,
    error_message
FROM governance_deltas
ORDER BY session_id, turn_number, stage_timestamp;

-- Create a view for incomplete/failed turns
CREATE OR REPLACE VIEW incomplete_turns AS
SELECT
    session_id,
    turn_number,
    turn_status,
    processing_stage,
    created_at,
    stage_timestamp,
    error_message,
    EXTRACT(EPOCH FROM (NOW() - stage_timestamp)) / 60 as minutes_since_last_update
FROM governance_deltas
WHERE turn_status NOT IN ('completed')
ORDER BY stage_timestamp DESC;
