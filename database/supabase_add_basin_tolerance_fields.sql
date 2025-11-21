-- Add basin_constant and constraint_tolerance to validation_telemetric_sessions
-- These fields record the governance settings used for each validation run

ALTER TABLE validation_telemetric_sessions
ADD COLUMN IF NOT EXISTS basin_constant REAL DEFAULT 1.0,
ADD COLUMN IF NOT EXISTS constraint_tolerance REAL DEFAULT 0.05;

COMMENT ON COLUMN validation_telemetric_sessions.basin_constant IS 'Basin radius constant used in r = basin_constant / rigidity formula. Standard value: 1.0 (Goldilocks)';
COMMENT ON COLUMN validation_telemetric_sessions.constraint_tolerance IS 'Constraint tolerance (τ) setting [0,1]. Standard value: 0.05 (strict drift detection)';
