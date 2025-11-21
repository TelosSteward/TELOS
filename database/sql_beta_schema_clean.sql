-- TELOS BETA Testing Schema
-- Stores governance metrics for beta sessions

CREATE TABLE beta_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_pa_config JSONB NOT NULL,
    ai_pa_config JSONB NOT NULL,
    basin_constant FLOAT DEFAULT 1.0,
    constraint_tolerance FLOAT DEFAULT 0.05,
    phase_1_complete BOOLEAN DEFAULT FALSE,
    phase_2_complete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    total_turns INTEGER DEFAULT 0
);

CREATE INDEX idx_beta_sessions_created ON beta_sessions(created_at);

CREATE TABLE beta_turns (
    turn_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES beta_sessions(session_id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    phase TEXT CHECK (phase IN ('pa_establishment', 'ab_testing', 'full_telos')),
    user_message TEXT NOT NULL,
    system_served TEXT CHECK (system_served IN ('telos', 'native')),
    telos_response TEXT NOT NULL,
    native_response TEXT NOT NULL,
    response_delivered TEXT NOT NULL,
    user_fidelity FLOAT,
    ai_fidelity FLOAT,
    primacy_state FLOAT,
    distance_from_pa FLOAT,
    in_basin BOOLEAN,
    intervention_calculated BOOLEAN DEFAULT FALSE,
    intervention_applied BOOLEAN DEFAULT FALSE,
    intervention_type TEXT,
    steward_interpretation TEXT,
    user_action TEXT CHECK (user_action IN ('thumbs_up', 'thumbs_down', 'regenerate', 'none')),
    user_preference TEXT CHECK (user_preference IN ('selected_telos', 'selected_native', 'no_preference')),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id, turn_number)
);

CREATE INDEX idx_beta_turns_session ON beta_turns(session_id, turn_number);
CREATE INDEX idx_beta_turns_phase ON beta_turns(phase);
CREATE INDEX idx_beta_turns_system_served ON beta_turns(system_served);
CREATE INDEX idx_beta_turns_in_basin ON beta_turns(in_basin);

CREATE VIEW beta_session_stats AS
SELECT
    s.session_id,
    s.created_at,
    s.completed_at,
    s.total_turns,
    s.phase_1_complete,
    s.phase_2_complete,
    COUNT(t.turn_id) as turns_recorded,
    AVG(t.user_fidelity) as avg_user_fidelity,
    AVG(t.ai_fidelity) as avg_ai_fidelity,
    AVG(t.primacy_state) as avg_primacy_state,
    SUM(CASE WHEN t.in_basin THEN 1 ELSE 0 END)::FLOAT / COUNT(t.turn_id) as basin_adherence_rate,
    SUM(CASE WHEN t.system_served = 'telos' THEN 1 ELSE 0 END) as telos_turns,
    SUM(CASE WHEN t.system_served = 'native' THEN 1 ELSE 0 END) as native_turns
FROM beta_sessions s
LEFT JOIN beta_turns t ON s.session_id = t.session_id
GROUP BY s.session_id, s.created_at, s.completed_at, s.total_turns, s.phase_1_complete, s.phase_2_complete;

CREATE VIEW beta_preference_analysis AS
SELECT
    session_id,
    system_served,
    COUNT(*) as times_served,
    SUM(CASE WHEN user_action = 'thumbs_up' THEN 1 ELSE 0 END) as thumbs_up_count,
    SUM(CASE WHEN user_action = 'thumbs_down' THEN 1 ELSE 0 END) as thumbs_down_count,
    SUM(CASE WHEN user_action = 'regenerate' THEN 1 ELSE 0 END) as regenerate_count,
    AVG(user_fidelity) as avg_user_fidelity_for_system,
    AVG(primacy_state) as avg_primacy_state_for_system
FROM beta_turns
WHERE system_served IS NOT NULL
GROUP BY session_id, system_served;

CREATE VIEW beta_drift_analysis AS
SELECT
    session_id,
    turn_number,
    phase,
    user_fidelity,
    distance_from_pa,
    in_basin,
    intervention_calculated,
    intervention_applied,
    intervention_type,
    CASE
        WHEN in_basin THEN 'aligned'
        ELSE 'drift'
    END as alignment_status
FROM beta_turns
ORDER BY session_id, turn_number;

COMMENT ON TABLE beta_sessions IS 'BETA session metadata - contains PA configurations and governance parameters';
COMMENT ON TABLE beta_turns IS 'BETA turn-level data - contains governance metrics AND conversation content for post-session review';
COMMENT ON COLUMN beta_turns.user_message IS 'User input - stored for post-session review, not sent to external analysis';
COMMENT ON COLUMN beta_turns.telos_response IS 'TELOS response - stored for comparison in AB testing';
COMMENT ON COLUMN beta_turns.native_response IS 'Native response - stored for comparison in AB testing';
COMMENT ON COLUMN beta_turns.user_fidelity IS 'PRIMARY METRIC: User question alignment to PA';
COMMENT ON COLUMN beta_turns.ai_fidelity IS 'PRIMARY METRIC: AI response alignment to PA';
COMMENT ON COLUMN beta_turns.primacy_state IS 'PRIMARY METRIC: Harmonic mean of user and AI fidelities';
COMMENT ON COLUMN beta_turns.distance_from_pa IS 'PRIMARY METRIC: Semantic distance in embedding space';
COMMENT ON COLUMN beta_turns.in_basin IS 'PRIMARY METRIC: Whether state falls within basin of attraction';
