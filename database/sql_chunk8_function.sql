CREATE OR REPLACE FUNCTION update_session_summary()
RETURNS TRIGGER AS $$
BEGIN
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
