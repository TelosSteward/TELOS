CREATE TRIGGER trigger_update_session_summary
AFTER INSERT ON governance_deltas
FOR EACH ROW
EXECUTE FUNCTION update_session_summary();
