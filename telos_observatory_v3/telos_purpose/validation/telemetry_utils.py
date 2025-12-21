"""
Telemetry export utilities for TELOS validation studies.

Exports turn-level CSV and session summary JSON per Internal Test 0 spec.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def export_telemetry(
    result: Dict[str, Any],
    output_dir: Path,
    session_id: str,
    condition: str
) -> None:
    """
    Export validation telemetry to CSV and JSON.
    
    Args:
        result: Output from runner.run_conversation()
        output_dir: Directory to write files
        session_id: Unique session identifier
        condition: stateless/prompt_only/cadence/observation/telos
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export turn-level CSV
    export_turn_csv(result, output_dir, session_id, condition)
    
    # Export session summary JSON
    export_session_json(result, output_dir, session_id, condition)
    
    print(f"✓ Exported telemetry: {session_id}")


def export_turn_csv(
    result: Dict[str, Any],
    output_dir: Path,
    session_id: str,
    condition: str
) -> None:
    """Export turn-level telemetry to CSV."""
    
    csv_path = output_dir / f"{session_id}_turns.csv"
    
    fieldnames = [
        "session_id",
        "condition",
        "turn_id",
        "timestamp",
        "delta_t_ms",
        "user_input",
        "model_output",
        "embedding_distance",
        "fidelity_score",
        "soft_fidelity",
        "lyapunov_delta",
        "intervention_triggered",
        "intervention_type",
        "governance_drift_flag",
        "governance_correction_applied",
        "notes"
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        prev_timestamp = None
        
        for turn in result["turn_results"]:
            # Calculate delta_t
            current_timestamp = turn.get("timestamp", 0)
            delta_t = 0
            if prev_timestamp:
                delta_t = int((current_timestamp - prev_timestamp) * 1000)
            prev_timestamp = current_timestamp
            
            # Format timestamp
            timestamp_str = datetime.fromtimestamp(current_timestamp).isoformat() if current_timestamp else ""
            
            # Determine flags
            drift_flag = not turn.get("in_basin", True) or turn.get("distance_to_attractor", 0) > 0.5
            
            intervention_triggered = turn.get("intervention_applied", False) or turn.get("would_have_intervened", False)
            intervention_type = turn.get("intervention_type", "none")
            
            correction_applied = turn.get("response_was_modified", False) or turn.get("governance_action") not in ["none", None]
            
            row = {
                "session_id": session_id,
                "condition": condition,
                "turn_id": turn["turn"],
                "timestamp": timestamp_str,
                "delta_t_ms": delta_t,
                "user_input": turn.get("user_input", ""),
                "model_output": turn.get("response", turn.get("final_response", "")),
                "embedding_distance": round(turn.get("distance_to_attractor", 0), 4),
                "fidelity_score": round(turn.get("fidelity", 0), 4),
                "soft_fidelity": round(turn.get("soft_fidelity", 0), 4) if "soft_fidelity" in turn else "",
                "lyapunov_delta": round(turn.get("lyapunov", 0), 4) if "lyapunov" in turn else "",
                "intervention_triggered": intervention_triggered,
                "intervention_type": intervention_type,
                "governance_drift_flag": drift_flag,
                "governance_correction_applied": correction_applied,
                "notes": turn.get("notes", "")
            }
            
            writer.writerow(row)


def export_session_json(
    result: Dict[str, Any],
    output_dir: Path,
    session_id: str,
    condition: str
) -> None:
    """Export session summary to JSON."""
    
    json_path = output_dir / f"{session_id}_summary.json"
    
    # Extract metrics
    final_metrics = result.get("final_metrics", {})
    metadata = result.get("metadata", {})
    
    # Count intervention events
    turn_results = result.get("turn_results", [])
    intervention_count = sum(
        1 for t in turn_results 
        if t.get("intervention_applied") or t.get("would_have_intervened")
    )
    
    # Count drift events
    drift_events = sum(
        1 for t in turn_results
        if not t.get("in_basin", True)
    )
    
    # Calculate Lyapunov convergence
    lyapunov_convergent = 0
    lyapunov_divergent = 0
    prev_lyapunov = None
    
    for turn in turn_results:
        current_lyapunov = turn.get("lyapunov")
        if current_lyapunov is not None and prev_lyapunov is not None:
            if current_lyapunov < prev_lyapunov:
                lyapunov_convergent += 1
            else:
                lyapunov_divergent += 1
        prev_lyapunov = current_lyapunov
    
    summary = {
        "session_metadata": {
            "session_id": session_id,
            "condition": condition,
            "date": datetime.now().isoformat(),
            "runner_type": result.get("runner_type", condition),
            "observation_mode": metadata.get("observation_mode", False),
            "intervention_mode": "none" if condition == "stateless" else 
                               "prompt_only" if condition == "prompt_only" else
                               "fixed_interval" if condition == "cadence" else
                               "observation" if condition == "observation" else
                               "adaptive",
            "runtime_version": "v1.0"
        },
        "session_metrics": {
            "total_turns": len(turn_results),
            "avg_fidelity": round(final_metrics.get("fidelity", 0), 4),
            "min_fidelity": round(min((t.get("fidelity", 0) for t in turn_results), default=0), 4),
            "max_fidelity": round(max((t.get("fidelity", 0) for t in turn_results), default=0), 4),
            "avg_distance": round(final_metrics.get("avg_distance", 0), 4),
            "basin_adherence": round(final_metrics.get("basin_adherence", 0), 4),
            "intervention_count": intervention_count,
            "intervention_rate": round(intervention_count / len(turn_results), 4) if turn_results else 0,
            "governance_breach_events": drift_events,
            "lyapunov_convergent_turns": lyapunov_convergent,
            "lyapunov_divergent_turns": lyapunov_divergent
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def aggregate_results(validation_dir: Path) -> Dict[str, Any]:
    """
    Aggregate results from multiple session summaries.
    
    Args:
        validation_dir: Directory containing session_*_summary.json files
        
    Returns:
        Comparative summary across conditions
    """
    summaries_by_condition = {}
    
    for json_file in Path(validation_dir).glob("*_summary.json"):
        with open(json_file) as f:
            summary = json.load(f)
        
        condition = summary["session_metadata"]["condition"]
        
        if condition not in summaries_by_condition:
            summaries_by_condition[condition] = []
        
        summaries_by_condition[condition].append(summary["session_metrics"])
    
    # Compute averages per condition
    comparative = {}
    
    for condition, sessions in summaries_by_condition.items():
        comparative[condition] = {
            "mean_fidelity": round(sum(s["avg_fidelity"] for s in sessions) / len(sessions), 4),
            "mean_basin_adherence": round(sum(s["basin_adherence"] for s in sessions) / len(sessions), 4),
            "mean_intervention_rate": round(sum(s["intervention_rate"] for s in sessions) / len(sessions), 4),
            "session_count": len(sessions)
        }
    
    return {
        "study_overview": {
            "total_sessions": sum(len(s) for s in summaries_by_condition.values()),
            "conditions": list(summaries_by_condition.keys())
        },
        "condition_comparison": comparative
    }