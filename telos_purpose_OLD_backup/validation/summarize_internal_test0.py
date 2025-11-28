"""
Aggregate and summarize TELOS Internal Test 0 results.

Reads session summary JSONs from validation_results/internal_test0/
and generates a comparative table with hypothesis test evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all *_summary.json files and return list of dicts."""
    summaries = []
    for file in results_dir.glob("*_summary.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            metrics = data["session_metrics"]
            metrics["condition"] = data["session_metadata"]["condition"]
            summaries.append(metrics)
        except Exception as e:
            print(f"Warning: Skipping malformed file {file.name}: {e}")
    return summaries


def aggregate_with_pandas(summaries: List[Dict[str, Any]]) -> None:
    """Aggregate using pandas and print markdown table."""
    df = pd.DataFrame(summaries)
    summary = (
        df.groupby("condition")
        .agg({
            "avg_fidelity": "mean",
            "basin_adherence": "mean",
            "intervention_rate": "mean",
            "governance_breach_events": "mean",
        })
        .round(4)
    )

    print("\n" + "=" * 70)
    print("TELOS Internal Test 0 - Comparative Results")
    print("=" * 70 + "\n")
    print(summary.to_markdown())
    print("\n(Single session per condition - std not computed)\n")


def aggregate_without_pandas(summaries: List[Dict[str, Any]]) -> None:
    """Fallback aggregation without pandas."""
    results = {}
    for s in summaries:
        c = s["condition"]
        results.setdefault(c, []).append(s)

    max_len = max(len(c) for c in results.keys()) if results else 10

    print("\n" + "=" * 70)
    print("TELOS Internal Test 0 - Comparative Results")
    print("=" * 70 + "\n")
    print(f"| {'Condition':{max_len}} | Avg Fidelity | Basin Adherence | Intervention Rate | Breach Events |")
    print(f"|{'-'*(max_len+2)}|--------------|-----------------|-------------------|---------------|")

    for c, items in sorted(results.items()):
        n = len(items)
        avg_fid = sum(i.get("avg_fidelity", 0) for i in items) / n
        avg_bas = sum(i.get("basin_adherence", 0) for i in items) / n
        avg_int = sum(i.get("intervention_rate", 0) for i in items) / n
        avg_br = sum(i.get("governance_breach_events", 0) for i in items) / n
        print(f"| {c:{max_len}} | {avg_fid:12.4f} | {avg_bas:15.4f} | {avg_int:17.4f} | {avg_br:13.1f} |")

    print()


def evaluate_hypotheses(summaries: List[Dict[str, Any]]) -> None:
    """Evaluate H1 and H2 hypothesis tests."""
    fidelities = {s["condition"]: s.get("avg_fidelity", 0) for s in summaries}

    if "telos" not in fidelities:
        print("Warning: TELOS condition not found in results")
        return

    telos_fidelity = fidelities["telos"]
    baselines = {k: v for k, v in fidelities.items() if k != "telos"}

    if not baselines:
        print("Warning: No baseline conditions found for comparison")
        return

    best_baseline = max(baselines.items(), key=lambda x: x[1])
    delta = telos_fidelity - best_baseline[1]

    print("=" * 70)
    print("Hypothesis Testing")
    print("=" * 70 + "\n")

    print(f"H1 (Minimum Improvement): Delta F > 0.15")
    print(f"  TELOS fidelity:        {telos_fidelity:.4f}")
    print(f"  Best baseline ({best_baseline[0]}): {best_baseline[1]:.4f}")
    print(f"  Delta F:               {delta:+.4f}")
    print(f"  Result:                {'PASS' if delta >= 0.15 else 'FAIL'}\n")

    best_overall = max(fidelities.items(), key=lambda x: x[1])
    print(f"H2 (TELOS is Best): TELOS achieves highest fidelity")
    print(f"  Best condition:        {best_overall[0]} ({best_overall[1]:.4f})")
    print(f"  Result:                {'PASS' if best_overall[0] == 'telos' else 'FAIL'}\n")

    if "observation" in fidelities:
        obs_data = [s for s in summaries if s["condition"] == "observation"][0]
        would_intervene = obs_data.get("would_have_intervened_count", 0)
        print("Observation Mode Check:")
        print(f"  Would-have-intervened count: {would_intervene}")
        print(f"  Math working independently:  {'YES' if would_intervene > 0 else 'UNCERTAIN'}\n")


def main():
    """Main execution function."""
    results_dir = Path("validation_results/internal_test0")

    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        print("Run Internal Test 0 first: python -m telos_purpose.validation.run_internal_test0")
        return

    summaries = load_summaries(results_dir)
    if not summaries:
        print(f"Error: No session summaries found in {results_dir}")
        return

    if HAS_PANDAS:
        aggregate_with_pandas(summaries)
    else:
        aggregate_without_pandas(summaries)

    evaluate_hypotheses(summaries)

    print("=" * 70)
    print("TELOS Internal Test 0 Summary Complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
