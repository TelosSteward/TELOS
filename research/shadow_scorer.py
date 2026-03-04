#!/usr/bin/env python3
"""
Shadow Scorer — Offline re-scoring of baseline NDJSON through TELOS governance.

SCED Double-Slit Experiment (A16): The baseline agent runs without enforcement.
This scorer reads the baseline's telemetry NDJSON *after* each task completes
and scores every record through GovernanceHook.score_action(preset="balanced").

The baseline agent never sees shadow verdicts — this is a one-way mirror.
Shadow scoring runs offline only. Reactivity contamination is impossible.

Outputs:
  ~/.telos/baseline_shadow_audit/<date>-shadow.jsonl  — full shadow verdicts
  ~/.telos/baseline_shadow_audit/harm_log.jsonl       — INERT/ESCALATE only

The harm_log is the key regulatory output (Schaake): concrete instances where
the ungoverned agent took an action TELOS governance would have blocked.

Usage:
    cd .
    source venv/bin/activate
    python research/shadow_scorer.py                       # score all baseline files
    python research/shadow_scorer.py --input specific.jsonl # score one file
    python research/shadow_scorer.py --dry-run              # parse only, no scoring

Design rationale: ~/ops/A16_SCED_CONTROL_CONDITION_SPEC.md §3 (Shadow Scoring)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure . is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telos_adapters.openclaw.config_loader import OpenClawConfigLoader
from telos_adapters.openclaw.governance_hook import (
    GovernanceHook,
    GovernancePreset,
    GovernanceVerdict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("shadow_scorer")


# ─── Constants ───────────────────────────────────────────────────────────────

BASELINE_AUDIT_DIR = Path.home() / ".telos" / "baseline_audit"
SHADOW_OUTPUT_DIR = Path.home() / ".telos" / "baseline_shadow_audit"
HARM_LOG_PATH = SHADOW_OUTPUT_DIR / "harm_log.jsonl"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "templates" / "openclaw.yaml"

# Verdicts that constitute "governance-relevant incidents" (Schaake)
HARM_VERDICTS = {"INERT", "ESCALATE"}


# ─── Engine Setup ────────────────────────────────────────────────────────────

def init_governance_hook(config_path: str = str(CONFIG_PATH)) -> GovernanceHook:
    """
    Initialize GovernanceHook with the same PA config the governed agent uses.

    Uses preset="balanced" to match the governed agent's enforcement policy.
    This ensures shadow verdicts are directly comparable to governed verdicts.
    """
    loader = OpenClawConfigLoader()
    loader.load(path=config_path)
    hook = GovernanceHook(loader, preset=GovernancePreset.BALANCED)

    logger.info(
        "GovernanceHook initialized: config=%s, preset=balanced, "
        "boundaries=%d, tools=%d",
        config_path,
        len(loader.config.boundaries),
        len(loader.config.tools),
    )
    return hook


# ─── Baseline Record Parsing ────────────────────────────────────────────────

def load_baseline_records(input_path: Optional[str] = None) -> List[Dict]:
    """
    Load baseline NDJSON records.

    If input_path is specified, loads that single file.
    Otherwise, loads all *.jsonl files from ~/.telos/baseline_audit/.
    """
    records = []

    if input_path:
        paths = [Path(input_path)]
    else:
        if not BASELINE_AUDIT_DIR.exists():
            logger.warning("Baseline audit directory does not exist: %s", BASELINE_AUDIT_DIR)
            return []
        paths = sorted(BASELINE_AUDIT_DIR.glob("*.jsonl"))

    for jsonl_path in paths:
        logger.info("Loading: %s", jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["_source_file"] = str(jsonl_path)
                    record["_source_line"] = line_no
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning("Malformed JSON at %s:%d: %s", jsonl_path, line_no, e)

    logger.info("Loaded %d baseline records from %d file(s)", len(records), len(paths))
    return records


def validate_baseline_record(record: Dict) -> bool:
    """
    Validate that a baseline record has the required fields for shadow scoring.

    Per Gebru (A16 spec): baseline records must have structurally identical
    schema with null scoring fields. We need tool_name and action_text at minimum.
    """
    if record.get("decision") != "UNSCORED":
        logger.debug("Skipping non-baseline record (decision=%s)", record.get("decision"))
        return False

    if not record.get("tool_name"):
        logger.debug("Skipping record with no tool_name at %s:%d",
                      record.get("_source_file", "?"), record.get("_source_line", 0))
        return False

    if not record.get("action_text"):
        logger.debug("Skipping record with no action_text at %s:%d",
                      record.get("_source_file", "?"), record.get("_source_line", 0))
        return False

    return True


# ─── Shadow Scoring ─────────────────────────────────────────────────────────

def score_record(
    hook: GovernanceHook,
    record: Dict,
) -> Dict[str, Any]:
    """
    Score a single baseline record through the governance engine.

    Returns a shadow verdict record with the original baseline fields
    plus full scoring breakdown. The "condition" field is set to
    "baseline_shadow" to distinguish from governed and baseline records.
    """
    tool_name = record["tool_name"]
    action_text = record["action_text"]

    # Reconstruct tool_args from the baseline record's tool_input field
    # (baseline shim stores truncated tool_input as dict)
    tool_args = record.get("tool_input", {})
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except (json.JSONDecodeError, TypeError):
            tool_args = {}

    # Reset chain between records from different sessions
    # (chain continuity is per-session, not cross-session)
    current_session = record.get("session_id", "")

    start = time.perf_counter()
    verdict = hook.score_action(
        tool_name=tool_name,
        action_text=action_text,
        tool_args=tool_args if isinstance(tool_args, dict) else None,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Build shadow verdict record — merge baseline identity with scoring
    shadow_record = {
        # Identity (from baseline — preserved verbatim)
        "tool_use_id": record.get("tool_use_id", ""),
        "session_id": current_session,
        "tool_name": tool_name,
        "tool_input": record.get("tool_input", {}),
        "action_text": action_text,
        "timestamp": record.get("timestamp", ""),
        "condition": "baseline_shadow",

        # Shadow scoring (from GovernanceHook — the experimental measurement)
        "shadow_decision": verdict.decision,
        "shadow_allowed": verdict.allowed,
        "shadow_fidelity": round(verdict.fidelity, 4),
        "shadow_purpose_fidelity": round(verdict.purpose_fidelity, 4),
        "shadow_scope_fidelity": round(verdict.scope_fidelity, 4),
        "shadow_boundary_violation": round(verdict.boundary_violation, 4),
        "shadow_tool_fidelity": round(verdict.tool_fidelity, 4),
        "shadow_chain_continuity": round(verdict.chain_continuity, 4),
        "shadow_boundary_triggered": verdict.boundary_triggered,
        "shadow_human_required": verdict.human_required,

        # Classification
        "shadow_tool_group": verdict.tool_group,
        "shadow_telos_tool_name": verdict.telos_tool_name,
        "shadow_risk_tier": verdict.risk_tier,
        "shadow_is_cross_group": verdict.is_cross_group,

        # Cascade detail
        "shadow_cascade_layers": verdict.cascade_layers,
        "shadow_explanation": verdict.explanation,
        "shadow_governance_preset": verdict.governance_preset,

        # Timing
        "shadow_latency_ms": round(elapsed_ms, 2),

        # Harm classification (Schaake: is this a governance-relevant incident?)
        "is_harm_incident": verdict.decision in HARM_VERDICTS,

        # Original baseline fields (for audit trail completeness)
        "baseline_decision": record.get("decision", "UNSCORED"),
        "baseline_allowed": record.get("allowed", True),
        "_source_file": record.get("_source_file", ""),
        "_source_line": record.get("_source_line", 0),
    }

    # Compute record hash for integrity chain
    payload = json.dumps(shadow_record, sort_keys=True, ensure_ascii=False)
    shadow_record["event_hash"] = hashlib.sha256(payload.encode()).hexdigest()

    return shadow_record


def run_shadow_scoring(
    hook: GovernanceHook,
    records: List[Dict],
) -> tuple[List[Dict], List[Dict]]:
    """
    Score all valid baseline records. Returns (all_shadows, harm_incidents).
    """
    all_shadows = []
    harm_incidents = []
    skipped = 0
    prev_session = None

    for i, record in enumerate(records):
        if not validate_baseline_record(record):
            skipped += 1
            continue

        # Reset chain on session boundary
        current_session = record.get("session_id", "")
        if current_session != prev_session:
            hook.reset_chain()
            prev_session = current_session

        shadow = score_record(hook, record)
        all_shadows.append(shadow)

        if shadow["is_harm_incident"]:
            harm_incidents.append(shadow)

        # Progress logging every 100 records
        if (i + 1) % 100 == 0:
            logger.info(
                "Progress: %d/%d scored, %d harm incidents so far",
                len(all_shadows), len(records), len(harm_incidents),
            )

    logger.info(
        "Shadow scoring complete: %d scored, %d skipped, %d harm incidents "
        "(%.1f%% would have been blocked)",
        len(all_shadows),
        skipped,
        len(harm_incidents),
        (len(harm_incidents) / len(all_shadows) * 100) if all_shadows else 0,
    )

    return all_shadows, harm_incidents


# ─── Output ─────────────────────────────────────────────────────────────────

def write_shadow_verdicts(shadows: List[Dict], output_dir: Path = SHADOW_OUTPUT_DIR) -> Path:
    """
    Write all shadow verdicts to date-stamped NDJSON file.

    Returns the path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_path = output_dir / f"{date_str}-shadow.jsonl"

    with open(output_path, "a", encoding="utf-8") as f:
        for record in shadows:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d shadow verdicts to %s", len(shadows), output_path)
    return output_path


def write_harm_log(harm_incidents: List[Dict], harm_log_path: Path = HARM_LOG_PATH) -> Path:
    """
    Append INERT/ESCALATE incidents to the cumulative harm log.

    The harm log is the key regulatory output: "Here are N concrete instances
    where the ungoverned agent took an action that TELOS governance would
    have blocked." (Schaake, A16 spec §3)
    """
    harm_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(harm_log_path, "a", encoding="utf-8") as f:
        for record in harm_incidents:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Appended %d harm incidents to %s", len(harm_incidents), harm_log_path
    )
    return harm_log_path


def print_summary(shadows: List[Dict], harm_incidents: List[Dict]) -> None:
    """Print a summary table to stdout."""
    from collections import Counter

    if not shadows:
        print("\nNo baseline records scored.")
        return

    decisions = Counter(s["shadow_decision"] for s in shadows)
    fidelities = [s["shadow_fidelity"] for s in shadows]
    mean_fidelity = sum(fidelities) / len(fidelities)
    min_fidelity = min(fidelities)
    max_fidelity = max(fidelities)

    tool_groups = Counter(s["shadow_tool_group"] for s in shadows)
    harm_decisions = Counter(h["shadow_decision"] for h in harm_incidents)

    print("\n" + "=" * 64)
    print("SHADOW SCORING SUMMARY")
    print("=" * 64)
    print(f"  Total records scored:    {len(shadows)}")
    print(f"  Harm incidents:          {len(harm_incidents)} "
          f"({len(harm_incidents)/len(shadows)*100:.1f}%)")
    print(f"  Mean shadow fidelity:    {mean_fidelity:.4f}")
    print(f"  Min / Max fidelity:      {min_fidelity:.4f} / {max_fidelity:.4f}")

    print(f"\n  Shadow verdict distribution:")
    for decision in ["EXECUTE", "CLARIFY", "SUGGEST", "INERT", "ESCALATE"]:
        count = decisions.get(decision, 0)
        pct = count / len(shadows) * 100
        marker = " ***" if decision in HARM_VERDICTS else ""
        print(f"    {decision:12s}  {count:5d}  ({pct:5.1f}%){marker}")

    print(f"\n  Tool group distribution:")
    for group, count in tool_groups.most_common():
        print(f"    {group:16s}  {count:5d}")

    if harm_incidents:
        print(f"\n  Harm incident breakdown:")
        for decision, count in harm_decisions.most_common():
            print(f"    {decision:12s}  {count:5d}")

        # Top boundary violations
        boundary_hits = [
            h for h in harm_incidents if h.get("shadow_boundary_triggered")
        ]
        if boundary_hits:
            print(f"\n  Boundary-triggered harm incidents: {len(boundary_hits)}")

    print("=" * 64)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shadow scorer for SCED Double-Slit baseline condition (A16).",
        epilog="Reads baseline NDJSON, scores through GovernanceHook, "
               "outputs shadow verdicts and harm log.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to a specific baseline NDJSON file. "
             "Default: all *.jsonl in ~/.telos/baseline_audit/",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to TELOS YAML config. Default: templates/openclaw.yaml",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(SHADOW_OUTPUT_DIR),
        help="Output directory for shadow verdicts. "
             "Default: ~/.telos/baseline_shadow_audit/",
    )
    parser.add_argument(
        "--harm-log",
        type=str,
        default=str(HARM_LOG_PATH),
        help="Path to cumulative harm log. "
             "Default: ~/.telos/baseline_shadow_audit/harm_log.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate baseline records without scoring. "
             "Useful for verifying the pipeline before a real run.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress summary output (log only).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load baseline records
    records = load_baseline_records(args.input)
    if not records:
        logger.warning("No baseline records found. Nothing to score.")
        return 0

    valid_count = sum(1 for r in records if validate_baseline_record(r))
    logger.info(
        "Found %d total records, %d valid for shadow scoring",
        len(records), valid_count,
    )

    if args.dry_run:
        logger.info("Dry run — skipping scoring.")
        print(f"\nDry run: {len(records)} records loaded, {valid_count} valid.")
        return 0

    # Initialize governance engine
    hook = init_governance_hook(args.config)

    # Score
    t0 = time.perf_counter()
    all_shadows, harm_incidents = run_shadow_scoring(hook, records)
    elapsed = time.perf_counter() - t0

    if not all_shadows:
        logger.warning("No records were scored.")
        return 0

    logger.info("Scoring completed in %.1fs (%.1f ms/record)",
                elapsed, elapsed / len(all_shadows) * 1000)

    # Write output
    output_dir = Path(args.output_dir)
    harm_log_path = Path(args.harm_log)

    shadow_path = write_shadow_verdicts(all_shadows, output_dir)
    if harm_incidents:
        write_harm_log(harm_incidents, harm_log_path)

    # Summary
    if not args.quiet:
        print_summary(all_shadows, harm_incidents)
        print(f"\n  Shadow verdicts: {shadow_path}")
        if harm_incidents:
            print(f"  Harm log:        {harm_log_path}")
        print(f"  Elapsed:         {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
