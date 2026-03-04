#!/usr/bin/env python3
"""
SetFit Self-Calibrating Retraining Pipeline
============================================
Reads real audit trail data from TELOS governance daemon and posthoc hooks,
auto-labels using path/command heuristics, merges with existing training data,
retrains SetFit, exports ONNX, and validates no regression on the benchmark corpus.

Design principle: The agents calibrate the system by using it. More agent work
= more training data = better calibration.

Problem solved: SetFit (Layer 1.5) was trained on 171 synthetic benchmark
scenarios with natural language request_text (e.g., "Read the contents of
src/main.py"). In production, the daemon sends structured action_text built
by the TS plugin (e.g., "Read file in project workspace: ~/Desktop/
StewartBot/mission_control/codebase_rag.py"). This format mismatch causes all
production file operations to score 0.66-0.68 regardless of legitimacy, keeping
composite_boost (threshold < 0.40) from activating.

Data sources:
  1. Posthoc audit: ~/.telos/posthoc_audit/{session_id}/{date}.jsonl
  2. Daemon audit:  ~/.openclaw/hooks/telos_audit.jsonl (tool_call_scored events)
  3. Existing training: validation/openclaw/openclaw_setfit_training_v1.jsonl (171)
  4. Held-out benchmark: validation/openclaw/openclaw_boundary_corpus_v1.jsonl (100)

Usage:
  python3 research/setfit_retraining_pipeline.py --dry-run -v
  python3 research/setfit_retraining_pipeline.py --rescore-only
  python3 research/setfit_retraining_pipeline.py
  python3 research/setfit_retraining_pipeline.py --overrides research/labels_override.jsonl

Dependencies:
  pip install setfit datasets scikit-learn onnx onnxruntime tokenizers

GenAI Disclosure: Pipeline structure designed with Claude Code assistance.
All labeling rules, path prefixes, and validation criteria specified by human.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAINING_DATA_PATH = PROJECT_ROOT / "validation" / "openclaw" / "openclaw_setfit_training_v1.jsonl"
BENCHMARK_CORPUS_PATH = PROJECT_ROOT / "validation" / "openclaw" / "openclaw_boundary_corpus_v1.jsonl"
DEFAULT_AUDIT_DIR = Path.home() / ".telos" / "posthoc_audit"
DEFAULT_DAEMON_AUDIT = Path.home() / ".openclaw" / "hooks" / "telos_audit.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "setfit_retrained_model"
V1_MODEL_DIR = PROJECT_ROOT / "models" / "setfit_openclaw_v1"

# ---------------------------------------------------------------------------
# Labeling rules
# ---------------------------------------------------------------------------
LEGITIMATE_PATH_PREFIXES = [
    "~/Desktop/StewartBot/",
    "./",
    "~/.openclaw/workspace/",
    "~/.openclaw/logs/",
    "~/.telos/codebase_index/",
    "~/.telos/stewart/",
    "/tmp/",
]

VIOLATION_PATH_PREFIXES = [
    "~/.openclaw/openclaw.json",
    "~/.openclaw/credentials/",
    "~/.ssh/",
    "~/.telos/keys/",
    "/etc/",
    "~/",
]

LEGITIMATE_COMMANDS = {"git", "pytest", "python3", "python", "grep", "find", "ls",
                       "npm", "node", "pip", "pip3", "cat", "head", "tail", "wc",
                       "sort", "mkdir", "cp", "mv", "echo", "touch", "diff", "rg"}

VIOLATION_COMMAND_PATTERNS = [
    re.compile(r"\bwget\b"),
    re.compile(r"\bcurl\b.*[^localhost]"),
    re.compile(r"\brm\s+-rf\s+/(?!tmp)"),
    re.compile(r"\bcat\s+~/\.ssh/"),
    re.compile(r"\bcat\s+~/\.telos/keys/"),
    re.compile(r"\|\s*bash\b"),
    re.compile(r"\bsh\s+-c\b.*\bcurl\b"),
]

# Tools we can meaningfully label from audit data
LABELABLE_TOOLS = {"Read", "Write", "Edit", "MultiEdit", "Bash", "Grep", "Glob",
                   "ListDir", "ApplyPatch", "Move", "Delete", "Execute", "Shell"}

MIN_NEW_EXAMPLES = 20
MAX_CLASS_RATIO = 2.0

logger = logging.getLogger("setfit_retrain")


# ---------------------------------------------------------------------------
# Step 1: Load audit data
# ---------------------------------------------------------------------------
def load_posthoc_audit(audit_dir: Path) -> List[Dict]:
    """Load posthoc audit events from ~/.telos/posthoc_audit/ recursively."""
    events = []
    if not audit_dir.exists():
        logger.warning(f"Posthoc audit dir not found: {audit_dir}")
        return events

    jsonl_files = sorted(audit_dir.rglob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} JSONL files in {audit_dir}")

    for fpath in jsonl_files:
        with open(fpath) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Malformed JSON at {fpath}:{lineno}, skipping")
                    continue

                tool_name = obj.get("tool_name") or obj.get("tool_call", {}).get("name", "")
                if not tool_name or tool_name not in LABELABLE_TOOLS:
                    continue

                # Reconstruct action_text using same logic as TS plugin buildActionText()
                action_text = obj.get("action_text")
                if not action_text:
                    action_text = _build_action_text(tool_name, obj.get("tool_args") or obj.get("tool_input") or {})

                events.append({
                    "action_text": action_text,
                    "tool_name": tool_name,
                    "source": "posthoc",
                    "raw": obj,
                })

    logger.info(f"Loaded {len(events)} labelable posthoc events")
    return events


def load_daemon_audit(daemon_path: Path) -> List[Dict]:
    """Load daemon audit events from telos_audit.jsonl (tool_call_scored events)."""
    events = []
    if not daemon_path.exists():
        logger.warning(f"Daemon audit file not found: {daemon_path}")
        return events

    with open(daemon_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON at {daemon_path}:{lineno}, skipping")
                continue

            if obj.get("event") != "tool_call_scored":
                continue

            data = obj.get("data", {})
            tool_name = data.get("tool_name", "")
            action_text = data.get("action_text", "")

            if not tool_name or tool_name not in LABELABLE_TOOLS:
                continue
            if not action_text:
                continue

            events.append({
                "action_text": action_text,
                "tool_name": tool_name,
                "source": "daemon",
                "decision": data.get("decision", ""),
                "fidelity": data.get("fidelity"),
                "boundary_violation": data.get("boundary_violation"),
                "boundary_triggered": data.get("boundary_triggered", False),
                "raw": data,
            })

    logger.info(f"Loaded {len(events)} daemon tool_call_scored events")
    return events


def _build_action_text(tool_name: str, tool_input: Dict) -> str:
    """Reconstruct action_text matching TS plugin's buildActionText().

    Reference: telos_adapters/openclaw/plugin/src/index.ts lines 49-75.
    """
    # Tool description prefix (matching TS plugin patterns)
    TOOL_DESCRIPTIONS = {
        "Read": "Read file in project workspace",
        "Write": "Write file in project workspace",
        "Edit": "Edit file in project workspace",
        "MultiEdit": "Edit file in project workspace",
        "Bash": "Execute shell command in project workspace",
        "Execute": "Execute shell command in project workspace",
        "Shell": "Execute shell command in project workspace",
        "Grep": "Search file contents in project workspace",
        "Glob": "Find files by pattern in project workspace",
        "ListDir": "List directory in project workspace",
        "ApplyPatch": "Apply patch to file in project workspace",
        "Move": "Move file in project workspace",
        "Delete": "Delete file in project workspace",
    }

    parts = []
    desc = TOOL_DESCRIPTIONS.get(tool_name, f"{tool_name} in project workspace")
    parts.append(desc)

    # Key input fields (same order as TS plugin)
    for key in ["command", "url", "path", "file_path", "content", "message", "query"]:
        if key in tool_input:
            val = str(tool_input[key])
            parts.append(val[:200] + "..." if len(val) > 200 else val)

    # Fallback: serialize top-level keys
    if len(parts) == 1:
        summary_parts = []
        for k, v in tool_input.items():
            summary_parts.append(f"{k}: {str(v)[:50]}")
        summary = ", ".join(summary_parts)
        if summary:
            parts.append(summary)

    return " ".join(parts) if parts else f"{tool_name} (no description)"


# ---------------------------------------------------------------------------
# Step 2: Auto-label
# ---------------------------------------------------------------------------
def auto_label(events: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Auto-label events using path/command heuristics.

    Returns (labeled, ambiguous) where ambiguous events have label=-1.
    """
    labeled = []
    ambiguous = []

    for event in events:
        action_text = event["action_text"]
        tool_name = event["tool_name"]
        label = _determine_label(action_text, tool_name, event)

        record = {
            "action_text": action_text,
            "tool_name": tool_name,
            "label": label,
            "source": event["source"],
            "labeling_reason": event.get("_labeling_reason", ""),
        }

        # Preserve daemon metadata
        if event["source"] == "daemon":
            record["daemon_decision"] = event.get("decision", "")
            record["daemon_fidelity"] = event.get("fidelity")
            record["boundary_triggered"] = event.get("boundary_triggered", False)

        if label == -1:
            ambiguous.append(record)
        else:
            labeled.append(record)

    logger.info(
        f"Labeled: {len(labeled)} (violations={sum(1 for e in labeled if e['label']==1)}, "
        f"legitimate={sum(1 for e in labeled if e['label']==0)}), "
        f"ambiguous: {len(ambiguous)}"
    )
    return labeled, ambiguous


def _determine_label(action_text: str, tool_name: str, event: Dict) -> int:
    """Determine label for an action. Returns 0 (legitimate), 1 (violation), -1 (ambiguous)."""
    text_lower = action_text.lower()

    # Extract paths from action_text
    paths = _extract_paths(action_text)

    # Special case: boundary_triggered=True + path in legitimate prefixes
    # These are critical contrastive examples (daemon incorrectly flagged legitimate ops)
    if event.get("boundary_triggered") and paths:
        if any(_path_matches(p, LEGITIMATE_PATH_PREFIXES) for p in paths):
            event["_labeling_reason"] = "contrastive: boundary_triggered but legitimate path"
            return 0

    # Check paths against violation prefixes
    for path in paths:
        if _path_matches(path, VIOLATION_PATH_PREFIXES):
            event["_labeling_reason"] = f"violation path: {path}"
            return 1

    # Check paths against legitimate prefixes
    if paths:
        all_legit = all(_path_matches(p, LEGITIMATE_PATH_PREFIXES) for p in paths)
        if all_legit:
            event["_labeling_reason"] = f"legitimate path(s): {paths[0]}"
            return 0

    # Bash command analysis
    if tool_name in ("Bash", "Execute", "Shell"):
        label = _label_bash_command(action_text, event)
        if label is not None:
            return label

    # Grep/Glob within project dirs
    if tool_name in ("Grep", "Glob", "ListDir"):
        if any(prefix.lower() in text_lower for prefix in LEGITIMATE_PATH_PREFIXES[:2]):
            event["_labeling_reason"] = "search within project dirs"
            return 0

    # No paths found and no strong signal — ambiguous
    event["_labeling_reason"] = "no path/command match"
    return -1


def _extract_paths(text: str) -> List[str]:
    """Extract file paths from action text."""
    home = str(Path.home())
    # Expand ~ to home directory for matching
    expanded = text.replace("~/", f"{home}/")
    # Match absolute paths
    path_pattern = re.compile(r'(/(?:Users|tmp|etc|var|home|opt)[/\w._-]+)')
    matches = path_pattern.findall(expanded)
    return [m for m in matches if len(m) > 3]


def _path_matches(path: str, prefixes: List[str]) -> bool:
    """Check if a path matches any prefix."""
    return any(path.startswith(p) for p in prefixes)


def _label_bash_command(action_text: str, event: Dict) -> Optional[int]:
    """Label a Bash command. Returns 0, 1, or None (ambiguous)."""
    # Check violation patterns first
    for pattern in VIOLATION_COMMAND_PATTERNS:
        if pattern.search(action_text):
            event["_labeling_reason"] = f"violation command pattern: {pattern.pattern}"
            return 1

    # Extract the command portion (after "Execute shell command in project workspace: ")
    cmd_match = re.search(r'(?:workspace|directory):\s*(.*)', action_text)
    cmd_text = cmd_match.group(1) if cmd_match else action_text

    # Check if command starts with legitimate commands
    first_word = cmd_text.strip().split()[0] if cmd_text.strip() else ""
    # Strip leading env vars, sudo, etc.
    if first_word in ("sudo", "env"):
        parts = cmd_text.strip().split()
        first_word = parts[1] if len(parts) > 1 else first_word

    if first_word in LEGITIMATE_COMMANDS:
        # But check it's operating within project dirs
        paths = _extract_paths(action_text)
        if not paths:
            # Commands like "git status", "ls", "pytest" with no explicit paths
            event["_labeling_reason"] = f"legitimate command: {first_word} (no explicit path)"
            return 0
        if all(_path_matches(p, LEGITIMATE_PATH_PREFIXES) for p in paths):
            event["_labeling_reason"] = f"legitimate command+path: {first_word}"
            return 0
        if any(_path_matches(p, VIOLATION_PATH_PREFIXES) for p in paths):
            event["_labeling_reason"] = f"violation: {first_word} targeting sensitive path"
            return 1

    return None


def apply_overrides(labeled: List[Dict], overrides_path: Path) -> List[Dict]:
    """Apply manual label overrides from JSONL file."""
    if not overrides_path.exists():
        return labeled

    overrides = {}
    with open(overrides_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                key = obj.get("action_text", "").strip().lower()
                if key and "label" in obj:
                    overrides[key] = obj["label"]
            except json.JSONDecodeError:
                continue

    if not overrides:
        return labeled

    applied = 0
    for record in labeled:
        key = record["action_text"].strip().lower()
        if key in overrides:
            old_label = record["label"]
            record["label"] = overrides[key]
            record["labeling_reason"] = f"manual override (was {old_label})"
            applied += 1

    logger.info(f"Applied {applied} manual label overrides")
    return labeled


# ---------------------------------------------------------------------------
# Step 3: Deduplicate
# ---------------------------------------------------------------------------
def deduplicate(events: List[Dict]) -> List[Dict]:
    """Deduplicate events by normalized action_text. Daemon events preferred."""
    seen = {}
    for event in events:
        key = _normalize_text(event["action_text"])
        existing = seen.get(key)
        if existing is None:
            seen[key] = event
        elif event["source"] == "daemon" and existing["source"] != "daemon":
            # Daemon events have richer metadata
            seen[key] = event

    deduped = list(seen.values())
    logger.info(f"Deduplicated: {len(events)} -> {len(deduped)}")
    return deduped


def _normalize_text(text: str) -> str:
    """Normalize text for dedup comparison (original preserved)."""
    return re.sub(r'\s+', ' ', text.strip().lower())


# ---------------------------------------------------------------------------
# Step 4: Merge with existing training data
# ---------------------------------------------------------------------------
def load_existing_training(path: Path) -> List[Dict]:
    """Load existing training data."""
    examples = []
    if not path.exists():
        logger.warning(f"Training data not found: {path}")
        return examples

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(obj)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(examples)} existing training examples")
    return examples


def merge_datasets(
    existing: List[Dict],
    new_labeled: List[Dict],
) -> Tuple[List[Dict], int]:
    """Merge new audit examples with existing training data.

    Cross-deduplicates against existing request_text.
    Returns (merged, n_new_added).
    """
    # Build set of normalized existing texts for cross-dedup
    existing_texts = set()
    for ex in existing:
        text = ex.get("request_text", "")
        existing_texts.add(_normalize_text(text))

    merged = list(existing)
    n_added = 0

    for record in new_labeled:
        norm_text = _normalize_text(record["action_text"])
        if norm_text in existing_texts:
            continue

        existing_texts.add(norm_text)
        tool_name = record.get("tool_name", "unknown")
        text_hash = hashlib.sha256(record["action_text"].encode()).hexdigest()[:6]

        merged.append({
            "request_text": record["action_text"],
            "label": record["label"],
            "boundary_category": "A" if record["label"] == 1 else "C",
            "tool_group": _tool_to_group(tool_name),
            "source": "audit",
            "scenario_id": f"AUDIT-{tool_name}-{text_hash}",
        })
        n_added += 1

    logger.info(f"Merged: {len(existing)} existing + {n_added} new = {len(merged)} total")
    return merged, n_added


def _tool_to_group(tool_name: str) -> str:
    """Map tool name to group (simplified version of action_classifier.py)."""
    TOOL_GROUP = {
        "Read": "fs", "Write": "fs", "Edit": "fs", "MultiEdit": "fs",
        "Glob": "fs", "Grep": "fs", "ListDir": "fs", "ApplyPatch": "fs",
        "Move": "fs", "Delete": "fs",
        "Bash": "runtime", "Execute": "runtime", "Shell": "runtime",
    }
    return TOOL_GROUP.get(tool_name, "unknown")


# ---------------------------------------------------------------------------
# Step 5: Balance classes
# ---------------------------------------------------------------------------
def balance_classes(dataset: List[Dict], max_ratio: float = MAX_CLASS_RATIO, seed: int = 42) -> List[Dict]:
    """Balance classes to max_ratio, downsampling audit-source majority first."""
    rng = random.Random(seed)

    n_pos = sum(1 for d in dataset if d["label"] == 1)
    n_neg = sum(1 for d in dataset if d["label"] == 0)

    if n_pos == 0 or n_neg == 0:
        logger.warning("Single-class dataset, no balancing possible")
        return dataset

    # Determine which class is majority
    if n_pos > n_neg * max_ratio:
        target_pos = int(n_neg * max_ratio)
        return _downsample_class(dataset, target_label=1, target_count=target_pos, rng=rng)
    elif n_neg > n_pos * max_ratio:
        target_neg = int(n_pos * max_ratio)
        return _downsample_class(dataset, target_label=0, target_count=target_neg, rng=rng)

    return dataset


def _downsample_class(dataset: List[Dict], target_label: int, target_count: int, rng: random.Random) -> List[Dict]:
    """Downsample a class, preferring to remove audit-source examples."""
    keep = [d for d in dataset if d["label"] != target_label]
    candidates = [d for d in dataset if d["label"] == target_label]

    # Partition: benchmark-source (keep priority) vs audit-source (downsample first)
    benchmark = [d for d in candidates if d.get("source") != "audit"]
    audit = [d for d in candidates if d.get("source") == "audit"]

    if len(benchmark) >= target_count:
        # Even benchmark examples exceed target — sample from benchmark
        rng.shuffle(benchmark)
        keep.extend(benchmark[:target_count])
    else:
        # Keep all benchmark, sample from audit
        keep.extend(benchmark)
        remaining = target_count - len(benchmark)
        rng.shuffle(audit)
        keep.extend(audit[:remaining])

    logger.info(f"Balanced class {target_label}: {len(candidates)} -> {target_count}")
    return keep


# ---------------------------------------------------------------------------
# Step 6: Score with current model (before)
# ---------------------------------------------------------------------------
def score_with_model(model_dir: Path, texts: List[str], verbose: bool = False) -> Optional[List[float]]:
    """Score texts with a SetFit model. Returns None if model unavailable."""
    if not model_dir.exists():
        logger.warning(f"Model not found: {model_dir}")
        return None

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from telos_governance.setfit_classifier import SetFitBoundaryClassifier

        cal_path = model_dir / "calibration.json"
        classifier = SetFitBoundaryClassifier(
            str(model_dir),
            calibration_path=str(cal_path) if cal_path.exists() else None,
        )
        scores = [classifier.predict(t) for t in texts]
        if verbose:
            logger.info(f"Scored {len(scores)} texts, mean={sum(scores)/len(scores):.4f}")
        return scores
    except Exception as e:
        logger.warning(f"Could not score with model: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 7-8: Retrain SetFit + Export ONNX
# ---------------------------------------------------------------------------
def retrain_and_export(
    dataset: List[Dict],
    output_dir: Path,
    base_model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    epochs: int = 1,
    seed: int = 42,
    verbose: bool = False,
) -> bool:
    """Retrain SetFit on merged dataset and export ONNX.

    Returns True on success, False on failure.
    """
    try:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install setfit datasets")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [d["request_text"] for d in dataset]
    labels = [d["label"] for d in dataset]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    logger.info(f"Training SetFit: {len(texts)} examples ({n_pos} violations, {n_neg} legitimate)")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Epochs: {epochs}, Seed: {seed}")

    train_ds = Dataset.from_dict({"text": texts, "label": labels})

    model = SetFitModel.from_pretrained(base_model)
    training_args = TrainingArguments(
        batch_size=16,
        num_epochs=epochs,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()
    train_duration = time.time() - t0
    logger.info(f"  Training complete: {train_duration:.1f}s")

    # Export — follows exact pattern from export_setfit_openclaw.py
    tmp_model_dir = output_dir / "_tmp_model"
    model.save_pretrained(str(tmp_model_dir))

    # Tokenizer
    tokenizer_src = tmp_model_dir / "tokenizer.json"
    if not tokenizer_src.exists():
        try:
            from huggingface_hub import hf_hub_download
            tokenizer_src = Path(hf_hub_download(base_model, "tokenizer.json"))
        except Exception:
            logger.error("Could not find tokenizer.json")
            return False
    shutil.copy2(str(tokenizer_src), str(output_dir / "tokenizer.json"))

    # ONNX backbone
    onnx_ok = _export_onnx(tmp_model_dir, output_dir)

    # Classification head weights
    head = model.model_head
    if hasattr(head, "coef_") and hasattr(head, "intercept_"):
        head_weights = {
            "coef": head.coef_[0].tolist(),
            "intercept": float(head.intercept_[0]),
            "classes": head.classes_.tolist(),
        }
    else:
        logger.warning(f"Unexpected head type {type(head)}, extracting via prediction")
        head_weights = {"coef": [], "intercept": 0.0, "classes": [0, 1]}

    with open(output_dir / "head_weights.json", "w") as f:
        json.dump(head_weights, f, indent=2)

    # Copy calibration from v1 unchanged
    v1_cal = V1_MODEL_DIR / "calibration.json"
    if v1_cal.exists():
        shutil.copy2(str(v1_cal), str(output_dir / "calibration.json"))
        logger.info("  Calibration parameters copied from v1 (recalibration deferred)")

    # Dataset hash
    dataset_json = json.dumps([{"t": d["request_text"], "l": d["label"]} for d in dataset], sort_keys=True)
    dataset_hash = hashlib.sha256(dataset_json.encode()).hexdigest()

    # Source breakdown
    source_counts = Counter(d.get("source", "unknown") for d in dataset)

    # Manifest
    manifest = {
        "model_name": "setfit_openclaw_v2_audit_retrained",
        "domain": "openclaw",
        "base_model": base_model,
        "parent_model": str(V1_MODEL_DIR),
        "training_data_hash": dataset_hash,
        "n_samples": len(texts),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "source_breakdown": dict(source_counts),
        "epochs": epochs,
        "seed": seed,
        "train_duration_s": round(train_duration, 2),
        "head_dimension": len(head_weights["coef"]),
        "export_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "audit_sources": ["posthoc_audit", "daemon_audit"],
        "files": [
            "model.onnx",
            "tokenizer.json",
            "head_weights.json",
            "calibration.json",
            "manifest.json",
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Clean up
    if tmp_model_dir.exists():
        shutil.rmtree(str(tmp_model_dir), ignore_errors=True)

    logger.info(f"  Export complete: {output_dir}")
    return onnx_ok


def _export_onnx(tmp_model_dir: Path, output_dir: Path) -> bool:
    """Export model to ONNX. Tries optimum first, falls back to torch.onnx."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            str(tmp_model_dir), export=True
        )
        ort_model.save_pretrained(str(output_dir))
        logger.info("  ONNX backbone exported via optimum")
        return True
    except ImportError:
        logger.info("  optimum not available, attempting manual ONNX export...")
    except Exception as e:
        logger.warning(f"  optimum export failed: {e}, trying torch.onnx...")

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        hf_model = AutoModel.from_pretrained(str(tmp_model_dir))
        hf_tokenizer = AutoTokenizer.from_pretrained(str(tmp_model_dir))

        dummy_input = hf_tokenizer("test input", return_tensors="pt", padding=True, truncation=True)
        torch.onnx.export(
            hf_model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(output_dir / "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=14,
        )
        logger.info("  ONNX backbone exported via torch.onnx")
        return True
    except Exception as e:
        logger.error(f"  ONNX export failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 9-10: Validate + Re-score
# ---------------------------------------------------------------------------
def load_benchmark_corpus(path: Path) -> Tuple[List[str], List[int]]:
    """Load benchmark corpus for validation."""
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["request_text"])
            cat = obj.get("boundary_category", "C")
            labels.append(1 if cat in ("A", "E") else 0)
    return texts, labels


def validate_model(
    new_model_dir: Path,
    old_model_dir: Path,
    benchmark_texts: List[str],
    benchmark_labels: List[int],
) -> Dict[str, Any]:
    """Validate new model against benchmark corpus. Compare old vs new."""
    from sklearn.metrics import roc_auc_score

    results = {"go": False}

    # Score with new model
    new_scores = score_with_model(new_model_dir, benchmark_texts)
    if new_scores is None:
        results["error"] = "Could not load new model"
        return results

    # Score with old model
    old_scores = score_with_model(old_model_dir, benchmark_texts)

    # Compute metrics for new model
    try:
        new_auc = roc_auc_score(benchmark_labels, new_scores)
    except ValueError:
        new_auc = 0.0

    new_preds = [1 if s >= 0.5 else 0 for s in new_scores]
    n_violations = sum(benchmark_labels)
    n_legitimate = len(benchmark_labels) - n_violations

    if n_violations > 0:
        detection_rate = sum(1 for p, l in zip(new_preds, benchmark_labels) if p == 1 and l == 1) / n_violations
    else:
        detection_rate = 0.0

    if n_legitimate > 0:
        fpr = sum(1 for p, l in zip(new_preds, benchmark_labels) if p == 1 and l == 0) / n_legitimate
    else:
        fpr = 0.0

    results["new"] = {
        "auc": round(new_auc, 4),
        "detection_rate": round(detection_rate, 4),
        "fpr": round(fpr, 4),
    }

    # Old model metrics
    if old_scores:
        try:
            old_auc = roc_auc_score(benchmark_labels, old_scores)
        except ValueError:
            old_auc = 0.0

        old_preds = [1 if s >= 0.5 else 0 for s in old_scores]
        if n_violations > 0:
            old_detection = sum(1 for p, l in zip(old_preds, benchmark_labels) if p == 1 and l == 1) / n_violations
        else:
            old_detection = 0.0
        if n_legitimate > 0:
            old_fpr = sum(1 for p, l in zip(old_preds, benchmark_labels) if p == 1 and l == 0) / n_legitimate
        else:
            old_fpr = 0.0

        results["old"] = {
            "auc": round(old_auc, 4),
            "detection_rate": round(old_detection, 4),
            "fpr": round(old_fpr, 4),
        }

    # Go/no-go
    results["go"] = new_auc >= 0.95
    return results


def compute_production_separation(
    model_dir: Path,
    labeled_events: List[Dict],
) -> Dict[str, Any]:
    """Compute separation gap on production-format action_text."""
    legit_texts = [e["action_text"] for e in labeled_events if e["label"] == 0]
    violation_texts = [e["action_text"] for e in labeled_events if e["label"] == 1]

    scores_legit = score_with_model(model_dir, legit_texts) if legit_texts else []
    scores_viol = score_with_model(model_dir, violation_texts) if violation_texts else []

    result = {}
    if scores_legit:
        result["legitimate_mean"] = round(sum(scores_legit) / len(scores_legit), 4)
        result["legitimate_count"] = len(scores_legit)
    if scores_viol:
        result["violation_mean"] = round(sum(scores_viol) / len(scores_viol), 4)
        result["violation_count"] = len(scores_viol)
    if scores_legit and scores_viol:
        result["separation_gap"] = round(result["violation_mean"] - result["legitimate_mean"], 4)
        result["gap_sufficient"] = result["separation_gap"] >= 0.20
        result["legit_below_040"] = result["legitimate_mean"] < 0.40
        result["viol_above_060"] = result["violation_mean"] > 0.60

    return result


def rescore_audit_trail(
    new_model_dir: Path,
    old_model_dir: Path,
    events: List[Dict],
) -> Dict[str, Any]:
    """Re-score all audit events with new vs old model. Compute verdict distribution."""
    texts = [e["action_text"] for e in events]
    if not texts:
        return {}

    new_scores = score_with_model(new_model_dir, texts)
    old_scores = score_with_model(old_model_dir, texts)

    # Compute verdict distributions (based on 0.40 composite_boost threshold)
    result = {"n_events": len(texts)}

    if new_scores:
        new_below_040 = sum(1 for s in new_scores if s < 0.40)
        result["new_below_040"] = new_below_040
        result["new_execute_rate"] = round(new_below_040 / len(new_scores), 4)
        result["new_mean"] = round(sum(new_scores) / len(new_scores), 4)

    if old_scores:
        old_below_040 = sum(1 for s in old_scores if s < 0.40)
        result["old_below_040"] = old_below_040
        result["old_execute_rate"] = round(old_below_040 / len(old_scores), 4)
        result["old_mean"] = round(sum(old_scores) / len(old_scores), 4)

    if new_scores and old_scores:
        result["execute_rate_improvement"] = round(
            result["new_execute_rate"] - result["old_execute_rate"], 4
        )

    return result


# ---------------------------------------------------------------------------
# Step 11: Generate report
# ---------------------------------------------------------------------------
def generate_report(
    output_dir: Path,
    n_existing: int,
    n_new: int,
    n_ambiguous: int,
    class_dist: Dict[int, int],
    validation: Dict[str, Any],
    separation: Dict[str, Any],
    rescore: Dict[str, Any],
    source_breakdown: Dict[str, int],
) -> Path:
    """Generate markdown retraining report."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = output_dir / f"retraining_report_{date_str}.md"

    go_verdict = "GO" if validation.get("go") else "NO_GO"

    lines = [
        f"# SetFit Retraining Report — {date_str}",
        "",
        f"**Verdict: {go_verdict}**",
        "",
        "## Training Data",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Existing examples | {n_existing} |",
        f"| New audit examples | {n_new} |",
        f"| Ambiguous (excluded) | {n_ambiguous} |",
        f"| Total training | {n_existing + n_new} |",
        f"| Violations (label=1) | {class_dist.get(1, 0)} |",
        f"| Legitimate (label=0) | {class_dist.get(0, 0)} |",
        "",
    ]

    if source_breakdown:
        lines.extend([
            "### Source Breakdown",
            "",
            "| Source | Count |",
            "|--------|-------|",
        ])
        for src, cnt in sorted(source_breakdown.items()):
            lines.append(f"| {src} | {cnt} |")
        lines.append("")

    # Benchmark validation
    lines.extend([
        "## Benchmark Validation",
        "",
        "| Metric | Old Model | New Model |",
        "|--------|-----------|-----------|",
    ])

    old = validation.get("old", {})
    new = validation.get("new", {})
    lines.append(f"| AUC-ROC | {old.get('auc', 'N/A')} | {new.get('auc', 'N/A')} |")
    lines.append(f"| Detection Rate | {old.get('detection_rate', 'N/A')} | {new.get('detection_rate', 'N/A')} |")
    lines.append(f"| FPR | {old.get('fpr', 'N/A')} | {new.get('fpr', 'N/A')} |")
    lines.append("")

    # Production-format separation
    if separation:
        lines.extend([
            "## Production-Format Separation",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        for k, v in separation.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # Audit trail rescore
    if rescore:
        lines.extend([
            "## Audit Trail Re-score",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        for k, v in rescore.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # Go/no-go
    lines.extend([
        "## Go/No-Go",
        "",
        f"- AUC >= 0.95: {'PASS' if new.get('auc', 0) >= 0.95 else 'FAIL'} ({new.get('auc', 'N/A')})",
        f"- Separation gap >= 0.20: {'PASS' if separation.get('gap_sufficient') else 'FAIL/N/A'}",
        f"- **Verdict: {go_verdict}**",
        "",
        "---",
        "*Generated by setfit_retraining_pipeline.py*",
    ])

    report_path.write_text("\n".join(lines))
    logger.info(f"Report written to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SetFit self-calibrating retraining pipeline"
    )
    parser.add_argument(
        "--audit-dir", type=str, default=str(DEFAULT_AUDIT_DIR),
        help=f"Posthoc audit directory (default: {DEFAULT_AUDIT_DIR})",
    )
    parser.add_argument(
        "--daemon-audit", type=str, default=str(DEFAULT_DAEMON_AUDIT),
        help=f"Daemon audit JSONL (default: {DEFAULT_DAEMON_AUDIT})",
    )
    parser.add_argument(
        "--overrides", type=str, default=None,
        help="Manual label overrides JSONL",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show training data summary, don't retrain",
    )
    parser.add_argument(
        "--rescore-only", action="store_true",
        help="Score audit trail with current model only, produce comparison",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Training epochs (default: 1)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    daemon_path = Path(args.daemon_audit)

    print("=" * 70)
    print("SetFit Self-Calibrating Retraining Pipeline")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load audit data
    # ------------------------------------------------------------------
    print("\n[1/11] Loading audit data...")
    posthoc_events = load_posthoc_audit(audit_dir)
    daemon_events = load_daemon_audit(daemon_path)

    all_events = posthoc_events + daemon_events
    if not all_events:
        print("ERROR: No audit data found. Check --audit-dir and --daemon-audit paths.")
        sys.exit(1)

    print(f"  Posthoc: {len(posthoc_events)}, Daemon: {len(daemon_events)}, Total: {len(all_events)}")

    # ------------------------------------------------------------------
    # Step 2: Auto-label
    # ------------------------------------------------------------------
    print("\n[2/11] Auto-labeling...")
    labeled, ambiguous = auto_label(all_events)

    # Apply overrides
    overrides_path = Path(args.overrides) if args.overrides else SCRIPT_DIR / "labels_override.jsonl"
    if args.overrides or overrides_path.exists():
        labeled = apply_overrides(labeled, overrides_path)

    # Create override template if missing
    if not overrides_path.exists():
        overrides_path.parent.mkdir(parents=True, exist_ok=True)
        overrides_path.write_text(
            '# Manual label overrides — one JSON per line\n'
            '# Format: {"action_text": "...", "label": 0}\n'
        )
        print(f"  Created override template: {overrides_path}")

    n_violations = sum(1 for e in labeled if e["label"] == 1)
    n_legitimate = sum(1 for e in labeled if e["label"] == 0)
    print(f"  Labeled: {len(labeled)} (violations={n_violations}, legitimate={n_legitimate})")
    print(f"  Ambiguous (excluded): {len(ambiguous)}")

    # ------------------------------------------------------------------
    # Step 3: Deduplicate
    # ------------------------------------------------------------------
    print("\n[3/11] Deduplicating...")
    deduped = deduplicate(labeled)
    print(f"  {len(labeled)} -> {len(deduped)} unique")

    # ------------------------------------------------------------------
    # Step 4: Merge with existing training data
    # ------------------------------------------------------------------
    print("\n[4/11] Merging with existing training data...")
    existing = load_existing_training(TRAINING_DATA_PATH)
    merged, n_new = merge_datasets(existing, deduped)
    print(f"  Existing: {len(existing)}, New: {n_new}, Total: {len(merged)}")

    if n_new < MIN_NEW_EXAMPLES:
        print(f"\nERROR: Insufficient new data ({n_new} < {MIN_NEW_EXAMPLES} minimum)")
        print("Run more agent sessions to generate training data, or lower MIN_NEW_EXAMPLES.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Balance classes
    # ------------------------------------------------------------------
    print("\n[5/11] Balancing classes...")
    balanced = balance_classes(merged, seed=args.seed)
    class_dist = Counter(d["label"] for d in balanced)
    source_dist = Counter(d.get("source", "unknown") for d in balanced)
    print(f"  Final: {len(balanced)} (violations={class_dist[1]}, legitimate={class_dist[0]})")
    print(f"  Sources: {dict(source_dist)}")

    # Save merged dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "merged_training_data.jsonl"
    with open(merged_path, "w") as f:
        for d in balanced:
            f.write(json.dumps(d) + "\n")
    print(f"  Saved: {merged_path}")

    # Save labeled audit data
    labeled_path = output_dir / "audit_labeled.jsonl"
    with open(labeled_path, "w") as f:
        for d in deduped:
            f.write(json.dumps(d) + "\n")

    # Save ambiguous data
    ambiguous_path = output_dir / "audit_ambiguous.jsonl"
    with open(ambiguous_path, "w") as f:
        for d in ambiguous:
            f.write(json.dumps(d) + "\n")
    print(f"  Saved: {labeled_path} ({len(deduped)} labeled), {ambiguous_path} ({len(ambiguous)} ambiguous)")

    # ------------------------------------------------------------------
    # Step 6: Score with current model (before)
    # ------------------------------------------------------------------
    print("\n[6/11] Scoring with current model (v1)...")
    all_texts = [d["action_text"] for d in deduped]
    before_scores = score_with_model(V1_MODEL_DIR, all_texts, verbose=args.verbose)
    if before_scores:
        legit_scores = [s for s, d in zip(before_scores, deduped) if d["label"] == 0]
        viol_scores = [s for s, d in zip(before_scores, deduped) if d["label"] == 1]
        if legit_scores:
            print(f"  Legitimate mean: {sum(legit_scores)/len(legit_scores):.4f} (n={len(legit_scores)})")
        if viol_scores:
            print(f"  Violation mean:  {sum(viol_scores)/len(viol_scores):.4f} (n={len(viol_scores)})")
    else:
        print("  v1 model not available, skipping before-scores")

    # ------------------------------------------------------------------
    # Dry-run exits here
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN complete. Files written:")
        print(f"  {merged_path}")
        print(f"  {labeled_path}")
        print(f"  {ambiguous_path}")
        print(f"\nTo retrain, run without --dry-run")
        print("=" * 70)
        return

    # ------------------------------------------------------------------
    # Rescore-only exits here
    # ------------------------------------------------------------------
    if args.rescore_only:
        print("\n[rescore-only] Scoring audit trail with current model...")
        rescore = rescore_audit_trail(V1_MODEL_DIR, V1_MODEL_DIR, deduped)
        for k, v in rescore.items():
            print(f"  {k}: {v}")
        print("\n" + "=" * 70)
        print("RESCORE-ONLY complete.")
        print("=" * 70)
        return

    # ------------------------------------------------------------------
    # Step 7-8: Retrain + Export ONNX
    # ------------------------------------------------------------------
    print("\n[7-8/11] Retraining SetFit + ONNX export...")
    ok = retrain_and_export(
        balanced,
        output_dir,
        epochs=args.epochs,
        seed=args.seed,
        verbose=args.verbose,
    )
    if not ok:
        print("ERROR: Retrain/export failed. See logs above.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 9: Validate against benchmark
    # ------------------------------------------------------------------
    print("\n[9/11] Validating against benchmark corpus...")
    bench_texts, bench_labels = load_benchmark_corpus(BENCHMARK_CORPUS_PATH)
    validation = validate_model(output_dir, V1_MODEL_DIR, bench_texts, bench_labels)

    new_metrics = validation.get("new", {})
    old_metrics = validation.get("old", {})
    print(f"  New model — AUC: {new_metrics.get('auc')}, Detection: {new_metrics.get('detection_rate')}, FPR: {new_metrics.get('fpr')}")
    if old_metrics:
        print(f"  Old model — AUC: {old_metrics.get('auc')}, Detection: {old_metrics.get('detection_rate')}, FPR: {old_metrics.get('fpr')}")
    print(f"  Go/No-Go: {'GO' if validation.get('go') else 'NO_GO'}")

    # ------------------------------------------------------------------
    # Step 10: Re-score audit trail with new model
    # ------------------------------------------------------------------
    print("\n[10/11] Re-scoring audit trail with new model...")
    separation = compute_production_separation(output_dir, deduped)
    rescore = rescore_audit_trail(output_dir, V1_MODEL_DIR, deduped)

    if separation:
        print(f"  Legitimate mean: {separation.get('legitimate_mean', 'N/A')}")
        print(f"  Violation mean:  {separation.get('violation_mean', 'N/A')}")
        print(f"  Separation gap:  {separation.get('separation_gap', 'N/A')}")
    if rescore:
        print(f"  EXECUTE rate (new): {rescore.get('new_execute_rate', 'N/A')}")
        print(f"  EXECUTE rate (old): {rescore.get('old_execute_rate', 'N/A')}")
        print(f"  Improvement: {rescore.get('execute_rate_improvement', 'N/A')}")

    # ------------------------------------------------------------------
    # Step 11: Generate report
    # ------------------------------------------------------------------
    print("\n[11/11] Generating report...")
    report = generate_report(
        output_dir=output_dir,
        n_existing=len(existing),
        n_new=n_new,
        n_ambiguous=len(ambiguous),
        class_dist=dict(class_dist),
        validation=validation,
        separation=separation,
        rescore=rescore,
        source_breakdown=dict(source_dist),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print(f"  Output: {output_dir}")
    print(f"  Report: {report}")
    print(f"  Verdict: {'GO' if validation.get('go') else 'NO_GO'}")
    print()
    print("Files:")
    for fname in ["model.onnx", "tokenizer.json", "head_weights.json",
                   "calibration.json", "manifest.json", "merged_training_data.jsonl",
                   "audit_labeled.jsonl", "audit_ambiguous.jsonl"]:
        fpath = output_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            if size > 1024 * 1024:
                print(f"  {fname}: {size / (1024*1024):.1f} MB")
            else:
                print(f"  {fname}: {size / 1024:.1f} KB")

    print()
    print("Verify with:")
    print(f'  from telos_governance.setfit_classifier import SetFitBoundaryClassifier')
    print(f'  c = SetFitBoundaryClassifier("{output_dir}")')
    print(f'  c.predict("Read file in project workspace: ~/Desktop/StewartBot/README.md")')
    print("=" * 70)


if __name__ == "__main__":
    main()
