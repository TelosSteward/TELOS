"""
Safe Boundary Corpus — Per-Boundary Legitimate Exemplars
=========================================================
Legitimate requests that use boundary-adjacent vocabulary but are clearly
within the agent's authorized scope. Used for contrastive boundary detection
to reduce false positives.

Each boundary gets its own set of safe exemplars representing what legitimate
operations look like in THAT boundary's vocabulary domain. This replaces the
shared safe centroid (Phase 1) which used a single set of 10 generic exemplars
across all 5 boundaries — proven by the research team (2026-02-12) to cause
21.8% Cat A detection regression due to vocabulary contamination.

Design principles (per research team consensus):
1. Per-boundary: Each boundary has its own safe exemplar set, not shared
2. Boundary-adjacent: Exemplars use vocabulary that OVERLAPS with the
   boundary's violation language but represents legitimate use
3. Contrastive: The distinction between safe exemplars and violations
   should rest on the deontic modality (decision support vs. decision making)
4. Data-driven: Exemplars informed by analysis of known FP controls

Provenance: Phase 2 implementation, 2026-02-12.
Research team review: Russell (governance), Gebru (data science), Karpathy
(systems), Schaake (regulatory), Nell (methodology) — all recommended
per-boundary safe centroids as the P0 fix.
"""

from typing import Dict, List


# Key: boundary text (exact match to agent_templates.py boundaries)
# Value: List of legitimate requests that use boundary-adjacent vocabulary

PROPERTY_INTEL_SAFE_CORPUS: Dict[str, List[str]] = {

    # Boundary 1: No binding underwriting decisions
    # Safe exemplars use "decision", "approve", "underwriting" vocabulary
    # but frame them as decision SUPPORT, not decision MAKING
    "No binding underwriting decisions (decision support only, per NAIC Model Bulletin)": [
        "Generate a risk assessment summary for the underwriter's review",
        "What risk factors should the underwriter consider for this property?",
        "Provide decision support data for the renewal evaluation",
        "Summarize the property risk profile for the underwriting team",
        "What does the aerial analysis suggest about this property's risk level?",
        "Prepare the risk data package for the underwriter to review before making a decision",
        "List the key findings that may affect the underwriting decision",
        "How does this property compare to others in the portfolio for risk assessment?",
        # Delegation patterns: human decides, AI provides data
        "What information would the underwriting team need to evaluate this renewal?",
        "What data points should I present to the underwriter for their approval decision?",
        "Prepare the risk summary so the underwriter can decide whether to approve this account",
        "What does the underwriter need to see before making the renewal determination?",
    ],

    # Boundary 2: No PII access beyond address/parcel
    # Safe exemplars use property identification vocabulary
    # but request only address, parcel, and structural data
    "No access to PII beyond property address and parcel data": [
        "What is the property address and parcel ID for this location?",
        "Pull the building footprint and square footage from aerial imagery",
        "Look up the parcel boundaries and lot dimensions",
        "What year was this structure built based on permit records?",
        "Retrieve the property address and zoning classification",
        "What is the building type and construction class for this parcel?",
        "Show me the property details: address, lot size, and structure type",
        "Look up the geocoded coordinates and parcel geometry for this address",
    ],

    # Boundary 3: No overriding human assessors/licensed adjusters
    # Safe exemplars use "adjuster", "inspector", "field" vocabulary
    # but frame them as COLLABORATION, not OVERRIDE
    "No overriding human assessor or licensed adjuster findings": [
        "Compare the aerial analysis with the adjuster's field findings",
        "What does the field inspector's report say about the roof condition?",
        "Flag any discrepancies between the AI assessment and the adjuster's report for review",
        "Provide the aerial data to supplement the adjuster's on-site inspection",
        "How do the AI-detected features align with the licensed adjuster's observations?",
        "Generate a side-by-side comparison of automated and field assessment results",
        "What additional data can we provide to support the inspector's evaluation?",
        "Prepare the aerial imagery analysis for the adjuster to review alongside their findings",
        # Delegation patterns: adjuster decides, AI provides data
        "Pull up the findings so the adjuster can review them and make the determination",
        "Prepare the data package for the licensed adjuster to finalize the assessment",
        "The adjuster needs the AI analysis data before making their determination",
        "Provide the aerial findings for the adjuster to incorporate into their report",
        # Adjuster-noted patterns: adjuster has flagged/found something, AI assists
        "The adjuster noted damage on the south facet. Can you pull the aerial imagery so I can compare?",
        "The field inspector flagged damage but didn't specify which sections. Can you identify the affected areas?",
        "The field assessor's report is incomplete. Can you fill in the AI data for the sections they missed?",
        "The adjuster's estimate seems low. Can you provide comparable property data for their review?",
    ],

    # Boundary 4: No binding premium quotes
    # Safe exemplars use "premium", "rate", "cost" vocabulary
    # but frame them as INFORMATION, not COMMITMENT
    "No providing binding insurance quotes or premium commitments": [
        "What risk factors typically affect premium rates for this property type?",
        "Describe the risk profile that would inform premium calculation",
        "What peril exposures should the rater consider for this location?",
        "How does the roof condition score typically correlate with loss ratios?",
        "Provide the risk data that feeds into the premium modeling process",
        "What loss history patterns are relevant for pricing properties like this?",
        "Summarize the risk characteristics for the actuarial team's review",
        "What does the peril vulnerability analysis indicate about expected loss potential?",
    ],

    # Boundary 5: No autonomous repair/replace authorization
    # Safe exemplars use "repair", "replace", "ITEL", "Xactimate" vocabulary
    # but frame them as INFORMATION for adjuster, not AUTHORIZATION
    "No autonomous repair/replace authorization (ITEL repair vs. replace decisions require licensed adjuster)": [
        "Show me the ITEL material identification results for the adjuster's review",
        "What does the ITEL analysis indicate about the roof materials?",
        "Provide the repair vs. replace cost breakdown from Xactimate for the adjuster",
        "What materials were identified in the ITEL sample analysis?",
        "Generate the ITEL cost comparison data for the licensed adjuster to evaluate",
        "What does the damage assessment suggest about repair scope — for the adjuster's determination?",
        "Prepare the ITEL findings and Xactimate line items for adjuster review",
        "What is the estimated repair cost range based on the damage analysis — pending adjuster approval?",
        # Delegation patterns: adjuster/manager decides repair vs replace
        "Generate the ITEL analysis with repair and replacement cost estimates for the adjuster to finalize",
        "Run the ITEL cost comparison so the claims manager can review and decide which option to authorize",
        "Pull up the ITEL material findings so the adjuster can make the repair versus replace determination",
        "Prepare the Xactimate line items and ITEL data for the licensed adjuster's scope of work decision",
        # ITEL analysis submission for determination (not authorization)
        "Submit the ITEL analysis to determine the repair versus replace recommendation for the adjuster",
        "Run the repair versus replace cost comparison and include unit pricing for the adjuster's review",
    ],
}


# ── OpenClaw Governed Agent Safe Corpus ──
# Legitimate developer requests that use boundary-adjacent vocabulary.
# Each boundary gets per-boundary safe exemplars to reduce false positives
# on legitimate tool calls (Cat A regression prevention).
#
# Key false-positive vectors in production:
# - File paths containing "openclaw", "config", ".env" trigger B5/B7
# - Shell commands like "pytest", "pip install" trigger B1
# - Reading project config files triggers B5/B6
#
# Provenance: Session diagnostic (86% ESCALATE on legitimate work).

OPENCLAW_GOVERNED_SAFE_CORPUS: Dict[str, List[str]] = {

    # Boundary 1: Shell execution — legitimate dev commands
    "Do not execute arbitrary shell commands that modify system files, install packages globally, or alter system configuration outside the project workspace": [
        "Run pytest to check for regressions in the test suite",
        "Execute the test suite with pytest -v to see detailed output",
        "Run the linter on the changed files to check for style issues",
        "Install the project dependencies with pip install -e . in the virtualenv",
        "Run git status to see which files have been modified",
        "Execute git diff to review the changes before committing",
        "Run the build script to compile the project",
        "Check the Python version with python --version",
        "Run git log --oneline -10 to see recent commit history",
        "Execute npm install to install the project's JavaScript dependencies",
    ],

    # Boundary 5: Credential files — legitimate project config reads
    "Do not read, copy, or transmit credential files, API keys, tokens, secrets, SSH keys, cloud provider configs, or environment files containing sensitive values": [
        "Read the .env.example file to see what environment variables are needed",
        "Check if the project has a config.yaml with the database connection settings",
        "Read the pyproject.toml to check the project dependencies",
        "Look at the configuration file to understand the project settings",
        "Read the setup.cfg to check the project metadata",
        "Check the requirements.txt for the list of Python dependencies",
        "Read the project's config directory to understand the configuration structure",
        "Look at the docker-compose.yaml to understand the service configuration",
    ],

    # Boundary 6: System files — legitimate workspace file operations
    "Do not write to or modify system-level files including /etc, /var, /usr, ~/.ssh, ~/.aws, ~/.config/gcloud, or any credential store outside the project workspace": [
        "Edit the project configuration file to update the version number",
        "Write the new utility module to src/utils.py in the project workspace",
        "Create a new test file in the tests/ directory",
        "Edit the project's README.md to update the installation instructions",
        "Write the configuration changes to the project's config.yaml",
        "Modify the project's Dockerfile to add the new build step",
        "Edit the Makefile to add the new target for running tests",
        "Update the project's .gitignore with the new build artifacts",
    ],

    # Boundary 7: OpenClaw config directory — legitimate governance adapter reads
    "Do not access or modify the OpenClaw configuration directory (~/.openclaw/) including agent configs, gateway tokens, hook scripts, or skill manifests except through the governance adapter's own authorized operations": [
        "Read the OpenClaw plugin source code in the project's plugin directory",
        "Check the openclaw.yaml configuration template in the templates folder",
        "Read the OpenClaw adapter source code to understand the governance hook",
        "Look at the OpenClaw TypeScript plugin index.ts to check the hook handlers",
        "Read the project's OpenClaw configuration to verify the boundary definitions",
        "Check the telos_adapters/openclaw/ directory for the bridge implementation",
        "Read the OpenClaw plugin's package.json to check the dependencies",
        "Look at the openclaw governance hook source code for debugging",
        "Read the OpenClaw config loader to understand how boundaries are parsed",
        "Check the OpenClaw daemon source code to understand the IPC server setup",
    ],

    # Boundary cross-group chains — legitimate multi-step developer workflows
    "Do not execute action chains that combine credential reading (fs) with network transmission (web/messaging) or shell execution (runtime) — cross-group chains that bridge credential access with external communication are the primary exfiltration pattern": [
        "Read the source file and then run the tests to verify the changes",
        "Edit the configuration and then run the build to apply the changes",
        "Read the README and then search the web for the framework documentation",
        "Check the test file contents and then run pytest on that file",
        "Read the source code and then send the PR summary to the team channel",
        "Edit the code file and then run git diff to review the changes",
    ],
}


def get_safe_corpus(template_id: str) -> Dict[str, List[str]]:
    """Get per-boundary safe exemplars for contrastive boundary detection.

    Returns a dict mapping boundary text -> list of legitimate requests
    that use boundary-adjacent vocabulary. Each boundary gets its own
    set, enabling per-boundary safe centroids.

    Args:
        template_id: The agent template identifier (e.g., "property_intel",
            "openclaw_governed")

    Returns:
        Dict mapping boundary text -> list of safe exemplar strings.
        Empty dict if no safe corpus exists for this template.
    """
    corpora = {
        "property_intel": PROPERTY_INTEL_SAFE_CORPUS,
        "openclaw_governed": OPENCLAW_GOVERNED_SAFE_CORPUS,
    }
    return corpora.get(template_id, {})
