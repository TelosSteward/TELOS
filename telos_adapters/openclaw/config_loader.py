"""
OpenClaw Config Loader — Loads openclaw.yaml and constructs the AgenticPA.

Bridges the YAML configuration (templates/openclaw.yaml) to the TELOS
governance engine by constructing an AgenticPA with all embeddings
pre-computed. This is done once at daemon startup (cold start: 3-5s)
and cached for the lifetime of the process.

The loader handles:
    1. Finding the openclaw.yaml config (project-local -> env var -> user-global)
    2. Loading and validating via telos_governance.config.load_config()
    3. Constructing AgenticPA via create_from_template() with embedding function
    4. Initializing AgenticFidelityEngine with the PA + violation keywords

Regulatory traceability:
    - IEEE 7000-2021: The PA purpose statement in openclaw.yaml serves as the
      Ethical Value Register (EVR), encoding 5 ethical values: user direction,
      scope limitation, authorization, data protection, boundary respect
    - SAAI claim TELOS-SAAI-011: All boundaries in openclaw.yaml trace to
      documented CVEs/incidents with provenance chain
    - EU AI Act Art. 9: Config validation ensures governance config integrity
    See: research/openclaw_regulatory_mapping.md §4
"""

import logging
import os
from pathlib import Path
from typing import Optional

from telos_governance.config import AgentConfig, load_config
from telos_governance.pa_signing import verify_config

logger = logging.getLogger(__name__)

# Config search order (Prasad CLI UX specification, M0)
CONFIG_SEARCH_PATHS = [
    "telos-openclaw.yaml",                    # Project-local
    ".telos/openclaw.yaml",                   # Project-local dotdir
    "templates/openclaw.yaml",                # Repository templates
]

ENV_CONFIG_VAR = "TELOS_OPENCLAW_CONFIG"
USER_GLOBAL_PATH = Path.home() / ".config" / "telos" / "openclaw.yaml"


def find_config(project_dir: Optional[str] = None) -> Optional[Path]:
    """Find the OpenClaw governance config file.

    Search order:
        1. TELOS_OPENCLAW_CONFIG environment variable
        2. Project-local paths (relative to project_dir or cwd)
        3. User-global path (~/.config/telos/openclaw.yaml)
        4. Built-in template (templates/openclaw.yaml in TELOS package)

    Args:
        project_dir: Project directory to search. Defaults to cwd.

    Returns:
        Path to config file, or None if not found.
    """
    # 1. Environment variable
    env_path = os.environ.get(ENV_CONFIG_VAR)
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            logger.info(f"Config from {ENV_CONFIG_VAR}: {p}")
            return p
        logger.warning(f"{ENV_CONFIG_VAR}={env_path} but file not found")

    # 2. Project-local
    base = Path(project_dir) if project_dir else Path.cwd()
    for rel in CONFIG_SEARCH_PATHS:
        p = (base / rel).resolve()
        if p.exists():
            logger.info(f"Config from project: {p}")
            return p

    # 3. User-global
    if USER_GLOBAL_PATH.exists():
        logger.info(f"Config from user-global: {USER_GLOBAL_PATH}")
        return USER_GLOBAL_PATH

    # 4. Built-in template
    builtin = Path(__file__).resolve().parent.parent.parent / "templates" / "openclaw.yaml"
    if builtin.exists():
        logger.info(f"Config from built-in template: {builtin}")
        return builtin

    return None


class OpenClawConfigLoader:
    """Loads OpenClaw governance configuration and constructs the PA + engine.

    Usage:
        loader = OpenClawConfigLoader()
        loader.load()  # or loader.load(path="custom/path.yaml")

        # Access the governance engine
        engine = loader.engine
        result = engine.score_action("Read the .env file")
    """

    def __init__(self):
        self._config: Optional[AgentConfig] = None
        self._pa = None
        self._engine = None
        self._embed_fn = None
        self._pa_verification: Optional[dict] = None

    def load(
        self,
        path: Optional[str] = None,
        project_dir: Optional[str] = None,
        embed_fn=None,
    ) -> "OpenClawConfigLoader":
        """Load configuration and initialize the governance engine.

        Args:
            path: Explicit path to config file. If None, auto-discovers.
            project_dir: Project directory for config search.
            embed_fn: Embedding function. If None, uses OnnxEmbeddingProvider.

        Returns:
            Self for chaining.

        Raises:
            FileNotFoundError: If no config file found.
            ConfigValidationError: If config is invalid.
        """
        # Resolve config path
        if path:
            config_path = Path(path).resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
        else:
            config_path = find_config(project_dir)
            if config_path is None:
                raise FileNotFoundError(
                    "No OpenClaw governance config found. Create one with:\n"
                    "  telos agent init --detect\n"
                    "Or set TELOS_OPENCLAW_CONFIG environment variable."
                )

        # Load and validate
        self._config = load_config(str(config_path))
        logger.info(
            f"Loaded OpenClaw config: {self._config.agent_name} "
            f"({len(self._config.boundaries)} boundaries, "
            f"{len(self._config.tools)} tools)"
        )

        # Initialize embedding function
        if embed_fn:
            self._embed_fn = embed_fn
        else:
            self._embed_fn = self._create_default_embed_fn()

        # Construct AgenticPA
        self._build_pa()

        # Initialize AgenticFidelityEngine
        self._build_engine()

        return self

    def verify_pa(self, config_path: str) -> dict:
        """Verify PA configuration TKey signature.

        Calls pa_signing.verify_config() and stores the result.
        This determines whether governance is active or inert.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Verification result dict with 'valid' and 'status' fields.
        """
        self._pa_verification = verify_config(config_path)
        return self._pa_verification

    def _create_default_embed_fn(self):
        """Create the default embedding function (ONNX MiniLM)."""
        try:
            from telos_core.embedding_provider import OnnxEmbeddingProvider
            provider = OnnxEmbeddingProvider()
            logger.info("Using OnnxEmbeddingProvider (MiniLM-L6-v2)")
            return provider.encode
        except ImportError:
            logger.warning(
                "ONNX provider unavailable, falling back to deterministic embeddings"
            )
            from telos_core.embedding_provider import DeterministicEmbeddingProvider
            return DeterministicEmbeddingProvider().encode

    def _build_pa(self) -> None:
        """Construct AgenticPA from loaded config.

        Two construction modes:
            "tool_grounded" — Uses PAConstructor to build per-tool centroids
                from canonical tool definitions (tool_semantics.py). Gate 1
                scores against per-tool centroids, Gate 2 uses behavioral
                constraints. Provenance: first-party platform documentation.

            "legacy" — Uses AgenticPA.create_from_template() with abstract
                purpose text. Default for all non-OpenClaw templates.
        """
        from telos_governance.agentic_pa import AgenticPA

        cfg = self._config

        # Convert boundary configs to the format create_from_template expects
        boundaries = [
            {"text": b.text, "severity": b.severity}
            for b in cfg.boundaries
        ]

        # Convert tool configs to objects with .name, .description, .risk_level
        from types import SimpleNamespace
        tools = [
            SimpleNamespace(
                name=t.name,
                description=t.description,
                risk_level=t.risk_level,
            )
            for t in cfg.tools
        ]

        # Check construction mode
        construction_mode = getattr(cfg, 'construction_mode', 'legacy')

        if construction_mode == "tool_grounded":
            # ── Tool-Grounded PA Construction ──
            from telos_governance.pa_constructor import PAConstructor

            constructor = PAConstructor(self._embed_fn)
            self._pa = constructor.construct(
                purpose=cfg.purpose,
                scope=cfg.scope,
                boundaries=boundaries,
                tools=tools,
                example_requests=cfg.example_requests,
                scope_example_requests=getattr(cfg, 'scope_example_requests', None),
                template_id=cfg.agent_id,
                safe_exemplars=cfg.safe_exemplars,
                max_chain_length=cfg.constraints.max_chain_length,
                max_tool_calls_per_step=cfg.constraints.max_tool_calls_per_step,
                escalation_threshold=cfg.constraints.escalation_threshold,
                require_human_above_risk=cfg.constraints.require_human_above_risk,
            )

            tool_centroid_count = len(getattr(self._pa, 'tool_centroids', {}))
            logger.info(
                f"AgenticPA constructed (tool_grounded mode): "
                f"{tool_centroid_count} tool centroids, "
                f"{len(self._pa.boundaries)} boundaries, "
                f"max_chain={self._pa.max_chain_length}"
            )
        else:
            # ── Legacy PA Construction ──
            self._pa = AgenticPA.create_from_template(
                purpose=cfg.purpose,
                scope=cfg.scope,
                boundaries=boundaries,
                tools=tools,
                embed_fn=self._embed_fn,
                example_requests=cfg.example_requests,
                scope_example_requests=getattr(cfg, 'scope_example_requests', None),
                template_id=cfg.agent_id,
                safe_exemplars=cfg.safe_exemplars,
                max_chain_length=cfg.constraints.max_chain_length,
                max_tool_calls_per_step=cfg.constraints.max_tool_calls_per_step,
                escalation_threshold=cfg.constraints.escalation_threshold,
                require_human_above_risk=cfg.constraints.require_human_above_risk,
            )

            logger.info(
                f"AgenticPA constructed (legacy mode): "
                f"{len(self._pa.boundaries)} boundaries, "
                f"{len(self._pa.tool_manifest)} tools, "
                f"max_chain={self._pa.max_chain_length}"
            )

    def _load_setfit_from_config(self):
        """Load SetFit classifier from config's setfit section.

        Returns SetFitBoundaryClassifier or None (graceful fallback).
        """
        cfg = self._config
        if not cfg.setfit or not cfg.setfit.enabled:
            return None

        try:
            from telos_governance.setfit_classifier import SetFitBoundaryClassifier

            model_dir = cfg.setfit.model_dir
            # Resolve relative paths against config file directory or project root
            if not os.path.isabs(model_dir) and cfg.config_path:
                config_dir = Path(cfg.config_path).resolve().parent
                # Try relative to config dir first, then project root
                candidate = config_dir / model_dir
                if not candidate.exists():
                    # Try relative to project root (2 levels up from templates/)
                    project_root = config_dir.parent
                    candidate = project_root / model_dir
                model_dir = str(candidate)

            # Resolve calibration path
            calibration_path = None
            if cfg.setfit.calibration:
                cal = cfg.setfit.calibration
                if not os.path.isabs(cal) and cfg.config_path:
                    config_dir = Path(cfg.config_path).resolve().parent
                    candidate = config_dir / cal
                    if not candidate.exists():
                        candidate = config_dir.parent / cal
                    cal = str(candidate)
                calibration_path = cal
            else:
                # Auto-detect calibration.json in model_dir
                auto_cal = Path(model_dir) / "calibration.json"
                if auto_cal.exists():
                    calibration_path = str(auto_cal)

            classifier = SetFitBoundaryClassifier(
                model_dir=model_dir,
                calibration_path=calibration_path,
                threshold=cfg.setfit.threshold,
                composite_boost=cfg.setfit.composite_boost,
            )
            logger.info(
                f"SetFit L1.5 loaded from config: {model_dir} "
                f"(threshold={cfg.setfit.threshold:.2f}, "
                f"asymmetric={cfg.setfit.asymmetric_override})"
            )
            return classifier
        except Exception as e:
            logger.warning(f"SetFit L1.5 load failed (graceful fallback): {e}")
            return None

    def _build_engine(self) -> None:
        """Initialize AgenticFidelityEngine from PA + optional SetFit."""
        from telos_governance.agentic_fidelity import AgenticFidelityEngine

        setfit_classifier = self._load_setfit_from_config()

        self._engine = AgenticFidelityEngine(
            embed_fn=self._embed_fn,
            pa=self._pa,
            violation_keywords=self._config.violation_keywords,
            setfit_classifier=setfit_classifier,
        )
        logger.info(
            "AgenticFidelityEngine initialized"
            + (" (with SetFit L1.5)" if setfit_classifier else " (no SetFit)")
        )

    @property
    def config(self) -> Optional[AgentConfig]:
        """The loaded configuration."""
        return self._config

    @property
    def pa(self):
        """The constructed AgenticPA."""
        return self._pa

    @property
    def engine(self):
        """The initialized AgenticFidelityEngine."""
        return self._engine

    @property
    def embed_fn(self):
        """The embedding function in use."""
        return self._embed_fn

    @property
    def is_loaded(self) -> bool:
        """Whether config has been loaded and engine initialized."""
        return self._engine is not None

    @property
    def pa_verification(self) -> Optional[dict]:
        """PA verification result from verify_pa()."""
        return self._pa_verification

    @property
    def governance_active(self) -> bool:
        """Whether governance scoring is active (PA is cryptographically verified).

        Returns True only if verify_pa() was called and returned VERIFIED status.
        Returns False if PA is unsigned, tampered, or verify_pa() was never called.
        """
        if self._pa_verification is None:
            return False
        return self._pa_verification.get("valid", False)
