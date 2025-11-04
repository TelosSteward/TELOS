#!/usr/bin/env python3
"""
Claude Project PA Controller
=============================

Establishes Primacy Attractor in .claude_project.md and updates it based
on ACTUAL TELOS measurements from external governance monitor.

Architecture:
1. External TELOS measures fidelity (dual_attractor.py)
2. This script updates .claude_project.md based on drift
3. .claude_project.md governs Claude Code behavior
4. Cycle repeats → ACTUAL runtime governance

This implements the "intervention controller" while external monitoring
provides the "measurement system" - complete dual attractor runtime.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))


class ClaudeProjectPAController:
    """
    Updates .claude_project.md based on TELOS measurements

    Orchestrated by Steward PM for intelligent intervention decisions
    """

    def __init__(self, project_file: str = ".claude_project.md", steward_pm=None):
        self.project_file = Path(project_file)
        self.session_pa = None
        self.current_fidelity = 1.0
        self.fidelity_history = []
        self.steward_pm = steward_pm  # Steward PM orchestrates interventions

    def establish_session_pa(self, user_pa: Dict[str, Any]):
        """
        Establish PA in .claude_project.md

        This becomes the "AI PA" that governs Claude Code behavior
        """
        self.session_pa = user_pa

        pa_section = self._generate_pa_section(user_pa)

        # Read existing .claude_project.md
        if self.project_file.exists():
            content = self.project_file.read_text()

            # Remove old PA section if exists
            if "## TELOS Session Governance" in content:
                parts = content.split("## TELOS Session Governance")
                content = parts[0].rstrip()

            # Append new PA section
            content += "\n\n" + pa_section

        else:
            content = pa_section

        # Write updated file
        self.project_file.write_text(content)

        print(f"✅ Session PA established in {self.project_file}")
        print(f"   Purpose: {user_pa['purpose'][0][:70]}...")

    def _generate_pa_section(self, user_pa: Dict[str, Any]) -> str:
        """Generate TELOS governance section for .claude_project.md"""

        purpose = user_pa['purpose'][0] if isinstance(user_pa['purpose'], list) else user_pa['purpose']
        scope_items = user_pa['scope'] if isinstance(user_pa['scope'], list) else [user_pa['scope']]
        boundaries = user_pa['boundaries'] if isinstance(user_pa['boundaries'], list) else [user_pa['boundaries']]

        section = f"""---

## TELOS Session Governance

**Active Primacy Attractor - Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}**

### Session Purpose (User PA)

{purpose}

### Scope (What's In-Bounds)

{chr(10).join(f'- {item}' for item in scope_items)}

### Boundaries (What's Out-of-Bounds)

{chr(10).join(f'- {item}' for item in boundaries)}

### Current Session Metrics

- **Mean Fidelity:** {self.current_fidelity:.3f}
- **Status:** {'✅ On track' if self.current_fidelity >= 0.7 else '🚨 Drift detected'}
- **Last Update:** {datetime.now().isoformat()}

### Claude Instructions (AI PA)

You are operating under TELOS dual primacy attractor governance:

**Your Role (AI PA):**
- Help the user accomplish the session purpose above
- Stay strictly within defined scope
- Enforce boundaries proactively
- Self-monitor fidelity on every response

**Fidelity Self-Check Protocol:**

Before each response, ask yourself:
1. Does this advance the session purpose?
2. Is this within defined scope?
3. Am I respecting all boundaries?
4. Would external TELOS measure this as high fidelity?

**If fidelity might be low (<0.7):**
- Stop and refocus on purpose
- Remind user of scope boundaries if they're drifting
- Suggest course correction back to critical path

**Intervention Protocol:**

If you detect drift:
- **Minor drift (F ~0.6-0.7):** Gentle reminder of purpose/scope
- **Significant drift (F <0.6):** Strong refocus suggestion
- **Critical drift (F <0.5):** Explicit intervention ("This is off-track from our session purpose: ...")

**Current Focus Areas (High Priority):**
{chr(10).join(f'- {item}' for item in scope_items[:3])}

**Active Blockers to Avoid:**
{chr(10).join(f'- {item}' for item in boundaries)}

---

**External TELOS Monitor Active**

This session is being monitored by external TELOS governance (claude_code_governance_monitor.py).
Metrics are measured turn-by-turn using actual dual PA architecture.
This .claude_project.md file will be updated if drift is detected.
"""

        return section

    def update_based_on_metrics(self, turn_metrics: Dict[str, Any], turn_context: Dict[str, Any] = None):
        """
        Update .claude_project.md based on latest TELOS measurements

        This is the "intervention" based on external measurement
        Orchestrated by Steward PM for intelligent decisions
        """
        user_fidelity = turn_metrics.get('user_fidelity', 1.0)
        ai_fidelity = turn_metrics.get('ai_fidelity', 1.0)
        overall_pass = turn_metrics.get('overall_pass', True)
        dominant_failure = turn_metrics.get('dominant_failure')

        self.fidelity_history.append(user_fidelity)
        self.current_fidelity = sum(self.fidelity_history) / len(self.fidelity_history)

        # Steward PM decides intervention strategy
        if self.steward_pm:
            intervention_recommendation = self.steward_pm.recommend_intervention(
                user_fidelity=user_fidelity,
                ai_fidelity=ai_fidelity,
                overall_pass=overall_pass,
                dominant_failure=dominant_failure,
                fidelity_history=self.fidelity_history,
                turn_context=turn_context
            )

            if intervention_recommendation:
                return self._apply_steward_intervention(intervention_recommendation)

        # Fallback to basic intervention logic if no Steward PM
        needs_update = False
        intervention_message = None

        if not overall_pass:
            needs_update = True

            if dominant_failure == 'user':
                intervention_message = f"""
🚨 DRIFT DETECTED (Turn {len(self.fidelity_history)})

**User PA Violation:** Response drifted from session purpose
**User Fidelity:** {user_fidelity:.3f}
**AI Fidelity:** {ai_fidelity:.3f}

**Intervention:**
- Refocus on: {self.session_pa['purpose'][0][:100]}...
- Avoid topics outside scope
- Next response MUST be high fidelity (F >0.7)

**Root Cause Analysis:**
User PA fidelity dropped below threshold. This indicates:
- Response didn't advance session purpose, OR
- Response went outside defined scope, OR
- Response violated a boundary constraint

**Corrective Action:**
- Review session purpose above
- Ensure next response is tightly aligned
- If user is drifting, remind them of purpose/scope
"""

            elif dominant_failure == 'ai':
                intervention_message = f"""
🚨 DRIFT DETECTED (Turn {len(self.fidelity_history)})

**AI PA Violation:** Response violated AI role constraints
**User Fidelity:** {user_fidelity:.3f}
**AI Fidelity:** {ai_fidelity:.3f}

**Intervention:**
- Your responses are not staying in role
- Review "Claude Instructions (AI PA)" section above
- Adjust tone/approach to match intended AI role

**Root Cause Analysis:**
AI PA fidelity dropped. This indicates:
- Response tone/style didn't match intended role, OR
- Response violated how you should help (too passive/aggressive), OR
- Response didn't properly support user's purpose

**Corrective Action:**
- Re-read AI PA instructions above
- Adjust approach to match role constraints
- Ensure you're helping in the right way
"""

            elif dominant_failure == 'both':
                intervention_message = f"""
🚨 CRITICAL DRIFT DETECTED (Turn {len(self.fidelity_history)})

**BOTH PAs Violated**
**User Fidelity:** {user_fidelity:.3f}
**AI Fidelity:** {ai_fidelity:.3f}

**IMMEDIATE INTERVENTION REQUIRED**

**What Went Wrong:**
- Response completely off-track from session purpose
- Response also violated AI role constraints
- Double failure = conversation has derailed

**Corrective Action:**
- STOP current direction immediately
- Re-read session purpose and scope above
- Explicitly ask user: "Should we refocus on [session purpose]?"
- Get conversation back on critical path
"""

        elif self.current_fidelity < 0.75:
            # Trend warning (not critical yet, but declining)
            needs_update = True
            intervention_message = f"""
⚠️  FIDELITY TREND WARNING (Turn {len(self.fidelity_history)})

**Mean Session Fidelity:** {self.current_fidelity:.3f} (below 0.75 target)
**Latest:** {user_fidelity:.3f}

**Not critical yet, but trending down**

**Preventive Action:**
- Review session purpose and scope
- Ensure next few responses are highly aligned
- Proactively steer conversation back to critical path
- Avoid tangents or nice-to-have features
"""

        # Update file if needed
        if needs_update and intervention_message:
            self._inject_intervention(intervention_message)

            print(f"\n{'='*60}")
            print(f"📝 .claude_project.md UPDATED - Intervention Applied")
            print(f"{'='*60}")
            print(intervention_message)
            print(f"{'='*60}\n")

        return needs_update

    def _inject_intervention(self, message: str):
        """Inject intervention message at top of PA section"""

        content = self.project_file.read_text()

        if "## TELOS Session Governance" in content:
            parts = content.split("## TELOS Session Governance")

            updated_section = f"""## TELOS Session Governance

{message}

**Active Primacy Attractor - Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}**
"""

            # Preserve rest of PA section after "Active Primacy Attractor"
            if len(parts) > 1:
                pa_section = parts[1]
                if "**Active Primacy Attractor" in pa_section:
                    pa_parts = pa_section.split("**Active Primacy Attractor", 1)
                    if len(pa_parts) > 1:
                        rest = "**Active Primacy Attractor" + pa_parts[1]
                        updated_section += rest.split("\n", 1)[1]  # Skip first line

            content = parts[0] + updated_section

            self.project_file.write_text(content)


def main():
    """Demo: Establish PA and simulate intervention"""

    controller = ClaudeProjectPAController()

    # Establish session PA
    user_pa = {
        'purpose': [
            "Guide TELOS development toward February 2026 institutional "
            "deployment and grant readiness through systematic progress "
            "on validation, platform infrastructure, and grant applications"
        ],
        'scope': [
            "Grant application preparation (LTFF, Emergent Ventures, EU)",
            "Validation study completion and metrics generation",
            "Observatory and TELOSCOPE development",
            "Platform infrastructure and deployment readiness",
            "Documentation for institutional partnerships"
        ],
        'boundaries': [
            "No consumer product features (institutional focus only)",
            "Protect proprietary IP (Dual PA, DMAIC, SPC, OriginMind, Telemetric Keys)",
            "Focus on critical path over nice-to-have features",
            "Prioritize blockers (pilot conversations, grant deadlines)"
        ]
    }

    print("\n🔭 Claude Project PA Controller")
    print("="*60)

    controller.establish_session_pa(user_pa)

    print("\n✅ Session PA established in .claude_project.md")
    print("\nNow run: python3 claude_code_governance_monitor.py")
    print("External monitor will measure fidelity and trigger updates here.")


if __name__ == "__main__":
    main()
