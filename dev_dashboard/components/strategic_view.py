"""Strategic View Component - Visualizes Steward PM data WITHOUT using LLM tokens."""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import os


class StrategicView:
    """
    Display strategic TELOS data (partnerships, grants, etc.) from Steward PM.
    This is pure visualization - NO LLM calls, NO token usage.
    """

    def __init__(self):
        # Load from governance system (if exists)
        self.data_dir = Path(os.getenv('TELOS_DATA_DIR', '.telos_governance'))
        self.load_strategic_data()

    def load_strategic_data(self):
        """Load saved strategic data from governance system - synced from Steward PM."""
        self.partnerships = {}
        self.grants = {}
        self.priorities = []
        self.risks = []
        self.validations = {}

        # Try to load saved state from governance system
        state_file = self.data_dir / 'state.pkl'
        if state_file.exists():
            try:
                import pickle
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.partnerships = state.get('partnerships', {})
                    self.grants = state.get('grants', {})
                    self.priorities = state.get('priorities', [])
                    self.risks = state.get('risks', [])
                    self.validations = state.get('validations', {})
            except:
                pass

        # If no saved data, show instructions
        if not self.partnerships and not self.grants:
            self.no_data = True
        else:
            self.no_data = False

    def render(self):
        """Render strategic view - NO TOKENS USED, just visualization!"""
        st.markdown("## 🎯 TELOS Strategic Overview")
        st.markdown("*Visualizing Steward PM data - No LLM tokens used!*")

        # Check if we have data
        if self.no_data:
            st.warning("⚠️ No strategic data available yet")
            st.info("""
            **To populate this view, run in terminal:**
            ```bash
            # Analyze partnerships
            python3 steward_pm.py partnerships

            # Analyze grants
            python3 steward_pm.py grants

            # Get priorities
            python3 steward_pm.py next

            # Then refresh this dashboard
            ```
            """)
            return

        tabs = st.tabs(["🤝 Partnerships", "💰 Grants", "🎯 Priorities", "⚠️ Risks"])

        with tabs[0]:
            self.render_partnerships()

        with tabs[1]:
            self.render_grants()

        with tabs[2]:
            self.render_priorities()

        with tabs[3]:
            self.render_risks()

        # Update button
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh from Steward PM", use_container_width=True):
                st.info("Run 'python3 steward_pm.py partnerships' in terminal to update data")
        with col2:
            if st.button("📊 Export Report", use_container_width=True):
                self.export_report()

    def render_partnerships(self):
        """Display partnership status visually."""
        st.markdown("### 🤝 Institutional Partnerships")

        if not self.partnerships:
            st.info("No partnership data available. Run `python3 steward_pm.py partnerships`")
            return

        for partner, data in self.partnerships.items():
            # Handle different data structures gracefully
            if isinstance(data, dict):
                col1, col2, col3 = st.columns([2, 3, 3])

                with col1:
                    status = data.get('status', 'Unknown')
                    status_color = {
                        'Critical': '🔴',
                        'In Progress': '🟡',
                        'Initiated': '🟢',
                        'On Track': '🟢'
                    }.get(status, '⚪')
                    st.markdown(f"### {status_color} {partner}")

                with col2:
                    progress = data.get('progress', 0)
                    st.markdown("**Progress**")
                    st.progress(progress / 100 if progress <= 100 else 1.0)
                    st.caption(f"{progress}% complete")

                with col3:
                    next_action = data.get('next_action', 'No action defined')
                    st.markdown("**Next Action**")
                    st.info(next_action)
            else:
                # Simple text display for non-dict data
                st.markdown(f"**{partner}:** {data}")

            st.markdown("---")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Partnerships", len(self.partnerships))
        with col2:
            avg_progress = sum(p['progress'] for p in self.partnerships.values()) / len(self.partnerships)
            st.metric("Average Progress", f"{avg_progress:.0f}%")
        with col3:
            critical = sum(1 for p in self.partnerships.values() if p['status'] == 'Critical')
            st.metric("Critical Items", critical, delta="Needs attention" if critical > 0 else None)

    def render_grants(self):
        """Display grant application status."""
        st.markdown("### 💰 Grant Applications")

        for grant, data in self.grants.items():
            col1, col2, col3, col4 = st.columns([2, 2, 3, 2])

            with col1:
                st.markdown(f"**{grant}**")
                st.caption(data['status'])

            with col2:
                st.markdown("**Deadline**")
                st.warning(data['deadline'])

            with col3:
                st.markdown("**Requirements Met**")
                st.progress(data['requirements_met'] / 100)
                st.caption(f"{data['requirements_met']}%")

            with col4:
                # Days until deadline (simplified)
                st.markdown("**Priority**")
                if data['requirements_met'] < 50:
                    st.error("High")
                elif data['requirements_met'] < 75:
                    st.warning("Medium")
                else:
                    st.success("On Track")

        # Grant summary
        st.markdown("---")
        st.markdown("### 📊 Grant Readiness")

        ltff_ready = self.grants.get('LTFF', {}).get('requirements_met', 0)

        if ltff_ready < 60:
            st.error(f"⚠️ LTFF Readiness: {ltff_ready}% - Need 60+ validation studies!")
        else:
            st.success(f"✅ LTFF Readiness: {ltff_ready}% - Meeting validation requirement")

    def render_risks(self):
        """Display risks and blockers from Steward PM."""
        st.markdown("### ⚠️ Risks & Blockers")

        if not self.risks:
            st.info("No risk data available. Run `python3 steward_pm.py risks`")
            return

        # Display risks as a simple list
        for i, risk in enumerate(self.risks, 1):
            if isinstance(risk, dict):
                severity = risk.get('severity', 'medium')
                severity_icon = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(severity, '⚪')

                st.markdown(f"{severity_icon} **Risk {i}:** {risk.get('description', risk)}")
                if 'mitigation' in risk:
                    st.caption(f"   Mitigation: {risk['mitigation']}")
            else:
                st.markdown(f"• {risk}")

            st.markdown("---")

    def render_priorities(self):
        """Display current priorities from Steward PM."""
        st.markdown("### 🎯 Strategic Priorities")

        if not self.priorities:
            st.info("No priorities data available. Run `python3 steward_pm.py next`")
            return

        # Display priorities - handle both list and dict structures
        for i, priority in enumerate(self.priorities[:5], 1):  # Top 5 priorities
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"#{i}")

            if isinstance(priority, dict):
                st.markdown(f"{medal} **{priority.get('item', priority)}**")
                if 'rationale' in priority:
                    st.caption(f"   Rationale: {priority['rationale']}")
                if 'next_action' in priority:
                    st.info(f"   Next: {priority['next_action']}")
            else:
                # Simple text priority
                st.markdown(f"{medal} **{priority}**")

            st.markdown("---")

    def export_report(self):
        """Export strategic report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "partnerships": self.partnerships,
            "grants": self.grants,
            "priorities": self.priorities,
            "risks": self.risks,
            "validations": self.validations
        }

        report_file = f"strategic_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        st.success(f"📊 Report exported: {report_file}")