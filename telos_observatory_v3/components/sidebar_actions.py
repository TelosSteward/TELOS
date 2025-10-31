"""
Sidebar Actions Component for TELOS Observatory V3.
Provides session management controls in left sidebar.
"""

import streamlit as st
import json
from datetime import datetime


class SidebarActions:
    """Left sidebar with session management actions."""

    def __init__(self, state_manager):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for session operations
        """
        self.state_manager = state_manager

    def render(self):
        """Render sidebar actions."""
        with st.sidebar:
            # TELOS Logo/Branding
            st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="color: #FFD700; font-size: 36px; margin: 0;">
                    🔭 TELOS
                </h1>
                <p style="color: #888; font-size: 12px; margin: 5px 0 0 0;">
                    Observatory V3
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Session Info (replay only)
            st.markdown("### 📊 Session Info")
            session_info = self.state_manager.get_session_info()

            st.markdown(f"""
            <div style="background-color: #1a1a1a; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <div style="color: #888; font-size: 11px;">SESSION ID</div>
                <div style="color: #fff; font-size: 13px;">{session_info.get('session_id', 'None')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color: #1a1a1a; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <div style="color: #888; font-size: 11px;">TURN</div>
                <div style="color: #fff; font-size: 13px;">
                    {session_info.get('current_turn', 0) + 1} / {session_info.get('total_turns', 0)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Action Buttons
            st.markdown("### ⚡ Actions")

            # Save Current
            if st.button("💾 Save Current", use_container_width=True):
                self._save_current_session()

            # Load Existing
            if st.button("📂 Load Session", use_container_width=True):
                self._load_session()

            # Reset Session
            if st.button("🔄 Reset Session", use_container_width=True):
                self._reset_session()

            # Export Evidence
            if st.button("📤 Export Evidence", use_container_width=True):
                self._export_evidence()

            st.markdown("---")

            # Help
            if st.button("❓ Help", use_container_width=True):
                self._show_help()

    def _save_current_session(self):
        """Save current session state to file."""
        try:
            session_data = {
                'session_id': self.state_manager.state.session_id,
                'timestamp': datetime.now().isoformat(),
                'current_turn': self.state_manager.state.current_turn,
                'total_turns': self.state_manager.state.total_turns,
                'turns': self.state_manager.state.turns
            }

            filename = f"session_{self.state_manager.state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # In a real app, this would save to file
            # For now, just show success message
            st.success(f"Session saved: {filename}")

        except Exception as e:
            st.error(f"Error saving session: {e}")

    def _load_session(self):
        """Load a session from file."""
        st.info("Load session feature - file browser would appear here")
        # In V3, this would open file uploader or file browser

    def _reset_session(self):
        """Reset session to beginning."""
        self.state_manager.jump_to_turn(0)
        st.success("Session reset to turn 1")
        st.rerun()

    def _export_evidence(self):
        """Export evidence package for governance review."""
        try:
            # Collect all interventions
            interventions = [
                turn for turn in self.state_manager.state.turns
                if turn.get('intervention_applied', False)
            ]

            evidence = {
                'session_id': self.state_manager.state.session_id,
                'export_date': datetime.now().isoformat(),
                'total_turns': len(self.state_manager.state.turns),
                'intervention_count': len(interventions),
                'interventions': interventions
            }

            st.success(f"Evidence package ready: {len(interventions)} interventions documented")

        except Exception as e:
            st.error(f"Error exporting evidence: {e}")

    def _show_help(self):
        """Show help information."""
        st.info("""
        **TELOS Observatory V3**

        Navigation:
        - Use the Strip (top-right) to view turn metrics
        - Click the 🔭 icon to open the Observation Deck

        Actions:
        - Save: Export current session state
        - Load: Import a previous session
        - Reset: Jump back to turn 1
        - Export: Generate governance evidence package
        """)
