"""
Observatory V2 - Basic Sidebar
================================

Left sidebar with basic session management functions.

Features:
- Save current session
- Load existing session
- Reset session
- Export evidence report
"""

import streamlit as st


class BasicSidebar:
    """
    Renders basic sidebar with session management.

    Pure component - emits actions through state manager.
    """

    def __init__(self, state_manager):
        """
        Initialize sidebar.

        Args:
            state_manager: StateManager instance
        """
        self.state_manager = state_manager

    def render(self):
        """Render sidebar content."""
        with st.sidebar:
            st.markdown("## 🔭 Observatory")
            st.markdown("---")

            # Session info
            session_info = self.state_manager.get_session_info()
            st.caption(f"**Session:** {session_info['session_id'][:12]}...")
            st.caption(f"**Turns:** {session_info['total_turns']}")

            st.markdown("---")

            # Save Session
            st.markdown("### 💾 Save")
            if st.button("Save Current Session", key="save_session", use_container_width=True):
                st.success("Session saved!")
                st.caption("*(Save functionality pending)*")

            st.markdown("---")

            # Load Session
            st.markdown("### 📂 Load")
            uploaded_file = st.file_uploader(
                "Load Session File",
                type=['json'],
                key="load_session"
            )
            if uploaded_file:
                st.info("Loading session...")
                st.caption("*(Load functionality pending)*")

            st.markdown("---")

            # Reset
            st.markdown("### 🔄 Reset")
            if st.button("Reset to Start", key="reset_session", use_container_width=True):
                self.state_manager.jump_to_turn(0)
                self.state_manager.stop_playback()
                st.rerun()

            st.markdown("---")

            # Export
            st.markdown("### 📥 Export")
            export_format = st.selectbox(
                "Format",
                options=["Evidence Report (PDF)", "Raw Data (JSON)", "CSV"],
                key="export_format"
            )
            if st.button("Export", key="export_button", use_container_width=True):
                st.success(f"Exporting as {export_format}")
                st.caption("*(Export functionality pending)*")

            st.markdown("---")

            # About
            with st.expander("ℹ️ About"):
                st.caption("""
                **TELOS Observatory V2**

                Frame-by-frame AI governance analysis platform.

                Built with modular architecture for extensibility.
                """)
