"""
Observatory V2 - TELOSCOPE Controls
====================================

Bottom control panel for navigating through turns.

Features:
- Previous/Play/Next buttons
- Timeline scrubber
- Speed controls
- Tools section (Deep Research, Steward, Export)

Design: Fixed bottom panel with gold accent theming.
"""

import streamlit as st
import time


class TELOSCOPEControls:
    """
    Renders TELOSCOPE navigation controls.

    Pure component - reads state, emits actions through state manager.
    """

    def __init__(self, state_manager):
        """
        Initialize TELOSCOPE controls.

        Args:
            state_manager: StateManager instance
        """
        self.state_manager = state_manager

    def render(self):
        """Render TELOSCOPE controls at bottom of page."""
        self._render_styles()
        self._render_controls_panel()
        self._handle_autoplay()

    def _render_styles(self):
        """Render CSS styles for TELOSCOPE."""
        st.markdown("""
            <style>
            .teloscope-container {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(0, 0, 0, 0.95);
                backdrop-filter: blur(15px);
                border-top: 2px solid rgba(255, 215, 0, 0.4);
                padding: 1.5rem 2rem;
                z-index: 998;
                box-shadow: 0 -6px 30px rgba(0, 0, 0, 0.6);
            }

            .teloscope-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }

            .teloscope-title {
                font-size: 1.1rem;
                font-weight: bold;
                color: #FFD700;
            }

            .gold-text {
                color: #FFD700;
            }

            /* Main content padding to accommodate TELOSCOPE */
            .main .block-container {
                padding-bottom: 180px !important;
            }
            </style>
        """, unsafe_allow_html=True)

    def _render_controls_panel(self):
        """Render the controls panel content."""
        st.markdown('<div class="teloscope-container">', unsafe_allow_html=True)

        # Header
        col_title, col_tools = st.columns([2, 3])
        with col_title:
            st.markdown('<div class="teloscope-title">🔭 TELOSCOPE Controls</div>', unsafe_allow_html=True)
        with col_tools:
            self._render_tools_section()

        # Navigation controls
        self._render_navigation()

        # Timeline scrubber
        self._render_timeline()

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_tools_section(self):
        """Render tools buttons (right side of header)."""
        tool_cols = st.columns([1, 1, 1, 1, 1])

        with tool_cols[0]:
            if st.button("📊 Deep Research", key="teloscope_research", help="Open Deep Research view"):
                st.info("Deep Research view (opens in new tab)")

        with tool_cols[1]:
            if st.button("🤝 Steward", key="teloscope_steward", help="Toggle Steward tool"):
                self.state_manager.toggle_component('steward')
                st.rerun()

        with tool_cols[2]:
            if st.button("📥 Export", key="teloscope_export", help="Export session data"):
                st.info("Export functionality (generates evidence report)")

        with tool_cols[3]:
            # Speed control
            speed = st.selectbox(
                "Speed",
                options=[0.5, 1.0, 2.0],
                index=1,
                key="playback_speed",
                help="Playback speed"
            )
            self.state_manager.set_playback_speed(speed)

        with tool_cols[4]:
            pass  # Reserved for future tools

    def _render_navigation(self):
        """Render navigation buttons (Prev/Play/Next)."""
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 6])

        with nav_col1:
            if st.button("⏮", key="teloscope_prev", help="Previous Turn"):
                if self.state_manager.previous_turn():
                    st.rerun()

        with nav_col2:
            # Play/Pause button
            is_playing = self.state_manager.is_playing()
            play_icon = "⏸" if is_playing else "▶"
            if st.button(play_icon, key="teloscope_play", help="Play/Pause"):
                if is_playing:
                    self.state_manager.stop_playback()
                else:
                    self.state_manager.start_playback()
                st.rerun()

        with nav_col3:
            if st.button("⏭", key="teloscope_next", help="Next Turn"):
                if self.state_manager.next_turn():
                    st.rerun()

        with nav_col4:
            # Stop button
            if st.button("⏹", key="teloscope_stop", help="Stop"):
                self.state_manager.stop_playback()
                self.state_manager.jump_to_turn(0)
                st.rerun()

        with nav_col5:
            # Turn counter
            session_info = self.state_manager.get_session_info()
            st.markdown(
                f'<div style="text-align: center; color: #FFD700; font-size: 1rem; padding-top: 0.25rem;">'
                f'Turn {session_info["current_turn"] + 1} / {session_info["total_turns"]}'
                f'</div>',
                unsafe_allow_html=True
            )

    def _render_timeline(self):
        """Render timeline scrubber."""
        session_info = self.state_manager.get_session_info()
        total_turns = session_info['total_turns']

        if total_turns > 0:
            # Timeline slider
            selected_turn = st.slider(
                "Timeline",
                min_value=0,
                max_value=total_turns - 1,
                value=session_info['current_turn'],
                key="timeline_scrubber",
                help="Drag to navigate to specific turn"
            )

            # Update turn if slider moved
            if selected_turn != session_info['current_turn']:
                self.state_manager.jump_to_turn(selected_turn)
                st.rerun()

            # Legend showing key events
            self._render_timeline_legend()

    def _render_timeline_legend(self):
        """Render legend showing event types on timeline."""
        legend_html = """
        <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 0.5rem; font-size: 0.75rem;">
            <div><span style="color: #90EE90;">●</span> Stable</div>
            <div><span style="color: #FFD700;">●</span> Watch</div>
            <div><span style="color: #FF6B6B;">●</span> Drift</div>
            <div><span style="color: #FF8C00;">⚡</span> Intervention</div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

    def _handle_autoplay(self):
        """Handle automatic playback if playing."""
        if not self.state_manager.is_playing():
            return

        # Calculate delay based on speed
        speed = self.state_manager.state.playback_speed
        delay = 2.0 / speed  # Base 2 second delay

        # Check if enough time has passed
        current_time = time.time()
        last_time = getattr(self.state_manager.state, 'last_autoplay_time', 0)

        if current_time - last_time >= delay:
            # Advance to next turn
            if self.state_manager.next_turn():
                self.state_manager.state.last_autoplay_time = current_time
                st.rerun()
            else:
                # Reached end, stop playback
                self.state_manager.stop_playback()
                st.rerun()
