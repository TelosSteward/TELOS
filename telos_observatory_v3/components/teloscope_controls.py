"""
TELOSCOPE Controls Component for TELOS Observatory V3.
Playback and analysis controls (simplified for V3 - no draggability yet).
"""

import streamlit as st


class TELOSCOPEControls:
    """TELOSCOPE Controls panel."""

    def __init__(self, state_manager):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for playback control
        """
        self.state_manager = state_manager

    def render(self):
        """Render TELOSCOPE controls panel."""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        ">
            <div style="text-align: center; margin-bottom: 15px;">
                <h3 style="color: #FFD700; margin: 0;">
                    🔭 TELOSCOPE Controls
                </h3>
                <p style="color: #888; font-size: 11px; margin: 5px 0 0 0;">
                    Session playback and analysis
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Playback controls
        col1, col2, col3 = st.columns(3)

        with col1:
            play_label = "⏸ Pause" if self.state_manager.is_playing() else "▶ Play"
            if st.button(play_label, use_container_width=True, key="teloscope_play"):
                if self.state_manager.is_playing():
                    self.state_manager.stop_playback()
                else:
                    self.state_manager.start_playback()
                st.rerun()

        with col2:
            if st.button("⏮ Reset", use_container_width=True, key="teloscope_reset"):
                self.state_manager.jump_to_turn(0)
                st.rerun()

        with col3:
            if st.button("⏭ Skip", use_container_width=True, key="teloscope_skip"):
                # Skip ahead 5 turns
                target = min(
                    self.state_manager.state.current_turn + 5,
                    self.state_manager.state.total_turns - 1
                )
                self.state_manager.jump_to_turn(target)
                st.rerun()

        # Playback speed
        st.markdown("---")
        st.markdown("**Playback Speed**")
        speed = st.select_slider(
            "Speed",
            options=[0.5, 1.0, 1.5, 2.0, 3.0],
            value=self.state_manager.state.playback_speed,
            key="teloscope_speed",
            label_visibility="collapsed"
        )
        if speed != self.state_manager.state.playback_speed:
            self.state_manager.set_playback_speed(speed)

        # Auto-advance toggle
        st.markdown("---")
        auto_advance = st.checkbox(
            "Auto-advance on intervention",
            value=False,
            key="teloscope_auto",
            help="Automatically skip to next intervention"
        )

        if auto_advance:
            st.info("Auto-advance: Will jump to next intervention event")
