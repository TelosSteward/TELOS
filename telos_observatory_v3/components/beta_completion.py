"""
Beta Completion Component for TELOS Observatory BETA
=====================================================

Displays community onboarding options after beta testing completion.
Offers Discord/Telegram access to full implementation.
"""

import streamlit as st
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BetaCompletion:
    """Handles beta completion flow and community onboarding."""

    def __init__(self, state_manager):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for session operations
        """
        self.state_manager = state_manager

    def render(self):
        """Render beta completion congratulations and community invitation."""

        # Main congratulations header
        st.markdown("""
        <div style="text-align: center; padding: 40px 0;">
            <h1 style="color: #FFD700; font-size: 48px; margin: 0;">
                🎉 Congratulations! 🎉
            </h1>
            <p style="color: #e0e0e0; font-size: 24px; margin-top: 20px;">
                You've completed the TELOS Beta Testing
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Thank you message
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                    border: 2px solid #FFD700; border-radius: 12px; padding: 30px; margin: 20px 0;">
            <h2 style="color: #FFD700; text-align: center;">Thank You for Your Contribution!</h2>
            <p style="color: #e0e0e0; font-size: 18px; line-height: 1.6; text-align: center;">
                Your feedback and governance deltas are invaluable in helping us build
                safer, more aligned AI systems. You've directly contributed to advancing
                the field of AI governance research.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Stats summary if available
        if hasattr(self.state_manager.state, 'beta_session_manager') and \
           self.state_manager.state.beta_session_manager:
            try:
                session = self.state_manager.state.beta_session_manager.get_current_session()
                if session:
                    st.markdown("""
                    <div class="message-container" style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
                                border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h3 style="color: #FFD700;">Your Beta Testing Stats</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Turns", session.turn_count)
                    with col2:
                        st.metric("Feedback Given", len(session.feedback_data))
                    with col3:
                        st.metric("Test Mode", session.test_condition or "Complete")
            except Exception as e:
                logger.error(f"Error displaying beta stats: {e}")

        # Community invitation
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                    border: 2px solid #FFD700; border-radius: 12px; padding: 30px; margin: 30px 0;">
            <h2 style="color: #FFD700; text-align: center;">
                🚀 Ready for the Full TELOS Experience?
            </h2>
            <p style="color: #e0e0e0; font-size: 18px; line-height: 1.6; text-align: center; margin: 20px 0;">
                As a beta tester, you've earned exclusive early access to the complete
                TELOS implementation with advanced features and capabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Community options
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a3a1a 0%, #2d4d2d 100%);
                        border: 1px solid #4CAF50; border-radius: 8px; padding: 20px; height: 200px;">
                <h3 style="color: #4CAF50; text-align: center;">💬 Join Discord</h3>
                <p style="color: #e0e0e0; text-align: center;">
                    Connect with researchers, developers, and other beta testers.
                    Get early access to new features and research updates.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔗 Join TELOS Discord", use_container_width=True, key="discord_invite"):
                st.markdown("""
                <script>
                window.open('https://discord.gg/telos-research', '_blank');
                </script>
                """, unsafe_allow_html=True)
                st.success("Opening Discord invite...")
                # Log the community join
                self._log_community_join("discord")

        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1a2a3a 0%, #2d3d4d 100%);
                        border: 1px solid #2196F3; border-radius: 8px; padding: 20px; height: 200px;">
                <h3 style="color: #2196F3; text-align: center;">📱 Join Telegram</h3>
                <p style="color: #e0e0e0; text-align: center;">
                    Stay updated with announcements, participate in discussions,
                    and access exclusive beta tester channels.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔗 Join TELOS Telegram", use_container_width=True, key="telegram_invite"):
                st.markdown("""
                <script>
                window.open('https://t.me/telos_labs', '_blank');
                </script>
                """, unsafe_allow_html=True)
                st.success("Opening Telegram invite...")
                # Log the community join
                self._log_community_join("telegram")

        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        # Feedback form option
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
                    border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #FFD700; text-align: center;">📝 Optional: Share Detailed Feedback</h3>
            <p style="color: #e0e0e0; text-align: center;">
                Have thoughts about your beta testing experience?
                We'd love to hear your detailed feedback!
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📋 Open Feedback Form", use_container_width=True, key="feedback_form"):
                st.markdown("""
                <script>
                window.open('https://forms.gle/telos-beta-feedback', '_blank');
                </script>
                """, unsafe_allow_html=True)
                st.info("Opening feedback form...")

        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

        # Research repository link
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-top: 1px solid #333; margin-top: 40px;">
            <p style="color: #888;">
                Learn more about the TELOS research project:
            </p>
            <p>
                <a href="https://github.com/TelosSteward/TelosLabs" target="_blank"
                   style="color: #FFD700; text-decoration: none;">
                    🔬 GitHub: TelosSteward/TelosLabs
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Option to restart beta testing
        st.markdown("<div style='margin: 60px 0;'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Start New Beta Session", use_container_width=True, key="restart_beta"):
                # Reset beta completion status
                st.session_state.beta_completed = False
                st.session_state.active_tab = "DEMO"
                # Clear beta session data
                if hasattr(self.state_manager.state, 'beta_session_manager'):
                    self.state_manager.state.beta_session_manager = None
                st.rerun()

    def _log_community_join(self, platform: str):
        """Log when user joins community platform.

        Args:
            platform: Platform joined (discord/telegram)
        """
        try:
            if hasattr(self.state_manager.state, 'metadata'):
                if 'community_joins' not in self.state_manager.state.metadata:
                    self.state_manager.state.metadata['community_joins'] = []

                self.state_manager.state.metadata['community_joins'].append({
                    'platform': platform,
                    'timestamp': datetime.now().isoformat()
                })

                logger.info(f"User joined {platform} community")
        except Exception as e:
            logger.error(f"Error logging community join: {e}")