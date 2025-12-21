"""
Beta Onboarding Component for TELOS Observatory V3.
Handles consent and data privacy explanation for beta testers.
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
import json


class BetaOnboarding:
    """Beta onboarding screen with consent and data privacy information."""

    def __init__(self, state_manager):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for session operations
        """
        self.state_manager = state_manager

    def _log_consent(self, session_id: str):
        """Log consent to persistent file for audit trail.

        Args:
            session_id: Session identifier for the consent
        """
        try:
            # Create beta_consents directory if it doesn't exist
            consents_dir = Path(__file__).parent.parent / 'beta_consents'
            consents_dir.mkdir(exist_ok=True)

            # Consent log file
            consent_log_file = consents_dir / 'consent_log.json'

            # Load existing consents
            if consent_log_file.exists():
                with open(consent_log_file, 'r') as f:
                    consent_log = json.load(f)
            else:
                consent_log = {'consents': []}

            # Add new consent
            consent_entry = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'consent_statement': 'I understand and consent to participate in TELOS Beta testing. I agree to share governance deltas (mathematical measurements only) to help improve the system. I understand my conversation content remains private.',
                'version': '1.0'
            }

            consent_log['consents'].append(consent_entry)

            # Save updated log
            with open(consent_log_file, 'w') as f:
                json.dump(consent_log, f, indent=2)

        except Exception as e:
            st.error(f"Error logging consent: {e}")

    def render(self):
        """Render beta onboarding screen - blocks all access until consent given."""
        # Check if user has already consented
        if st.session_state.get('beta_consent_given', False):
            return True  # User has consented, proceed to beta interface

        # Center all content using standard column layout
        col_spacer_left, col_center, col_spacer_right = st.columns([0.5, 3, 0.5])

        with col_center:
            # Show onboarding screen
            st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="color: #F4D03F; font-size: 48px; margin: 0;">Welcome to TELOS Beta</h1>
                <p style="color: #e0e0e0; font-size: 18px; margin-top: 10px;">Help us build the future of AI governance</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

            # What is Beta?
            st.markdown("""
            <div class="message-container" style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid #F4D03F; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 0 15px rgba(244, 208, 63, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);">
                <h3 style="color: #F4D03F; margin-top: 0;">What is Beta Testing?</h3>
                <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                    You're getting early access to TELOS AI Governance features before they're publicly released.
                    Your feedback helps us refine the system and improve AI governance for everyone.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # How We Handle Your Data
            st.markdown("""
            <div class="message-container" style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid #F4D03F; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 0 15px rgba(244, 208, 63, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);">
                <h3 style="color: #F4D03F; margin-top: 0;">How We Handle Your Data</h3>
                <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                    <strong style="color: #F4D03F;">Ephemeral Sessions:</strong> Your conversation exists only during your active session in your browser using Railway's servers. When you close the browser or end your session, the conversation is gone. We cannot retrieve past conversations.
                </p>
                <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                    <strong style="color: #F4D03F;">What We Collect:</strong> As you converse, governance deltas (mathematical measurements like fidelity scores and intervention patterns) are transmitted to our research database. Your actual conversations are never sent to our servers. Only the numerical measurements are collected. All deltas are anonymized and aggregated. Your individual session cannot be identified or singled out from the collective dataset.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

            # Consent checkbox
            st.markdown("""
            <div class="message-container" style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 2px solid #F4D03F; border-radius: 8px; padding: 25px; margin: 10px 0; box-shadow: 0 0 20px rgba(244, 208, 63, 0.2), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);">
                <h3 style="color: #F4D03F; margin-top: 0; text-align: center;">Beta Consent</h3>
            </div>
            """, unsafe_allow_html=True)

            consent = st.checkbox(
                "I understand and consent to participate in TELOS Beta testing. I agree to share governance deltas (mathematical measurements only) to help improve the system. I understand my conversation content remains private.",
                key="beta_consent_checkbox"
            )

            st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)

            # Continue button (only enabled if consent given)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue to Beta", use_container_width=True, disabled=not consent):
                    if consent:
                        # Record consent in session state
                        st.session_state.beta_consent_given = True
                        st.session_state.beta_consent_timestamp = datetime.now().isoformat()

                        # Get session ID from state manager
                        session_id = self.state_manager.state.session_id

                        # Log consent to persistent file
                        self._log_consent(session_id)

                        # Log consent to session metadata (non-sensitive)
                        if hasattr(self.state_manager.state, 'metadata'):
                            self.state_manager.state.metadata['beta_consent'] = True
                            self.state_manager.state.metadata['beta_consent_timestamp'] = st.session_state.beta_consent_timestamp
                        else:
                            # Create metadata dict if it doesn't exist
                            self.state_manager.state.metadata = {
                                'beta_consent': True,
                                'beta_consent_timestamp': st.session_state.beta_consent_timestamp
                            }

                        st.rerun()

            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

            # Footer note
            st.markdown("""
            <div style="text-align: center; color: #888; font-size: 14px; margin-top: 15px;">
                Questions about data privacy? Contact us at contact@telos-labs.ai
            </div>
            """, unsafe_allow_html=True)

        return False  # User has not consented yet

    def has_consent(self):
        """Check if user has given beta consent.

        Returns:
            bool: True if consent given, False otherwise
        """
        return st.session_state.get('beta_consent_given', False)

    def revoke_consent(self):
        """Revoke beta consent and clear associated data."""
        if 'beta_consent_given' in st.session_state:
            del st.session_state.beta_consent_given
        if 'beta_consent_timestamp' in st.session_state:
            del st.session_state.beta_consent_timestamp

        # Clear from state manager metadata if exists
        if hasattr(self.state_manager.state, 'metadata'):
            if 'beta_consent' in self.state_manager.state.metadata:
                del self.state_manager.state.metadata['beta_consent']
            if 'beta_consent_timestamp' in self.state_manager.state.metadata:
                del self.state_manager.state.metadata['beta_consent_timestamp']
