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

        # Kill Streamlit's default top padding and header gap
        st.markdown("""
        <style>
            .block-container,
            section.main > div.block-container,
            [data-testid="stMainBlockContainer"],
            .stMainBlockContainer.block-container,
            div[data-testid="stMainBlockContainer"].block-container,
            .stApp .main .block-container,
            .stApp section.main .block-container {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
            [data-testid="stAppViewContainer"] > .main {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
            section.main {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }
        </style>
        """, unsafe_allow_html=True)

        # Center all content using standard column layout
        col_spacer_left, col_center, col_spacer_right = st.columns([0.5, 3, 0.5])

        with col_center:
            # Show onboarding screen
            st.markdown("""
            <div style="text-align: center; padding: 0 0 4px 0; margin-top: -6rem;">
                <h1 style="color: #F4D03F; font-size: 48px; margin: 0;">Welcome to TELOS Beta</h1>
                <p style="color: #e0e0e0; font-size: 18px; margin-top: 4px; margin-bottom: 0;">Help us build the future of AI governance</p>
            </div>

            <div class="message-container" style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid #F4D03F; border-radius: 8px; padding: 16px 20px; margin: 8px 0 4px 0; box-shadow: 0 0 15px rgba(244, 208, 63, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);">
                <h3 style="color: #F4D03F; margin-top: 0; margin-bottom: 6px;">What is Beta Testing?</h3>
                <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6; margin-bottom: 0;">
                    You're getting early access to TELOS AI Governance features before they're publicly released.
                    Your feedback helps us refine the system and improve AI governance for everyone.
                </p>
            </div>

            <div class="message-container" style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid #F4D03F; border-radius: 8px; padding: 16px 20px; margin: 4px 0 8px 0; box-shadow: 0 0 15px rgba(244, 208, 63, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);">
                <h3 style="color: #F4D03F; margin-top: 0; margin-bottom: 6px;">Your Data</h3>
                <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6; margin-bottom: 0;">
                    Conversations stay in your browser and are never sent to our servers. When you close the session, they're gone. We collect only anonymized governance measurements (fidelity scores, intervention patterns) — never conversation content.
                </p>
            </div>
            """, unsafe_allow_html=True)

            consent = st.checkbox(
                "I understand and consent to participate in TELOS Beta testing. I agree to share governance deltas (mathematical measurements only) to help improve the system. I understand my conversation content remains private.",
                key="beta_consent_checkbox"
            )

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

            # Footer note
            st.markdown("""
            <div style="text-align: center; color: #888; font-size: 14px; margin-top: 4px;">
                Questions about data privacy? Contact us at <a href="mailto:JB@telos-labs.ai" style="color: #F4D03F; text-decoration: none;">JB@telos-labs.ai</a>
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
