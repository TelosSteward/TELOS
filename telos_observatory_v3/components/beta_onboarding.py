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

        # Show onboarding screen
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #FFD700; font-size: 48px; margin: 0;">Welcome to TELOS Beta</h1>
            <p style="color: #e0e0e0; font-size: 18px; margin-top: 10px;">Help us build the future of AI governance</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        # What is Beta?
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #FFD700; margin-top: 0;">What is Beta Testing?</h3>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                You're getting early access to TELOS Observatory features before they're publicly released.
                Your feedback helps us refine the system and improve AI governance for everyone.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # How We Handle Your Data
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #FFD700; margin-top: 0;">How We Handle Your Data</h3>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                <strong style="color: #FFD700;">We only collect governance deltas — not your conversations.</strong>
            </p>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                Deltas are mathematical measurements of how the AI stays aligned with its purpose:
            </p>
            <ul style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                <li><strong>Fidelity scores</strong> — how well responses match intended purpose</li>
                <li><strong>Distance measurements</strong> — mathematical drift from target behavior</li>
                <li><strong>Intervention patterns</strong> — when and how corrections were applied</li>
            </ul>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                <strong style="color: #FFD700;">Your actual conversation content stays with you.</strong>
                We never see what you talk about — only the governance telemetry.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Privacy Protection
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #FFD700; margin-top: 0;">Privacy Protection</h3>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                Your session data is protected by <strong style="color: #FFD700;">Telemetric Keys</strong> —
                encryption that evolves with each turn of conversation.
            </p>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                Key properties:
            </p>
            <ul style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                <li><strong>Session-bound:</strong> Keys exist only during your session, then disappear</li>
                <li><strong>Non-reproducible:</strong> Cannot be recreated without exact session telemetry</li>
                <li><strong>Mathematically secure:</strong> Based on physical randomness from your interaction patterns</li>
            </ul>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                Even if someone accessed the stored deltas, they couldn't reconstruct your conversations —
                only the mathematical governance measurements.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # What We Use Deltas For
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #FFD700; margin-top: 0;">What We Use Deltas For</h3>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                Your governance deltas help us:
            </p>
            <ul style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                <li><strong>Improve alignment quality</strong> — learn better correction patterns</li>
                <li><strong>Refine governance standards</strong> — understand what keeps AI systems on track</li>
                <li><strong>Advance research</strong> — contribute to safer, more controllable AI</li>
            </ul>
            <p style="color: #e0e0e0; font-size: 19px; line-height: 1.6;">
                All deltas are anonymized and aggregated. Your individual session cannot be identified
                or singled out from the collective dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

        # Consent checkbox
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); border: 2px solid #FFD700; border-radius: 8px; padding: 25px; margin: 20px 0;">
            <h3 style="color: #FFD700; margin-top: 0; text-align: center;">Beta Consent</h3>
        </div>
        """, unsafe_allow_html=True)

        consent = st.checkbox(
            "I understand and consent to participate in TELOS Beta testing. I agree to share governance deltas (mathematical measurements only) to help improve the system. I understand my conversation content remains private.",
            key="beta_consent_checkbox"
        )

        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

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

        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        # Footer note
        st.markdown("""
        <div style="text-align: center; color: #888; font-size: 14px; margin-top: 30px;">
            Questions about data privacy? Contact us at research@teloslabs.com
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
