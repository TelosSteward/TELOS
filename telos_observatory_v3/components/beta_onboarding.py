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
                'consent_statement': 'I understand and consent to participate in TELOS Beta testing. I agree that governance deltas (mathematical measurements) from my ephemeral sessions will be transmitted to the research database. I understand my conversations are never sent to servers and exist only during my active browser session.',
                'version': '3.0'  # Ephemeral session architecture with delta-only transmission
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

        # Centered container wrapper - constrains width and centers content
        st.markdown("""
        <style>
        .beta-main-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Use columns for centering
        col1, col2, col3 = st.columns([1, 6, 1])

        with col2:
            # Title
            st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="color: #FFD700; font-size: 48px; margin: 0;">Welcome to TELOS Beta</h1>
                <p style="color: #B0B0B0; font-size: 18px; margin-top: 10px;">Help us build the future of AI governance</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

            # What is Beta?
            st.markdown('<div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;"><h3 style="color: #FFD700; margin-top: 0;">What is Beta Testing?</h3><p style="color: #B0B0B0; font-size: 18px; line-height: 1.6;">You\'re getting early access to TELOS Observatory features before they\'re publicly released. Your feedback helps us refine the system and improve AI governance for everyone.</p></div>', unsafe_allow_html=True)

            # How We Handle Your Data - Honest and Consolidated
            st.markdown('<div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;"><h3 style="color: #FFD700; margin-top: 0;">How We Handle Your Data</h3><p style="color: #B0B0B0; font-size: 18px; line-height: 1.6;"><strong style="color: #FFD700;">Ephemeral Sessions:</strong> Your conversation exists only during your active session in your browser. When you close the browser or end your session, the conversation is gone. We cannot retrieve past conversations.</p><p style="color: #B0B0B0; font-size: 18px; line-height: 1.6;"><strong style="color: #FFD700;">What We Collect:</strong> As you converse, governance deltas (mathematical measurements like fidelity scores and intervention patterns) are transmitted to our research database. Your actual conversations are never sent to our servers. Only the numerical measurements are collected.</p><p style="color: #B0B0B0; font-size: 18px; line-height: 1.6;"><strong style="color: #FFD700;">Best Practice:</strong> Complete sessions of 15-20 turns provide the most useful governance data for improving TELOS.</p></div>', unsafe_allow_html=True)

            # What We Use Deltas For
            st.markdown('<div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;"><h3 style="color: #FFD700; margin-top: 0;">What We Use Deltas For</h3><p style="color: #B0B0B0; font-size: 18px; line-height: 1.6;">Your governance deltas help us:</p><ul style="color: #B0B0B0; font-size: 18px; line-height: 1.6; padding-left: 30px;"><li><strong>Improve alignment quality</strong> by learning better correction patterns</li><li><strong>Refine governance standards</strong> to understand what keeps AI systems on track</li><li><strong>Advance research</strong> to contribute to safer, more controllable AI</li></ul><p style="color: #B0B0B0; font-size: 18px; line-height: 1.6;">All deltas are anonymized and aggregated. Your individual session cannot be identified or singled out from the collective dataset.</p></div>', unsafe_allow_html=True)

            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

            # Consent section
            st.markdown('<div class="message-container" style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); border: 2px solid #FFD700; border-radius: 8px; padding: 25px; margin: 20px 0;"><h3 style="color: #FFD700; margin-top: 0; text-align: center;">Beta Consent</h3></div>', unsafe_allow_html=True)

            # Custom CSS for visible yellow checkbox
            st.markdown("""
                <style>
                /* Make checkbox container more visible */
                div[data-testid="stCheckbox"] {
                    background: rgba(255, 215, 0, 0.1) !important;
                    border: 2px solid #FFD700 !important;
                    border-radius: 8px !important;
                    padding: 20px !important;
                    margin: 15px 0 !important;
                }

                /* Style the checkbox itself - larger and more visible */
                div[data-testid="stCheckbox"] input[type="checkbox"] {
                    width: 24px !important;
                    height: 24px !important;
                    border: 3px solid #FFD700 !important;
                    border-radius: 4px !important;
                    cursor: pointer !important;
                    background-color: #1a1a1a !important;
                }

                /* Yellow background when checked */
                div[data-testid="stCheckbox"] input[type="checkbox"]:checked {
                    background-color: #FFD700 !important;
                    border-color: #FFD700 !important;
                }

                /* Checkmark styling */
                div[data-testid="stCheckbox"] input[type="checkbox"]:checked::after {
                    content: "✓" !important;
                    color: #000000 !important;
                    font-size: 18px !important;
                    font-weight: bold !important;
                }

                /* Make label text more visible */
                div[data-testid="stCheckbox"] label {
                    color: #FAFAFA !important;
                    font-size: 18px !important;
                    font-weight: 500 !important;
                    line-height: 1.6 !important;
                    padding-left: 10px !important;
                }

                /* Hover effect */
                div[data-testid="stCheckbox"]:hover {
                    background: rgba(255, 215, 0, 0.15) !important;
                    border-color: #FFC700 !important;
                }
                </style>
            """, unsafe_allow_html=True)

            # Consent checkbox
            consent = st.checkbox(
                "I understand and consent to participate in TELOS Beta testing. I agree that governance deltas (mathematical measurements) from my ephemeral sessions will be transmitted to the research database. I understand my conversations are never sent to servers and exist only during my active browser session.",
                key="beta_consent_checkbox"
            )

            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

            # Continue button (centered within the column)
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col2:
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
            st.markdown('<div style="text-align: center; color: #888; font-size: 14px; margin-top: 30px; margin-bottom: 30px;">Questions about data privacy? Contact us at research@teloslabs.com</div>', unsafe_allow_html=True)

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
