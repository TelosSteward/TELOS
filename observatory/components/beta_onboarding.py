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
                'consent_statement': 'I understand and consent to participate in TELOS Beta testing. I understand that my full conversations will be stored temporarily for testing and development purposes, and will be permanently deleted after beta testing concludes. My data will not be shared externally or sold to third parties.',
                'version': '2.0'
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

        # Inject compact container CSS
        st.markdown("""
        <style>
        /* Compact centered container for beta consent */
        .beta-compact {
            max-width: 900px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Show onboarding screen
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #FFD700; font-size: 48px; margin: 0;">Welcome to TELOS Beta</h1>
            <p style="color: #e0e0e0; font-size: 18px; margin-top: 10px;">Help us build the future of AI governance</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        # Concise beta information in single box
        st.markdown("""
<div class="beta-compact">
<div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 2px solid #FFD700; border-radius: 8px; padding: 30px; margin: 20px auto;">
<h3 style="color: #FFD700; margin-top: 0; font-size: 24px;">Beta Testing & Data Privacy</h3>

<p style="color: #e0e0e0; font-size: 18px; line-height: 1.7; margin-bottom: 20px;">
<strong style="color: #FFD700;">During Beta:</strong> We store your full conversations to test and refine TELOS governance.
Your data helps us validate alignment tracking, improve algorithms, and build the delta-only infrastructure.
Sessions are anonymous—no login, tracking, or personal information required.
</p>

<p style="color: #e0e0e0; font-size: 18px; line-height: 1.7; margin-bottom: 20px;">
<strong style="color: #FFD700;">After Beta:</strong> All conversation data will be permanently deleted.
Full TELOS will use delta-only storage with Telemetric Keys—storing only mathematical measurements, not conversations.
</p>

<p style="color: #e0e0e0; font-size: 18px; line-height: 1.7; margin-bottom: 0;">
<strong style="color: #FFD700;">Data Use:</strong> Internal development only. Never shared externally, sold, or used to train other AI models.
</p>
</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

        # Consent checkbox inside Beta Consent box with proper alignment
        st.markdown("""
        <div class="beta-compact">
            <div class="message-container" style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); border: 2px solid #FFD700; border-radius: 8px; padding: 40px; margin: 20px auto; text-align: center;">
                <h1 style="color: #FFD700; margin: 0; font-size: 48px;">Beta Consent</h1>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Add CSS to style the checkbox container to match Beta Consent box width exactly
        st.markdown("""
        <style>
        /* Style the checkbox container to match Beta Consent box - force full width */
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stCheckbox"]) {
            max-width: 100% !important;
            width: 100% !important;
        }

        div[data-testid="stCheckbox"] {
            max-width: 900px !important;
            width: 900px !important;
            margin: 20px auto !important;
            padding: 25px !important;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
            border: 2px solid #FFD700 !important;
            border-radius: 8px !important;
            box-sizing: border-box !important;
        }

        /* Style the actual checkbox input - fill with gold when checked, no checkmark */
        div[data-testid="stCheckbox"] input[type="checkbox"]:checked {
            background-color: #FFD700 !important;
            border-color: #FFD700 !important;
            background-image: none !important;
        }

        /* Hide all pseudo-elements that might contain the checkmark */
        div[data-testid="stCheckbox"] input[type="checkbox"]:checked::before,
        div[data-testid="stCheckbox"] input[type="checkbox"]:checked::after {
            content: "" !important;
            display: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }

        /* Override Streamlit's SVG checkmark */
        div[data-testid="stCheckbox"] input[type="checkbox"]:checked {
            -webkit-appearance: none !important;
            -moz-appearance: none !important;
            appearance: none !important;
            width: 20px !important;
            height: 20px !important;
            background-color: #FFD700 !important;
            border: 2px solid #FFD700 !important;
            border-radius: 3px !important;
        }

        /* Make sure unchecked state is visible */
        div[data-testid="stCheckbox"] input[type="checkbox"] {
            -webkit-appearance: none !important;
            -moz-appearance: none !important;
            appearance: none !important;
            width: 20px !important;
            height: 20px !important;
            background-color: transparent !important;
            border: 2px solid #666 !important;
            border-radius: 3px !important;
        }

        @media (max-width: 950px) {
            div[data-testid="stCheckbox"] {
                width: calc(100vw - 50px) !important;
                max-width: 900px !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        consent = st.checkbox(
            "I understand and consent to participate in TELOS Beta testing. I understand that my full conversations will be stored temporarily for testing and development purposes, and will be permanently deleted after beta testing concludes. My data will not be shared externally or sold to third parties.",
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

                    # Initialize beta intro to start from slide 0
                    st.session_state.beta_intro_slide = 0
                    st.session_state.beta_intro_complete = False

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
