"""
Beta PA Establishment Component
================================

Expedited 1-2 turn Primacy Attractor establishment for BETA testing.

Flow:
1. User states goal/purpose (Turn 1)
2. Extract PA components using LLM
3. Show for confirmation/refinement (Turn 2)
4. Derive AI PA (hidden during session)
5. Save to session state and backend storage
"""

import streamlit as st
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from services.pa_extractor import PAExtractor
from services.backend_client import BackendService

logger = logging.getLogger(__name__)


class BetaPAEstablishment:
    """
    Expedited PA establishment for BETA sessions.

    Creates user and AI Primacy Attractors in 1-2 turns instead of 5-10.
    """

    def __init__(self, state_manager, backend_client: Optional[BackendService] = None):
        """
        Initialize PA establishment component.

        Args:
            state_manager: StateManager instance
            backend_client: Optional BackendService for logging
        """
        self.state_manager = state_manager
        self.backend = backend_client
        self.extractor = PAExtractor()

    def render(self) -> bool:
        """
        Render PA establishment flow.

        Returns:
            True if PA establishment complete, False if still in progress
        """
        # Check if PA already established
        if st.session_state.get('beta_pa_established', False):
            return True

        # Initialize PA establishment state
        if 'pa_extraction_step' not in st.session_state:
            st.session_state.pa_extraction_step = 'statement'
            st.session_state.extracted_pa = None
            st.session_state.ai_pa = None

        # Render current step
        if st.session_state.pa_extraction_step == 'statement':
            return self._render_statement_step()
        elif st.session_state.pa_extraction_step == 'confirmation':
            return self._render_confirmation_step()
        else:
            # Fallback - should not reach here
            return False

    def _render_statement_step(self) -> bool:
        """
        Step 1: User states their goal/purpose.

        Returns:
            False (PA not established yet)
        """
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #F4D03F; font-size: 36px; margin: 0;">Establish Your Purpose</h2>
            <p style="color: #e0e0e0; font-size: 16px; margin-top: 10px;">
                Tell TELOS what you want to accomplish in this session
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        # Show examples
        with st.expander("💡 Examples of good purpose statements", expanded=False):
            st.markdown("""
            **Work & Development:**
            - "I want to debug my Python API authentication issue"
            - "Help me refactor this messy codebase for better maintainability"
            - "I need to design a database schema for a social media app"

            **Learning & Research:**
            - "I want to understand TELOS without overwhelming technical details"
            - "Explain quantum computing concepts at an undergraduate level"
            - "Help me write a literature review on AI alignment research"

            **Creative & Planning:**
            - "Help me brainstorm ideas for a sci-fi short story"
            - "I need to plan a 2-week trip to Japan on a budget"
            - "Assist with writing a grant proposal for climate research"

            **Tips:**
            - Be specific about your goal
            - Mention your expertise level if relevant
            - Note any constraints (time, complexity, scope)
            """)

        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        # Input form
        with st.form(key="pa_statement_form"):
            st.markdown("**What do you want to accomplish?**")

            user_statement = st.text_area(
                label="Your purpose",
                placeholder="Example: I want to debug my Python API authentication issue without getting overwhelmed by security theory...",
                height=100,
                label_visibility="collapsed",
                key="pa_statement_input"
            )

            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit_button = st.form_submit_button(
                    "Extract Purpose",
                    use_container_width=True,
                    type="primary"
                )

        # Process submission
        if submit_button:
            if not user_statement or len(user_statement.strip()) < 10:
                st.error("Please provide a more detailed purpose statement (at least 10 characters)")
                return False

            # Extract PA components
            with st.spinner("🔍 Analyzing your purpose..."):
                try:
                    extracted_pa = self.extractor.extract_from_statement(user_statement)

                    # Store in session state
                    st.session_state.extracted_pa = extracted_pa
                    st.session_state.pa_extraction_step = 'confirmation'

                    # Move to confirmation step
                    st.rerun()

                except Exception as e:
                    logger.error(f"PA extraction failed: {e}")
                    st.error(f"Failed to extract purpose: {e}")
                    return False

        return False  # Not established yet

    def _render_confirmation_step(self) -> bool:
        """
        Step 2: Show extracted PA for confirmation/refinement.

        Returns:
            True if confirmed and saved, False otherwise
        """
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #F4D03F; font-size: 36px; margin: 0;">Confirm Your Purpose</h2>
            <p style="color: #e0e0e0; font-size: 16px; margin-top: 10px;">
                TELOS extracted these components from your statement
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        extracted_pa = st.session_state.extracted_pa

        # Show extracted PA
        st.markdown("""
        <div class="message-container" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
             border: 1px solid #F4D03F; border-radius: 8px; padding: 20px; margin: 20px 0;">
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Purpose")
        st.markdown(f"**{extracted_pa['purpose'][0]}**")

        st.markdown("### 🔭 Scope")
        for item in extracted_pa['scope']:
            st.markdown(f"- {item}")

        st.markdown("### 🚧 Boundaries")
        for item in extracted_pa['boundaries']:
            st.markdown(f"- {item}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        # Confirmation options
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✓ Looks Good - Start Session", use_container_width=True, type="primary"):
                return self._finalize_pa_establishment(extracted_pa)

        with col2:
            refine_mode = st.button("✎ Refine Purpose", use_container_width=True)

        # Refinement interface
        if refine_mode or st.session_state.get('pa_refine_mode', False):
            st.session_state.pa_refine_mode = True

            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

            with st.form(key="pa_refinement_form"):
                st.markdown("**How would you like to refine your purpose?**")

                refinement_request = st.text_area(
                    label="Refinement",
                    placeholder="Example: Make the scope more focused on just authentication, not general API issues...",
                    height=80,
                    label_visibility="collapsed",
                    key="pa_refinement_input"
                )

                col1, col2 = st.columns(2)
                with col1:
                    refine_submit = st.form_submit_button("Apply Refinement", use_container_width=True)
                with col2:
                    cancel_refine = st.form_submit_button("Cancel", use_container_width=True)

            if refine_submit and refinement_request:
                with st.spinner("🔄 Refining your purpose..."):
                    try:
                        refined_pa = self.extractor.refine_pa(extracted_pa, refinement_request)
                        st.session_state.extracted_pa = refined_pa
                        st.session_state.pa_refine_mode = False
                        st.rerun()
                    except Exception as e:
                        logger.error(f"PA refinement failed: {e}")
                        st.error(f"Refinement failed: {e}")

            if cancel_refine:
                st.session_state.pa_refine_mode = False
                st.rerun()

        return False  # Not finalized yet

    def _finalize_pa_establishment(self, user_pa: Dict[str, Any]) -> bool:
        """
        Finalize PA establishment: derive AI PA and save everything.

        Args:
            user_pa: Confirmed user PA

        Returns:
            True if successful
        """
        try:
            with st.spinner("⚙️ Setting up TELOS governance..."):
                # Derive AI PA
                ai_pa = self.extractor.derive_ai_pa(user_pa)

                # Store in session state
                st.session_state.user_pa = user_pa
                st.session_state.ai_pa = ai_pa
                st.session_state.beta_pa_established = True

                # Create BETA session in backend storage
                if self.backend:
                    try:
                        session_id = st.session_state.get('session_id', self.state_manager.state.session_id)

                        session_data = {
                            'session_id': session_id,
                            'user_pa_config': user_pa,
                            'ai_pa_config': ai_pa,
                            'basin_constant': 1.0,  # Proven effective
                            'constraint_tolerance': 0.05,  # Strict governance
                            'created_at': datetime.now().isoformat(),
                            'total_turns': 0
                        }

                        self.backend.insert_beta_session(session_data)
                        logger.info(f"Created BETA session: {session_id}")

                    except Exception as e:
                        logger.error(f"Failed to create backend session: {e}")
                        # Don't block - session can work without backend logging

                # Add to state manager metadata
                if hasattr(self.state_manager.state, 'metadata'):
                    self.state_manager.state.metadata['user_pa'] = user_pa
                    self.state_manager.state.metadata['ai_pa'] = ai_pa
                    self.state_manager.state.metadata['pa_established_at'] = datetime.now().isoformat()
                else:
                    self.state_manager.state.metadata = {
                        'user_pa': user_pa,
                        'ai_pa': ai_pa,
                        'pa_established_at': datetime.now().isoformat()
                    }

                logger.info("PA establishment complete")

                # Show success message
                st.success("✓ Purpose established! Starting your BETA session...")
                st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

                # Small delay to show success message
                import time
                time.sleep(1)

                st.rerun()

                return True

        except Exception as e:
            logger.error(f"PA finalization failed: {e}")
            st.error(f"Failed to finalize purpose: {e}")
            return False

    def get_user_pa(self) -> Optional[Dict[str, Any]]:
        """
        Get established user PA.

        Returns:
            User PA dictionary or None if not established
        """
        return st.session_state.get('user_pa')

    def get_ai_pa(self) -> Optional[Dict[str, Any]]:
        """
        Get derived AI PA.

        Returns:
            AI PA dictionary or None if not established
        """
        return st.session_state.get('ai_pa')

    def is_established(self) -> bool:
        """
        Check if PA establishment is complete.

        Returns:
            True if PA established
        """
        return st.session_state.get('beta_pa_established', False)
