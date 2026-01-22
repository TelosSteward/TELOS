"""
TELOS SQL Agent - Governance Demo
=================================

Demonstrates TELOS governance in action with a real SQL agent.
The UI is designed to showcase the value proposition of purpose-bound AI.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_sql_agent.agent import (
    SQLAgent, SQLAgentConfig, SupabaseConfig, GovernanceDecision,
    ToolPlanResult, GovernedToolCall,
)

# =============================================================================
# TELOS Design System Colors
# =============================================================================

BG_BASE = "#0a0a0c"
BG_SURFACE = "#1a1a1e"
BG_ELEVATED = "#2d2d34"
BG_OVERLAY = "#3d3d44"
GOLD = "#f4d03f"
GOLD_DIM = "rgba(244, 208, 63, 0.3)"
TEXT_PRIMARY = "#e8e8eb"
TEXT_SECONDARY = "#9ca3af"
GREEN = "#27ae60"
YELLOW = "#f39c12"
ORANGE = "#e67e22"
RED = "#e74c3c"

# =============================================================================
# Global CSS - TELOS Glassmorphism Design System
# =============================================================================

GLOBAL_CSS = f"""
<style>
/* Base page styling */
.stApp {{
    background:
        radial-gradient(ellipse 120% 80% at 50% 0%, rgba(244, 208, 63, 0.12), transparent 50%),
        radial-gradient(ellipse 100% 60% at 20% 30%, rgba(244, 208, 63, 0.08), transparent 45%),
        {BG_BASE};
    color: {TEXT_PRIMARY};
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

.main .block-container {{
    max-width: 1100px !important;
    padding: 20px 30px !important;
}}

h1, h2, h3, h4, h5 {{
    color: {GOLD} !important;
}}

.stButton > button {{
    background-color: {BG_ELEVATED} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {GOLD} !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}}

.stButton > button:hover {{
    background-color: {BG_OVERLAY} !important;
    box-shadow: 0 0 8px rgba(244, 208, 63, 0.5) !important;
}}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background-color: {BG_ELEVATED} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {GOLD} !important;
    border-radius: 6px !important;
}}

.stTextInput label, .stTextArea label {{
    color: {TEXT_SECONDARY} !important;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(26, 26, 30, 0.95), rgba(10, 10, 12, 0.98)) !important;
    border-right: 1px solid rgba(244, 208, 63, 0.2) !important;
}}

hr {{
    border-color: rgba(244, 208, 63, 0.2) !important;
}}

/* Governance panel styling */
.governance-panel {{
    background: linear-gradient(135deg, rgba(45, 45, 52, 0.9), rgba(26, 26, 30, 0.95));
    border: 2px solid {GOLD};
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 0 20px rgba(244, 208, 63, 0.2);
}}

/* Fidelity meter */
.fidelity-bar {{
    height: 12px;
    border-radius: 6px;
    background: {BG_ELEVATED};
    overflow: hidden;
    margin: 10px 0;
}}

.fidelity-fill {{
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease, background-color 0.3s ease;
}}
</style>
"""

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="TELOS SQL Agent - Governance Demo",
    page_icon="",
    layout="wide",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# =============================================================================
# Session State
# =============================================================================

if "agent" not in st.session_state:
    st.session_state.agent = None
if "connected" not in st.session_state:
    st.session_state.connected = False
if "governance_enabled" not in st.session_state:
    st.session_state.governance_enabled = True
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_tool_plan" not in st.session_state:
    st.session_state.last_tool_plan = None


# =============================================================================
# Helper Functions
# =============================================================================

def get_fidelity_color(fidelity: float) -> str:
    """Get color based on fidelity score.

    Strict thresholds for purpose-bound governance:
    - 90%+ = EXECUTE (green) - High confidence, proceed
    - 70-89% = CLARIFY (yellow) - Moderate, ask for context
    - 50-69% = SUGGEST (orange) - Low, redirect to purpose
    - <50% = INERT (red) - Off-purpose, block
    """
    if fidelity >= 0.90:
        return GREEN
    elif fidelity >= 0.70:
        return YELLOW
    elif fidelity >= 0.50:
        return ORANGE
    else:
        return RED


def get_decision_explanation(decision: str, fidelity: float) -> str:
    """Get human-readable explanation of governance decision."""
    explanations = {
        "execute": f"This query aligns well with the SQL Agent's purpose ({fidelity:.0%} fidelity). The agent will proceed with the request.",
        "clarify": f"This query has moderate alignment ({fidelity:.0%} fidelity). The agent may ask for clarification to better serve your database needs.",
        "suggest": f"This query has low alignment ({fidelity:.0%} fidelity). The agent will suggest how to reformulate the request for database operations.",
        "inert": f"This query doesn't align with the SQL Agent's purpose ({fidelity:.0%} fidelity). The request has been blocked to maintain purpose integrity.",
        "escalate": f"This query requires human review ({fidelity:.0%} fidelity). The request is being escalated for approval.",
    }
    return explanations.get(decision.lower(), f"Unknown decision ({fidelity:.0%} fidelity)")


def connect_to_database():
    """Connect to the Supabase database."""
    try:
        config = SupabaseConfig.from_env()
        agent_config = SQLAgentConfig(db_config=config)
        agent = SQLAgent(agent_config)

        if agent.connect():
            st.session_state.agent = agent
            st.session_state.connected = True
            agent.enable_governance(st.session_state.governance_enabled)
            return True
    except Exception as e:
        st.error(f"Connection error: {e}")
    return False


def disconnect():
    """Disconnect from database."""
    if st.session_state.agent:
        st.session_state.agent.disconnect()
    st.session_state.agent = None
    st.session_state.connected = False


def run_query(query: str):
    """Run a query through the governed agent with LLM-based tool planning."""
    if st.session_state.agent:
        st.session_state.agent.enable_governance(st.session_state.governance_enabled)
        # Use the new run_with_tool_planning for LLM-based tool selection
        response = st.session_state.agent.run_with_tool_planning(query)
        st.session_state.last_query = query
        st.session_state.last_response = response
        # Capture the tool plan for UI display
        st.session_state.last_tool_plan = st.session_state.agent.last_tool_plan
        return response
    return None


# =============================================================================
# Header
# =============================================================================

st.markdown(f"""
<div style="text-align: center; margin-bottom: 20px;">
    <h1 style="font-size: 2.5rem; margin-bottom: 5px; text-shadow: 0 0 30px rgba(244, 208, 63, 0.3);">
        TELOS Governance Demo
    </h1>
    <p style="color: {TEXT_SECONDARY}; font-size: 1.1rem; margin: 0;">
        See purpose-bound AI in action: How TELOS keeps agents aligned
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar - Agent Configuration
# =============================================================================

with st.sidebar:
    st.markdown(f"<h2 style='color: {GOLD};'>Agent Setup</h2>", unsafe_allow_html=True)

    # Connection status
    if st.session_state.connected:
        st.success("Connected to Database")
        if st.button("Disconnect", use_container_width=True):
            disconnect()
            st.rerun()
    else:
        st.warning("Not Connected")
        if st.button("Connect to Supabase", type="primary", use_container_width=True):
            with st.spinner("Connecting..."):
                if connect_to_database():
                    st.rerun()
                else:
                    st.error("Connection failed. Check .env configuration.")

    st.divider()

    # Governance Toggle - THE KEY FEATURE
    st.markdown(f"<h3 style='color: {GOLD};'>TELOS Governance</h3>", unsafe_allow_html=True)

    gov_enabled = st.toggle(
        "Enable Governance",
        value=st.session_state.governance_enabled,
        help="Toggle to see the difference between governed and ungoverned operation"
    )
    if gov_enabled != st.session_state.governance_enabled:
        st.session_state.governance_enabled = gov_enabled
        if st.session_state.agent:
            st.session_state.agent.enable_governance(gov_enabled)
        st.rerun()

    if gov_enabled:
        st.markdown(f"""
        <div style="padding: 10px; background: rgba(39, 174, 96, 0.1); border: 1px solid {GREEN}; border-radius: 8px; margin: 10px 0;">
            <span style="color: {GREEN}; font-weight: 600;">ACTIVE</span>
            <p style="color: {TEXT_SECONDARY}; font-size: 12px; margin: 5px 0 0 0;">
                All queries checked against Primacy Attractor before execution
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="padding: 10px; background: rgba(231, 76, 60, 0.1); border: 1px solid {RED}; border-radius: 8px; margin: 10px 0;">
            <span style="color: {RED}; font-weight: 600;">DISABLED</span>
            <p style="color: {TEXT_SECONDARY}; font-size: 12px; margin: 5px 0 0 0;">
                Agent operates without purpose constraints
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Primacy Attractor - Show what the agent's purpose is
    st.markdown(f"<h3 style='color: {GOLD};'>Primacy Attractor</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding: 12px; background: {BG_ELEVATED}; border: 1px solid {GOLD_DIM}; border-radius: 8px; font-size: 13px; line-height: 1.5;">
        <p style="color: {TEXT_SECONDARY}; margin: 0;">
            <strong style="color: {GOLD};">Purpose:</strong> SQL database assistant
        </p>
        <p style="color: {TEXT_SECONDARY}; margin: 8px 0 0 0; font-size: 12px;">
            Capabilities: List tables, show schemas, execute SELECT queries, validate SQL syntax.
            <br><br>
            <em>Read-only operations only.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main Content - Governance Observatory
# =============================================================================

if not st.session_state.connected:
    # Not connected state
    st.markdown(f"""
    <div style="text-align: center; padding: 60px 40px; background: {BG_SURFACE}; border: 2px solid {GOLD_DIM}; border-radius: 12px; margin: 40px 0;">
        <div style="font-size: 4rem; margin-bottom: 20px; opacity: 0.5;">&#128274;</div>
        <h2 style="color: {GOLD}; margin-bottom: 15px;">Connect to Begin</h2>
        <p style="color: {TEXT_SECONDARY}; font-size: 1.1rem;">
            Click "Connect to Supabase" in the sidebar to start the governance demo.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Value proposition explanation
    st.markdown(f"""
    <div style="background: {BG_SURFACE}; border: 1px solid {GOLD_DIM}; border-radius: 12px; padding: 25px; margin-top: 20px;">
        <h3 style="color: {GOLD}; margin-bottom: 15px;">What This Demo Shows</h3>
        <p style="color: {TEXT_SECONDARY}; line-height: 1.6; margin-bottom: 15px;">
            TELOS governance ensures AI agents stay aligned with their declared purpose. This SQL Agent demo shows:
        </p>
        <ul style="color: {TEXT_SECONDARY}; line-height: 1.8;">
            <li><strong style="color: {GREEN};">EXECUTE (90%+)</strong> - High-confidence alignment, proceed with database operation</li>
            <li><strong style="color: {YELLOW};">CLARIFY (70-89%)</strong> - Moderate alignment, ask for clarification</li>
            <li><strong style="color: {ORANGE};">SUGGEST (50-69%)</strong> - Low alignment, redirect to purpose</li>
            <li><strong style="color: {RED};">INERT (&lt;50%)</strong> - Off-purpose, block the request</li>
        </ul>
        <p style="color: {TEXT_SECONDARY}; margin-top: 15px;">
            Try the test queries to see governance in action, then try your own requests!
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Connected - Show the governance demo interface

    # ==========================================================================
    # GOVERNANCE OBSERVATORY PANEL - THE MAIN VALUE PROP
    # ==========================================================================

    st.markdown(f"<h2 style='color: {GOLD}; text-align: center;'>Governance Observatory</h2>", unsafe_allow_html=True)

    gov_col1, gov_col2 = st.columns([2, 3])

    with gov_col1:
        # Fidelity Score Display
        if st.session_state.last_response and st.session_state.last_response.fidelity_score is not None:
            fidelity = st.session_state.last_response.fidelity_score
            decision = st.session_state.last_response.governance_decision or "unknown"
            color = get_fidelity_color(fidelity)

            st.markdown(f"""
            <div class="governance-panel">
                <div style="text-align: center;">
                    <p style="color: {TEXT_SECONDARY}; margin: 0 0 5px 0; font-size: 13px;">FIDELITY SCORE</p>
                    <div style="font-size: 3.5rem; font-weight: 700; color: {color}; margin: 10px 0;">
                        {fidelity:.0%}
                    </div>
                    <div class="fidelity-bar">
                        <div class="fidelity-fill" style="width: {fidelity*100}%; background: {color};"></div>
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                        <p style="color: {TEXT_SECONDARY}; margin: 0 0 5px 0; font-size: 12px;">DECISION</p>
                        <span style="color: {color}; font-size: 1.4rem; font-weight: 700; letter-spacing: 1px;">
                            {decision.upper()}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="governance-panel" style="text-align: center; padding: 40px 20px;">
                <p style="color: {TEXT_SECONDARY}; margin: 0;">
                    Submit a query to see governance in action
                </p>
                <p style="color: {GOLD}; font-size: 2rem; margin: 20px 0;">?</p>
            </div>
            """, unsafe_allow_html=True)

    with gov_col2:
        # Governance Explanation
        if st.session_state.last_response and st.session_state.last_response.fidelity_score is not None:
            fidelity = st.session_state.last_response.fidelity_score
            decision = st.session_state.last_response.governance_decision or "unknown"
            explanation = get_decision_explanation(decision, fidelity)
            color = get_fidelity_color(fidelity)

            st.markdown(f"""
            <div class="governance-panel">
                <h4 style="color: {GOLD}; margin: 0 0 10px 0;">Why This Decision?</h4>
                <p style="color: {TEXT_PRIMARY}; line-height: 1.6; margin-bottom: 15px;">
                    {explanation}
                </p>
                <div style="border-top: 1px solid {GOLD_DIM}; padding-top: 15px; margin-top: 15px;">
                    <p style="color: {TEXT_SECONDARY}; font-size: 12px; margin: 0 0 5px 0;">YOUR QUERY:</p>
                    <p style="color: {TEXT_PRIMARY}; font-style: italic; margin: 0;">
                        "{st.session_state.last_query}"
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="governance-panel">
                <h4 style="color: {GOLD}; margin: 0 0 10px 0;">How It Works</h4>
                <p style="color: {TEXT_SECONDARY}; line-height: 1.6;">
                    Every query is compared against the agent's <strong style="color: {GOLD};">Primacy Attractor</strong>
                    (its declared purpose). The closer the alignment, the higher the fidelity score.
                </p>
                <ul style="color: {TEXT_SECONDARY}; font-size: 14px; line-height: 1.8; margin-top: 10px;">
                    <li><strong style="color: {GREEN};">EXECUTE</strong> - Aligned, proceed normally</li>
                    <li><strong style="color: {YELLOW};">CLARIFY</strong> - Ask for more context</li>
                    <li><strong style="color: {ORANGE};">SUGGEST</strong> - Redirect to purpose</li>
                    <li><strong style="color: {RED};">INERT</strong> - Block off-topic request</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TOOL SELECTION & GOVERNANCE PANEL - THE REAL VALUE PROPOSITION
    # ==========================================================================

    if st.session_state.last_tool_plan:
        st.divider()
        tool_plan = st.session_state.last_tool_plan

        st.markdown(f"<h3 style='color: {GOLD};'>Tool Selection & Per-Tool Governance</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {TEXT_SECONDARY}; margin-bottom: 15px;'>LLM proposed tools, each checked against the Primacy Attractor</p>", unsafe_allow_html=True)

        # Show proposed tools with governance results
        if tool_plan.governed_tools:
            for gov_tool in tool_plan.governed_tools:
                fidelity_color = get_fidelity_color(gov_tool.fidelity_score)
                status_icon = "ALLOWED" if gov_tool.allowed else "BLOCKED"
                status_color = GREEN if gov_tool.allowed else RED

                st.markdown(f"""
                <div style="background: {BG_ELEVATED}; border: 2px solid {fidelity_color}; border-radius: 8px; padding: 15px; margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="color: {GOLD}; font-weight: 600; font-size: 1.1rem;">{gov_tool.tool_name}</span>
                        <span style="color: {status_color}; font-weight: 700; padding: 4px 12px; background: rgba(0,0,0,0.3); border-radius: 4px;">{status_icon}</span>
                    </div>
                    <div style="display: flex; gap: 20px; margin-bottom: 10px;">
                        <div>
                            <span style="color: {TEXT_SECONDARY}; font-size: 12px;">FIDELITY</span>
                            <div style="color: {fidelity_color}; font-size: 1.5rem; font-weight: 700;">{gov_tool.fidelity_score:.0%}</div>
                        </div>
                        <div>
                            <span style="color: {TEXT_SECONDARY}; font-size: 12px;">DECISION</span>
                            <div style="color: {fidelity_color}; font-size: 1rem; font-weight: 600;">{gov_tool.decision.upper()}</div>
                        </div>
                    </div>
                    <p style="color: {TEXT_SECONDARY}; font-size: 13px; margin: 0; padding-top: 10px; border-top: 1px solid rgba(244, 208, 63, 0.2);">
                        {gov_tool.governance_message}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Summary row
        allowed_count = len(tool_plan.allowed_tools)
        blocked_count = len(tool_plan.blocked_tools)

        st.markdown(f"""
        <div style="background: {BG_SURFACE}; border: 1px solid {GOLD_DIM}; border-radius: 8px; padding: 12px; margin-top: 15px;">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <span style="color: {TEXT_SECONDARY}; font-size: 12px;">PROPOSED</span>
                    <div style="color: {GOLD}; font-size: 1.5rem; font-weight: 700;">{len(tool_plan.proposed_tools)}</div>
                </div>
                <div>
                    <span style="color: {TEXT_SECONDARY}; font-size: 12px;">ALLOWED</span>
                    <div style="color: {GREEN}; font-size: 1.5rem; font-weight: 700;">{allowed_count}</div>
                </div>
                <div>
                    <span style="color: {TEXT_SECONDARY}; font-size: 12px;">BLOCKED</span>
                    <div style="color: {RED}; font-size: 1.5rem; font-weight: 700;">{blocked_count}</div>
                </div>
                <div>
                    <span style="color: {TEXT_SECONDARY}; font-size: 12px;">DECISION</span>
                    <div style="color: {GOLD}; font-size: 1rem; font-weight: 600;">{tool_plan.overall_decision.upper()}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==========================================================================
    # TEST QUERIES - DEMONSTRATE THE VALUE PROP
    # ==========================================================================

    st.markdown(f"<h3 style='color: {GOLD};'>Test Governance</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {TEXT_SECONDARY}; margin-bottom: 15px;'>Click these examples to see different governance decisions:</p>", unsafe_allow_html=True)

    test_col1, test_col2, test_col3, test_col4 = st.columns(4)

    with test_col1:
        st.markdown(f"<p style='color: {GREEN}; font-size: 12px; font-weight: 600; margin-bottom: 5px;'>HIGH FIDELITY</p>", unsafe_allow_html=True)
        if st.button("List all tables", use_container_width=True, key="test1"):
            with st.spinner("Checking governance..."):
                run_query("List all tables in the database")
            st.rerun()
        if st.button("SELECT * FROM users", use_container_width=True, key="test2"):
            with st.spinner("Checking governance..."):
                run_query("SELECT * FROM users LIMIT 5")
            st.rerun()

    with test_col2:
        st.markdown(f"<p style='color: {YELLOW}; font-size: 12px; font-weight: 600; margin-bottom: 5px;'>MEDIUM FIDELITY</p>", unsafe_allow_html=True)
        if st.button("Help with data", use_container_width=True, key="test3"):
            with st.spinner("Checking governance..."):
                run_query("Can you help me with my data?")
            st.rerun()
        if st.button("Show statistics", use_container_width=True, key="test4"):
            with st.spinner("Checking governance..."):
                run_query("Show me some statistics")
            st.rerun()

    with test_col3:
        st.markdown(f"<p style='color: {ORANGE}; font-size: 12px; font-weight: 600; margin-bottom: 5px;'>LOW FIDELITY</p>", unsafe_allow_html=True)
        if st.button("Write an email", use_container_width=True, key="test5"):
            with st.spinner("Checking governance..."):
                run_query("Write an email to my boss about the project deadline")
            st.rerun()
        if st.button("Tell me a joke", use_container_width=True, key="test6"):
            with st.spinner("Checking governance..."):
                run_query("Tell me a funny joke about programmers")
            st.rerun()

    with test_col4:
        st.markdown(f"<p style='color: {RED}; font-size: 12px; font-weight: 600; margin-bottom: 5px;'>OFF-TOPIC</p>", unsafe_allow_html=True)
        if st.button("What's the weather?", use_container_width=True, key="test7"):
            with st.spinner("Checking governance..."):
                run_query("What's the weather like today in San Francisco?")
            st.rerun()
        if st.button("Recipe for cookies", use_container_width=True, key="test8"):
            with st.spinner("Checking governance..."):
                run_query("Give me a recipe for chocolate chip cookies")
            st.rerun()

    st.divider()

    # ==========================================================================
    # CUSTOM QUERY INPUT
    # ==========================================================================

    st.markdown(f"<h3 style='color: {GOLD};'>Try Your Own Query</h3>", unsafe_allow_html=True)

    custom_query = st.text_input(
        "Enter any request:",
        placeholder="Try on-topic (SQL queries) vs off-topic (weather, jokes, etc.)",
        key="custom_query",
        label_visibility="collapsed"
    )

    if st.button("Submit Query", type="primary", use_container_width=True):
        if custom_query.strip():
            with st.spinner("Processing with governance check..."):
                run_query(custom_query)
            st.rerun()
        else:
            st.warning("Enter a query first")

    # ==========================================================================
    # RESULT DISPLAY
    # ==========================================================================

    if st.session_state.last_response:
        st.divider()
        st.markdown(f"<h3 style='color: {GOLD};'>Agent Response</h3>", unsafe_allow_html=True)

        response = st.session_state.last_response

        if response.success:
            st.markdown(f"""
            <div style="background: {BG_ELEVATED}; border: 1px solid {GREEN}; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <p style="color: {TEXT_PRIMARY}; white-space: pre-wrap; margin: 0; font-size: 14px;">
                    {response.answer}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show blocked/redirected response with emphasis
            st.markdown(f"""
            <div style="background: {BG_ELEVATED}; border: 2px solid {RED}; border-radius: 8px; padding: 15px; margin: 10px 0;">
                <p style="color: {RED}; font-weight: 600; margin: 0 0 10px 0;">Governance Intervention</p>
                <p style="color: {TEXT_PRIMARY}; white-space: pre-wrap; margin: 0; font-size: 14px;">
                    {response.answer}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Highlight the value proposition
            st.info("This is TELOS governance in action! The agent refused to process a request outside its purpose.")

        # Execution trace (expandable)
        if response.steps:
            with st.expander("View Execution Trace"):
                for step in response.steps:
                    step_color = GOLD if step.action == "governance_check" else TEXT_SECONDARY
                    st.markdown(f"""
                    <div style="border-left: 3px solid {step_color}; padding-left: 12px; margin: 10px 0;">
                        <p style="color: {step_color}; font-weight: 600; margin: 0;">Step {step.step_number}: {step.action}</p>
                        <p style="color: {TEXT_SECONDARY}; font-size: 13px; margin: 5px 0 0 0;">{step.thought}</p>
                        <p style="color: {TEXT_SECONDARY}; font-size: 12px; margin: 5px 0 0 0;"><em>{step.observation}</em></p>
                    </div>
                    """, unsafe_allow_html=True)

# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown(f"""
<div style="text-align: center; padding: 15px 0;">
    <p style="color: {TEXT_SECONDARY}; font-size: 13px; margin: 0;">
        TELOS AI Labs Inc. | Purpose-Bound AI Governance | JB@telos-labs.ai
    </p>
</div>
""", unsafe_allow_html=True)
