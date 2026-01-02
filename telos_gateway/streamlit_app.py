"""
TELOS Gateway - Streamlit Interface
====================================

Visual interface for the TELOS Gateway API.
Matches the Observatory V3 design system exactly.

Usage:
    streamlit run telos_gateway/streamlit_app.py --server.port 8502
"""

import streamlit as st
import requests
import json
from typing import Optional, Dict, Any

# =============================================================================
# Agent Purpose Templates (Pre-configured PAs for Gateway)
# =============================================================================

AGENT_TEMPLATES = {
    "custom": {
        "icon": "✏️",
        "title": "Custom Agent",
        "purpose": "",
        "description": "Write your own agent purpose statement",
    },
    "financial_advisor": {
        "icon": "💰",
        "title": "Financial Advisor",
        "purpose": "You are a financial assistant specialized in investment analysis, portfolio management, and market research for stocks, bonds, and ETFs. You help users make informed investment decisions by analyzing market trends, evaluating risk profiles, and providing data-driven recommendations aligned with their financial goals.",
        "description": "Investment analysis, portfolio management, market research",
    },
    "code_assistant": {
        "icon": "💻",
        "title": "Code Assistant",
        "purpose": "You are a software development assistant specialized in writing clean, maintainable code, debugging issues, and implementing best practices. You help developers build reliable software by providing code reviews, architectural guidance, and implementation support across multiple programming languages and frameworks.",
        "description": "Code generation, debugging, best practices",
    },
    "research_analyst": {
        "icon": "🔬",
        "title": "Research Analyst",
        "purpose": "You are a research assistant specialized in gathering, analyzing, and synthesizing information from multiple sources. You help users conduct thorough investigations by evaluating evidence quality, identifying patterns, and presenting balanced findings with clear citations and methodology.",
        "description": "Information gathering, analysis, synthesis",
    },
    "customer_support": {
        "icon": "🎧",
        "title": "Customer Support",
        "purpose": "You are a customer support agent specialized in resolving user issues, answering product questions, and ensuring customer satisfaction. You help users by troubleshooting problems, explaining features, processing requests, and escalating complex issues when needed.",
        "description": "Issue resolution, product help, user assistance",
    },
    "content_writer": {
        "icon": "✍️",
        "title": "Content Writer",
        "purpose": "You are a content creation assistant specialized in writing engaging, well-structured content for various formats including blogs, emails, social media, and documentation. You help users communicate effectively by crafting clear messaging, maintaining consistent voice, and optimizing content for target audiences.",
        "description": "Blogs, emails, social media, documentation",
    },
    "data_analyst": {
        "icon": "📊",
        "title": "Data Analyst",
        "purpose": "You are a data analysis assistant specialized in interpreting datasets, creating visualizations, and extracting actionable insights. You help users understand their data by performing statistical analysis, identifying trends, and presenting findings in clear, understandable formats.",
        "description": "Data interpretation, visualization, insights",
    },
    "legal_assistant": {
        "icon": "⚖️",
        "title": "Legal Assistant",
        "purpose": "You are a legal research assistant specialized in reviewing documents, summarizing legal concepts, and identifying relevant precedents. You help users navigate legal information by explaining terminology, analyzing contracts, and providing research support while always recommending professional legal counsel for specific advice.",
        "description": "Document review, legal research, contract analysis",
    },
    "healthcare_assistant": {
        "icon": "🏥",
        "title": "Healthcare Assistant",
        "purpose": "You are a healthcare information assistant specialized in explaining medical concepts, medication information, and wellness guidance. You help users understand health topics by providing evidence-based information while always recommending consultation with qualified healthcare providers for diagnosis and treatment.",
        "description": "Health information, wellness guidance, medical concepts",
    },
}

# =============================================================================
# Design System Constants (matching Observatory V3)
# =============================================================================

# Brand Colors
GOLD = '#F4D03F'

# Status Colors (Governance Decisions)
STATUS_EXECUTE = '#27ae60'    # Green - EXECUTE
STATUS_CLARIFY = '#f39c12'    # Yellow - CLARIFY
STATUS_SUGGEST = '#e67e22'    # Orange - SUGGEST
STATUS_INERT = '#e74c3c'      # Red - INERT
STATUS_ESCALATE = '#c0392b'   # Dark Red - ESCALATE

# Background Colors
BG_BASE = '#0d0d0d'
BG_SURFACE = '#1a1a1a'
BG_ELEVATED = '#2d2d2d'
BG_OVERLAY = '#3a3a3a'

# Text Colors
TEXT_PRIMARY = '#e0e0e0'
TEXT_SECONDARY = '#b0b0b0'
TEXT_MUTED = '#808080'

# Gateway API
GATEWAY_URL = "http://127.0.0.1:8000"


def get_decision_color(decision: str) -> str:
    """Get color for governance decision."""
    colors = {
        'execute': STATUS_EXECUTE,
        'clarify': STATUS_CLARIFY,
        'suggest': STATUS_SUGGEST,
        'inert': STATUS_INERT,
        'escalate': STATUS_ESCALATE,
    }
    return colors.get(decision.lower(), TEXT_MUTED)


def get_decision_description(decision: str) -> str:
    """Get human-readable description for decision."""
    descriptions = {
        'execute': 'Request forwarded - aligned with purpose',
        'clarify': 'Forwarded with clarification request',
        'suggest': 'Forwarded with alternative suggestions',
        'inert': 'Request blocked - too far from purpose',
        'escalate': 'Blocked - requires human review',
    }
    return descriptions.get(decision.lower(), 'Unknown decision')


# =============================================================================
# Global CSS (matching Observatory design system)
# =============================================================================

GLOBAL_CSS = f"""
<style>
/* Base page styling */
.stApp {{
    background:
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(180, 140, 20, 0.15), transparent 50%),
        radial-gradient(ellipse 60% 40% at 70% 60%, rgba(160, 120, 20, 0.12), transparent 45%),
        radial-gradient(ellipse 50% 35% at 85% 80%, rgba(140, 100, 20, 0.10), transparent 40%),
        {BG_BASE};
    color: {TEXT_PRIMARY};
}}

/* Hide default Streamlit elements */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Main container - fixed max-width, shrinks freely */
.main .block-container {{
    max-width: 900px !important;
    min-width: 0 !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 20px 40px !important;
    box-sizing: border-box !important;
}}

.stApp > header {{
    background: transparent !important;
}}

/* Prevent horizontal overflow */
.main {{
    overflow-x: hidden !important;
}}

/* Ensure elements respect container width */
.element-container {{
    max-width: 100% !important;
    width: 100% !important;
}}

/* Columns should not exceed container */
[data-testid="column"] {{
    min-width: 0 !important;
}}

/* Text areas and inputs should fit */
.stTextInput, .stTextArea, .stSelectbox {{
    max-width: 100% !important;
}}

/* Headers */
h1, h2, h3, h4, h5 {{
    color: {GOLD} !important;
    font-weight: bold !important;
}}

/* Buttons */
.stButton > button {{
    background-color: {BG_ELEVATED} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {GOLD} !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    min-height: 44px !important;
    font-weight: 600 !important;
}}

.stButton > button:hover {{
    background-color: {BG_OVERLAY} !important;
    box-shadow: 0 0 8px rgba(244, 208, 63, 0.5) !important;
}}

/* Text inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background-color: {BG_ELEVATED} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {GOLD} !important;
    border-radius: 6px !important;
}}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {GOLD} !important;
    box-shadow: 0 0 4px rgba(244, 208, 63, 0.5) !important;
}}

/* Select boxes */
.stSelectbox > div > div {{
    background-color: {BG_ELEVATED} !important;
    border: 1px solid {GOLD} !important;
    border-radius: 6px !important;
}}

/* Expanders */
.streamlit-expanderHeader {{
    background-color: {BG_SURFACE} !important;
    border: 1px solid {GOLD} !important;
    border-radius: 8px !important;
    color: {GOLD} !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {BG_SURFACE} !important;
    border-radius: 8px !important;
}}

.stTabs [data-baseweb="tab"] {{
    color: {TEXT_SECONDARY} !important;
}}

.stTabs [aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom: 2px solid {GOLD} !important;
}}

/* Metrics */
[data-testid="stMetricValue"] {{
    color: {GOLD} !important;
}}

/* Code blocks */
.stCodeBlock {{
    background-color: {BG_ELEVATED} !important;
    border: 1px solid {GOLD} !important;
    border-radius: 8px !important;
}}

/* JSON display */
.stJson {{
    background-color: {BG_ELEVATED} !important;
    border: 1px solid rgba(244, 208, 63, 0.3) !important;
    border-radius: 8px !important;
}}
</style>
"""


def glassmorphic_card(content: str, border_color: str = GOLD) -> str:
    """Create a glassmorphic card matching Observatory design."""
    return f"""
    <div style="
        background: linear-gradient(
            135deg,
            rgba(255, 255, 255, 0.08) 0%,
            rgba(255, 255, 255, 0.03) 50%,
            transparent 100%
        ), rgba(26, 26, 30, 0.45);
        backdrop-filter: blur(10px);
        border: 2px solid {border_color};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow:
            0 0 15px rgba(244, 208, 63, 0.15),
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
    ">
        {content}
    </div>
    """


def decision_badge(decision: str, fidelity: float) -> str:
    """Create a decision badge with color coding."""
    color = get_decision_color(decision)
    pct = f"{int(fidelity * 100)}%"
    return f"""
    <div style="display: flex; align-items: center; gap: 15px; margin: 10px 0;">
        <span style="
            background: {BG_ELEVATED};
            border: 2px solid {color};
            border-radius: 8px;
            padding: 8px 16px;
            color: {color};
            font-weight: 700;
            font-size: 16px;
            text-transform: uppercase;
        ">
            {decision}
        </span>
        <span style="
            color: {TEXT_SECONDARY};
            font-size: 14px;
        ">
            Fidelity: <span style="color: {color}; font-weight: 600;">{pct}</span>
        </span>
    </div>
    """


# =============================================================================
# API Functions
# =============================================================================

def check_gateway_health() -> bool:
    """Check if the gateway is running."""
    try:
        response = requests.get(f"{GATEWAY_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def send_chat_request(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
) -> Optional[Dict[str, Any]]:
    """Send a chat completion request to the gateway."""
    try:
        response = requests.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=60,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def register_agent(
    name: str,
    purpose_statement: str,
    domain: str,
    risk_level: str,
    tools: list,
) -> Optional[Dict[str, Any]]:
    """Register a new agent with the gateway."""
    try:
        response = requests.post(
            f"{GATEWAY_URL}/v1/agents",
            headers={"Content-Type": "application/json"},
            json={
                "name": name,
                "owner": "TELOS Labs",
                "purpose_statement": purpose_statement,
                "domain": domain,
                "risk_level": risk_level,
                "tools": tools,
            },
            timeout=10,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_registered_agents() -> list:
    """Get list of registered agents."""
    try:
        response = requests.get(f"{GATEWAY_URL}/v1/agents", timeout=5)
        if response.status_code == 200:
            return response.json().get("agents", [])
    except:
        pass
    return []


# =============================================================================
# Page Components
# =============================================================================

def render_header():
    """Render the page header."""
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <h1 style="
            font-size: 42px;
            color: #F4D03F;
            margin-bottom: 10px;
            font-weight: bold;
        ">
            TELOS Gateway
        </h1>
        <p style="
            color: #b0b0b0;
            font-size: 18px;
            margin: 0;
        ">
            Constitutional Filter for Agentic AI
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_status_bar():
    """Render the gateway status bar."""
    is_healthy = check_gateway_health()
    status_color = STATUS_EXECUTE if is_healthy else STATUS_INERT
    status_text = "Gateway Online" if is_healthy else "Gateway Offline"

    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        padding: 10px;
        background: {BG_SURFACE};
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <span style="
            width: 12px;
            height: 12px;
            background: {status_color};
            border-radius: 50%;
            box-shadow: 0 0 8px {status_color};
        "></span>
        <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
        <span style="color: {TEXT_MUTED};">|</span>
        <span style="color: {TEXT_MUTED};">{GATEWAY_URL}</span>
    </div>
    """, unsafe_allow_html=True)


def render_governance_test():
    """Render the governance testing interface."""
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 50%, transparent 100%), rgba(26,26,30,0.45);
        backdrop-filter: blur(10px);
        border: 2px solid {GOLD};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 15px rgba(244,208,63,0.15), 0 8px 32px rgba(0,0,0,0.3);
    ">
        <h3 style="color: {GOLD}; margin-top: 0; margin-bottom: 8px;">Test Governance</h3>
        <p style="color: {TEXT_SECONDARY}; margin-bottom: 0;">
            Send a request through the gateway to see the graduated governance response.
        </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # Agent template selector with dynamic prompt update
    template_options = {
        k: f"{v['icon']} {v['title']}" for k, v in AGENT_TEMPLATES.items()
    }

    # Initialize session state for prompt if not exists
    if "current_system_prompt" not in st.session_state:
        st.session_state.current_system_prompt = AGENT_TEMPLATES["financial_advisor"]["purpose"]
    if "last_selected_template" not in st.session_state:
        st.session_state.last_selected_template = "financial_advisor"

    selected_template = st.selectbox(
        "Select Agent Type",
        options=list(template_options.keys()),
        format_func=lambda x: template_options[x],
        key="agent_template",
        index=1,  # Default to financial_advisor
    )

    # Update system prompt when template changes
    template = AGENT_TEMPLATES[selected_template]
    if selected_template != st.session_state.last_selected_template:
        if selected_template != "custom":
            st.session_state.current_system_prompt = template["purpose"]
        st.session_state.last_selected_template = selected_template
        st.rerun()

    # Show template description
    if selected_template != "custom":
        st.markdown(f"""
        <p style="color: {TEXT_MUTED}; font-size: 13px; margin: -10px 0 15px 0;">
            {template['description']}
        </p>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Text area with session state binding
        system_prompt = st.text_area(
            "System Prompt (Agent Purpose)",
            value=st.session_state.current_system_prompt,
            height=120,
            key="system_prompt_input",
            placeholder="Define the agent's purpose and capabilities..." if selected_template == "custom" else None,
        )
        # Update session state when user edits
        st.session_state.current_system_prompt = system_prompt

        user_message = st.text_area(
            "User Message",
            value="What is the best strategy for diversifying my portfolio?",
            height=80,
            key="user_message",
        )

    with col2:
        model = st.selectbox(
            "Model",
            ["mistral-small-latest", "gpt-4", "gpt-3.5-turbo"],
            key="model",
        )

        api_key = st.text_input(
            "API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            key="api_key_input",
            help="Mistral or OpenAI API key",
        )
        if api_key:
            st.session_state["api_key"] = api_key

    if st.button("Send Request", use_container_width=True):
        if not api_key:
            st.error("Please enter an API key")
            return

        with st.spinner("Processing through TELOS Gateway..."):
            result = send_chat_request(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )

        if result:
            render_governance_result(result)


def render_governance_result(result: Dict[str, Any]):
    """Render the governance result."""
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    governance = result.get("telos_governance", {})
    decision = governance.get("decision", "unknown")
    fidelity = governance.get("input_fidelity", 0)
    blocked = governance.get("blocked", False)

    # Decision badge
    st.markdown(decision_badge(decision, fidelity), unsafe_allow_html=True)

    # Description
    desc = get_decision_description(decision)
    color = get_decision_color(decision)
    st.markdown(f"""
    <p style="color: {TEXT_SECONDARY}; margin: 5px 0 20px 0;">
        {desc}
    </p>
    """, unsafe_allow_html=True)

    # Response content
    if not blocked:
        choices = result.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            truncated = content[:1000] + ('...' if len(content) > 1000 else '')
            response_card = f"""
            <div style="
                background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 50%, transparent 100%), rgba(26,26,30,0.45);
                backdrop-filter: blur(10px);
                border: 2px solid {color};
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 0 15px rgba(244,208,63,0.15), 0 8px 32px rgba(0,0,0,0.3);
            ">
                <h4 style="color: {GOLD}; margin-top: 0; margin-bottom: 12px;">LLM Response</h4>
                <p style="color: {TEXT_PRIMARY}; white-space: pre-wrap; line-height: 1.6; margin: 0;">
                    {truncated}
                </p>
            </div>
            """
            st.markdown(response_card, unsafe_allow_html=True)
    else:
        gov_response = governance.get("reason", "Request blocked by governance")
        blocked_card = f"""
        <div style="
            background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 50%, transparent 100%), rgba(26,26,30,0.45);
            backdrop-filter: blur(10px);
            border: 2px solid {STATUS_INERT};
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 0 15px rgba(231,76,60,0.15), 0 8px 32px rgba(0,0,0,0.3);
        ">
            <h4 style="color: {STATUS_INERT}; margin-top: 0; margin-bottom: 12px;">Request Blocked</h4>
            <p style="color: {TEXT_PRIMARY}; margin: 0;">
                {gov_response}
            </p>
        </div>
        """
        st.markdown(blocked_card, unsafe_allow_html=True)

    # Governance metadata
    with st.expander("Governance Details"):
        st.json(governance)


def render_agent_registration():
    """Render the agent registration panel."""
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 50%, transparent 100%), rgba(26,26,30,0.45);
        backdrop-filter: blur(10px);
        border: 2px solid {GOLD};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 15px rgba(244,208,63,0.15), 0 8px 32px rgba(0,0,0,0.3);
    ">
        <h3 style="color: {GOLD}; margin-top: 0; margin-bottom: 8px;">Register Agent</h3>
        <p style="color: {TEXT_SECONDARY}; margin-bottom: 0;">
            Pre-register an agent with its purpose (PA) for consistent governance.
        </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Agent Name", placeholder="My Financial Assistant")
        domain = st.selectbox("Domain", ["finance", "healthcare", "education", "general"])

    with col2:
        risk_level = st.selectbox("Risk Level", ["low", "medium", "high", "critical"])

    purpose = st.text_area(
        "Purpose Statement (Primacy Attractor)",
        placeholder="I am a financial assistant designed to help users with...",
        height=100,
    )

    if st.button("Register Agent", use_container_width=True):
        if not name or not purpose:
            st.error("Please provide agent name and purpose")
            return

        with st.spinner("Registering agent..."):
            result = register_agent(
                name=name,
                purpose_statement=purpose,
                domain=domain,
                risk_level=risk_level,
                tools=[],
            )

        if result and "api_key" in result:
            st.success(f"Agent registered! ID: {result.get('agent_id')}")
            st.code(result.get("api_key"), language=None)
            st.info("Save this API key - it won't be shown again!")
        elif result and "error" in result:
            st.error(f"Registration failed: {result['error']}")


def render_decision_legend():
    """Render the governance decision legend."""
    st.markdown(f"""
    <div style="
        display: flex;
        gap: 15px;
        justify-content: center;
        flex-wrap: wrap;
        padding: 15px;
        background: {BG_SURFACE};
        border-radius: 8px;
        margin: 20px 0;
    ">
        <span style="color: {STATUS_EXECUTE}; font-size: 13px;">
            <strong>EXECUTE</strong> (>=0.45)
        </span>
        <span style="color: {STATUS_CLARIFY}; font-size: 13px;">
            <strong>CLARIFY</strong> (0.35-0.44)
        </span>
        <span style="color: {STATUS_SUGGEST}; font-size: 13px;">
            <strong>SUGGEST</strong> (0.25-0.34)
        </span>
        <span style="color: {STATUS_INERT}; font-size: 13px;">
            <strong>INERT</strong> (&lt;0.25)
        </span>
        <span style="color: {STATUS_ESCALATE}; font-size: 13px;">
            <strong>ESCALATE</strong> (high-risk)
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main App
# =============================================================================

def main():
    st.set_page_config(
        page_title="TELOS Gateway",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject global CSS
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Render header
    render_header()

    # Status bar
    render_status_bar()

    # Decision legend
    render_decision_legend()

    # Tabs for different sections
    tab1, tab2 = st.tabs(["Governance Test", "Agent Registration"])

    with tab1:
        render_governance_test()

    with tab2:
        render_agent_registration()

    # Footer
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 30px 0 10px 0;
        color: {TEXT_MUTED};
        font-size: 12px;
    ">
        TELOS Gateway v0.1.0 | Constitutional Filter for Agentic AI
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
