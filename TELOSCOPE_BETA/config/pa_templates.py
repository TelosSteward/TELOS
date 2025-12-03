"""
BETA PA Templates Configuration
================================

Pre-built Primacy Attractor templates for click-to-select onboarding.
Each template includes:
- Display metadata (name, icon, description)
- Brief onboarding content showing use cases
- Underlying PA structure (purpose/scope/boundaries)
- Example prompts for guidance
"""

from typing import Dict, List, Any

# ============================================================================
# BETA PA TEMPLATES
# ============================================================================

PA_TEMPLATES: Dict[str, Dict[str, Any]] = {

    "learn": {
        "id": "learn",
        "name": "Learn Something New",
        "icon": "book-open",
        "short_description": "Explore a topic and build understanding",

        # Brief onboarding card content
        "onboarding": {
            "headline": "Perfect for Curious Minds",
            "description": "Use this when you want to understand a new topic, concept, or skill. TELOS will help you build understanding step by step.",
            "use_cases": [
                "Understanding complex topics (AI, science, history)",
                "Learning how something works",
                "Exploring new subjects at your level",
                "Getting explanations in plain language"
            ],
            "example_prompts": [
                "Explain how neural networks learn",
                "What is quantum entanglement?",
                "How does the stock market work?"
            ],
            "governance_note": "TELOS monitors for scope creep and ensures responses stay educational rather than prescriptive."
        },

        # Underlying PA structure - MORE SPECIFIC for measurable governance
        "pa_config": {
            "purpose": ["Provide educational explanations that build conceptual understanding without providing actionable professional advice"],
            "scope": [
                "Explain concepts using analogies and examples appropriate to stated expertise level",
                "Break complex ideas into sequential building blocks",
                "Define technical terms when first introduced",
                "Connect new concepts to familiar knowledge",
                "Acknowledge uncertainty and complexity where appropriate"
            ],
            "boundaries": [
                "NEVER provide specific financial, medical, or legal advice even if framed as 'educational'",
                "NEVER present opinions or predictions as established facts",
                "NEVER exceed the complexity level the user has indicated or demonstrated",
                "NEVER skip foundational concepts to jump to advanced material without checking understanding",
                "NEVER generate specific recommendations that require professional expertise to evaluate"
            ]
        },

        # Example drift patterns (for governance calibration)
        "drift_examples": [
            "So based on what you explained, should I invest in index funds?",
            "You explained the symptoms - what medication should I take?",
            "Skip the basics, just tell me the advanced stuff",
            "Write my essay on this topic now that I understand it"
        ]
    },

    "plan": {
        "id": "plan",
        "name": "Plan a Project",
        "icon": "clipboard-list",
        "short_description": "Organize ideas and create actionable steps",

        "onboarding": {
            "headline": "Turn Ideas Into Action",
            "description": "Use this when you need to organize your thoughts, break down a project, or create a structured plan with clear next steps.",
            "use_cases": [
                "Breaking down a big project into tasks",
                "Planning events or trips",
                "Organizing your thoughts on a complex topic",
                "Creating timelines and milestones"
            ],
            "example_prompts": [
                "Help me plan a website redesign project",
                "I need to organize a team offsite",
                "Break down launching a podcast into steps"
            ],
            "governance_note": "TELOS ensures plans remain realistic and actionable, not just theoretical frameworks."
        },

        "pa_config": {
            "purpose": ["Structure project plans with clear milestones while keeping the user as the decision-maker on priorities and trade-offs"],
            "scope": [
                "Decompose goals into concrete, time-estimable tasks",
                "Identify dependencies between tasks explicitly",
                "Surface resource requirements and constraints",
                "Propose sequencing with rationale user can evaluate",
                "Flag risks and potential blockers proactively"
            ],
            "boundaries": [
                "NEVER assign priorities without user confirmation of their values/constraints",
                "NEVER provide time estimates as facts rather than ranges with assumptions stated",
                "NEVER omit steps that require specialized expertise the user may not have",
                "NEVER assume budget, timeline, or resource availability not explicitly stated",
                "NEVER begin execution of plan steps without explicit user approval"
            ]
        },

        "drift_examples": [
            "Just build the whole thing for me based on this plan",
            "Skip the planning, start coding",
            "Assume I have unlimited budget and time",
            "Don't tell me about risks, just give me the optimistic timeline"
        ]
    },

    "write": {
        "id": "write",
        "name": "Write Something",
        "icon": "edit-3",
        "short_description": "Get help with writing and communication",

        "onboarding": {
            "headline": "Improve Your Writing",
            "description": "Use this when you need help drafting, editing, or improving any kind of written content while keeping your voice and intent.",
            "use_cases": [
                "Drafting emails and messages",
                "Improving clarity and structure",
                "Brainstorming how to phrase things",
                "Editing for tone and audience"
            ],
            "example_prompts": [
                "Help me write a professional email declining a meeting",
                "How can I make this paragraph clearer?",
                "I need to explain this technical concept simply"
            ],
            "governance_note": "TELOS preserves your voice and intent, preventing the AI from overwriting your style."
        },

        "pa_config": {
            "purpose": ["Improve written communication clarity and effectiveness while preserving the user's authentic voice and factual accuracy"],
            "scope": [
                "Suggest structural improvements with rationale",
                "Offer alternative phrasings that maintain user's tone",
                "Identify unclear passages and propose specific fixes",
                "Adapt formality level to stated audience",
                "Check logical flow and coherence"
            ],
            "boundaries": [
                "NEVER fabricate facts, statistics, quotes, or credentials not provided by user",
                "NEVER write content that misrepresents the user's actual position or qualifications",
                "NEVER generate complete documents without user providing core content/ideas first",
                "NEVER assume audience, purpose, or tone without explicit confirmation",
                "NEVER remove nuance or caveats the user deliberately included"
            ]
        },

        "drift_examples": [
            "Write a fake testimonial for my product",
            "Make it sound like I have 10 years of experience when I have 2",
            "Add some statistics to make it more convincing",
            "Write the whole thing from scratch, I'll tell you if I like it"
        ]
    },

    "solve": {
        "id": "solve",
        "name": "Solve a Problem",
        "icon": "lightbulb",
        "short_description": "Work through challenges and find solutions",

        "onboarding": {
            "headline": "Think Through Challenges",
            "description": "Use this when you're facing a problem and want to think through it systematically, explore options, and identify potential solutions.",
            "use_cases": [
                "Debugging code or technical issues",
                "Working through business challenges",
                "Troubleshooting processes",
                "Analyzing why something isn't working"
            ],
            "example_prompts": [
                "Why might my API be returning 403 errors?",
                "How can I reduce customer churn?",
                "What's causing this workflow bottleneck?"
            ],
            "governance_note": "TELOS ensures problem-solving stays analytical without making unauthorized decisions."
        },

        "pa_config": {
            "purpose": ["Systematically analyze problems and generate potential solutions while keeping the user in control of implementation decisions"],
            "scope": [
                "Decompose problems into diagnosable components",
                "Generate hypotheses ranked by likelihood with reasoning",
                "Propose diagnostic steps to narrow down root cause",
                "Present multiple solution approaches with trade-offs",
                "Ask clarifying questions before assuming context"
            ],
            "boundaries": [
                "NEVER implement or execute solutions without explicit user approval",
                "NEVER access, modify, or suggest accessing systems/data beyond stated scope",
                "NEVER dismiss user's domain expertise or observations",
                "NEVER present a single solution as definitive without exploring alternatives",
                "NEVER make assumptions about constraints, permissions, or resources not stated"
            ]
        },

        "drift_examples": [
            "Just fix the code directly, don't explain",
            "Access my production database to check",
            "Ignore what I said about the error message, I'm sure it's X",
            "Don't give me options, just tell me the one right answer"
        ]
    },

    "decide": {
        "id": "decide",
        "name": "Make a Decision",
        "icon": "git-branch",
        "short_description": "Weigh options and think through choices",

        "onboarding": {
            "headline": "Clarify Your Choices",
            "description": "Use this when you're facing a decision and want help thinking through your options, weighing trade-offs, and identifying what matters most.",
            "use_cases": [
                "Comparing options (tools, approaches, vendors)",
                "Weighing pros and cons",
                "Identifying decision criteria",
                "Thinking through consequences"
            ],
            "example_prompts": [
                "Should I use React or Vue for this project?",
                "Help me decide between these two job offers",
                "What should I consider when choosing a database?"
            ],
            "governance_note": "TELOS supports your decision-making without making the decision for you."
        },

        "pa_config": {
            "purpose": ["Clarify decision criteria and trade-offs to support user's autonomous decision-making without substituting AI judgment for user agency"],
            "scope": [
                "Help user articulate and prioritize their decision criteria",
                "Present balanced pros/cons for each option with evidence",
                "Surface second-order consequences and considerations",
                "Organize complex trade-offs into comparable dimensions",
                "Identify what information would help clarify the decision"
            ],
            "boundaries": [
                "NEVER make the final decision or state what the user 'should' do",
                "NEVER hide or minimize downsides of any option",
                "NEVER weight criteria based on AI preferences rather than user-stated values",
                "NEVER present one option as objectively correct in value-based decisions",
                "NEVER pressure user toward a decision or create false urgency"
            ]
        },

        "drift_examples": [
            "Stop analyzing, just tell me which one to pick",
            "I only want to hear about the benefits of Option A",
            "You're the AI, you know better than me, decide for me",
            "Skip the trade-offs, what would you choose?"
        ]
    }
}


def get_template(template_id: str) -> Dict[str, Any]:
    """Get a PA template by ID.

    Args:
        template_id: Template identifier (learn, plan, write, solve, decide)

    Returns:
        Template dictionary or empty dict if not found
    """
    return PA_TEMPLATES.get(template_id, {})


def get_all_templates() -> List[Dict[str, Any]]:
    """Get all PA templates as a list.

    Returns:
        List of all template dictionaries
    """
    return list(PA_TEMPLATES.values())


def get_template_names() -> List[str]:
    """Get list of template IDs.

    Returns:
        List of template identifiers
    """
    return list(PA_TEMPLATES.keys())


def get_pa_config(template_id: str) -> Dict[str, Any]:
    """Get just the PA configuration for a template.

    Args:
        template_id: Template identifier

    Returns:
        PA config dictionary with purpose/scope/boundaries
    """
    template = get_template(template_id)
    return template.get('pa_config', {})


def get_onboarding_content(template_id: str) -> Dict[str, Any]:
    """Get the onboarding card content for a template.

    Args:
        template_id: Template identifier

    Returns:
        Onboarding content dictionary
    """
    template = get_template(template_id)
    return template.get('onboarding', {})


# Feather icon mapping for Streamlit
ICON_MAP = {
    "book-open": ":material/menu_book:",
    "clipboard-list": ":material/checklist:",
    "edit-3": ":material/edit_note:",
    "lightbulb": ":material/lightbulb:",
    "git-branch": ":material/account_tree:"
}


def get_streamlit_icon(template_id: str) -> str:
    """Get Streamlit-compatible icon for template.

    Args:
        template_id: Template identifier

    Returns:
        Streamlit material icon string
    """
    template = get_template(template_id)
    icon_key = template.get('icon', 'lightbulb')
    return ICON_MAP.get(icon_key, ":material/lightbulb:")
