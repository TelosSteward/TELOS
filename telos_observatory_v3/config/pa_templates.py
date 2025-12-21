"""
PA Templates Configuration
==========================

Pre-defined Primacy Attractor templates for quick session setup.
Users can click to select a template instead of typing their purpose.

These templates demonstrate semantic understanding - TELOS tracks meaning
and conceptual relationships, not keywords. Each template establishes
a clear attractor in embedding space.

Template Ordering: Based on objective LLM usage data (2024-2025):
- Writing tasks: 42% of work-related ChatGPT messages
- Learning/Education: Heavy Claude/ChatGPT education use
- Information/Research: 78% combined (practical guidance + info seeking)
- Coding: 39% of Claude usage
- Planning: Part of "practical guidance" category
- Analysis/Summarization: 41% of enterprise users
- Brainstorming: Creative problem-solving is major use
- Debugging: Subset of coding tasks (least frequent as standalone)

NOTE: Domain-specific templates (like "Explore TELOS") require RAG corpus
and will be added in a future version with Steward integration.
"""

# Click-to-select PA templates
# Human-facing language: grounded, clear, unambiguous reference points
# No "I" statements. Direct statements about what the conversation accomplishes.
#
# CRITICAL: Purpose statements should START with intent verbs that map to INTENT_TO_ROLE_MAP
# in beta_dual_attractor.py. This enables proper AI PA derivation.
# Example: "Write content..." → AI role = "help write"
#          "Learn and understand..." → AI role = "teach"
#
# DESIGN PRINCIPLE: Each template establishes a complete Primacy Attractor with:
# - PURPOSE: The core intent that governs what this conversation accomplishes
# - SCOPE: Specific domains and activities that ARE within the conversation's territory
# - BOUNDARIES: Clear limits on what falls OUTSIDE the conversation's purpose
#
# These work together to create a semantic basin in embedding space that captures
# legitimate variations while discriminating against off-topic drift.
PA_TEMPLATES = [
    {
        "id": "creative_writing",
        "icon": "✍️",
        "title": "Creative Writing",
        "short_desc": "Write stories, content, or copy",
        # INTENT: 'write' → AI role: 'help write'
        "purpose": "Write and craft compelling written content that achieves a specific communicative goal - whether that means engaging readers emotionally, conveying complex information accessibly, persuading an audience to action, or establishing a distinctive voice and brand presence",
        "scope": [
            "Drafting original content from outlines, briefs, or rough ideas",
            "Voice, tone, and style calibration matched to target audience",
            "Structural decisions: openings, transitions, pacing, and closings",
            "Word choice, sentence rhythm, and rhetorical techniques",
            "Revision and editing to strengthen weak passages",
            "Format adaptation across mediums: blogs, emails, scripts, social posts",
            "Feedback on drafts with specific improvement suggestions"
        ],
        "boundaries": [
            "Factual research and source verification belong to Research mode",
            "Technical documentation with precise specifications belongs to Code/Technical mode",
            "Strategic planning of content calendars belongs to Planning mode",
            "Analysis of existing content performance belongs to Review mode"
        ],
        "style": "Collaborative drafting partner focused on craft and effect",
        "category": "creative",
        "example_queries": [
            "Help me write an opening paragraph for my blog post",
            "I need to draft a persuasive email to my team",
            "Can you help me write a product description?",
            "I'm working on a short story and need help with dialogue",
            "Write a compelling headline for this article",
            "Help me revise this paragraph to be more engaging",
            "I need copy for a landing page that converts",
            "Can you help me write a professional bio?",
            "Draft a creative social media caption",
            "Help me improve the flow of this essay",
        ]
    },
    {
        "id": "learn_concept",
        "icon": "📚",
        "title": "Learn a Concept",
        "short_desc": "Understand something new",
        # INTENT: 'learn' → AI role: 'teach'
        "purpose": "Learn and deeply understand a concept or domain - building robust mental models that connect new ideas to existing knowledge, developing intuition through multiple angles of explanation, and achieving the kind of understanding that enables confident application and teaching to others",
        "scope": [
            "Breaking down complex ideas into digestible components",
            "Core principles and foundational concepts explained clearly",
            "Multiple explanatory approaches: analogies, metaphors, visualizations",
            "Concrete examples showing abstract concepts in real-world action",
            "Common misconceptions, pitfalls, and how to avoid them",
            "Progressive depth: building from basics to nuance",
            "Checking understanding through questions and scenarios",
            "Connections between this concept and related ideas"
        ],
        "boundaries": [
            "Implementing or building things belongs to Code/Build mode",
            "Comprehensive literature review belongs to Research mode",
            "Evaluating or critiquing existing work belongs to Review mode",
            "Planning how to apply knowledge belongs to Planning mode"
        ],
        "style": "Patient teacher adapting to your current understanding",
        "category": "learning",
        "example_queries": [
            "Explain how machine learning works in simple terms",
            "I want to understand quantum computing basics",
            "Help me grasp the concept of recursion",
            "What is the difference between REST and GraphQL?",
            "Explain blockchain like I'm a beginner",
            "I don't understand why we need async/await",
            "Can you break down how neural networks learn?",
            "What's the intuition behind derivatives in calculus?",
            "Help me understand how encryption works",
            "Explain object-oriented programming concepts",
        ]
    },
    {
        "id": "research_topic",
        "icon": "🔍",
        "title": "Research Topic",
        "short_desc": "Investigate and synthesize information",
        # INTENT: 'research' → AI role: 'help research'
        "purpose": "Research and investigate a topic systematically - gathering diverse perspectives, critically evaluating evidence quality, identifying areas of consensus and controversy, and synthesizing findings into clear, actionable understanding that supports informed decision-making",
        "scope": [
            "Identifying key questions and sub-questions to explore",
            "Gathering information from multiple angles and sources",
            "Distinguishing well-supported conclusions from speculation",
            "Evaluating evidence quality, methodology, and limitations",
            "Identifying areas of scientific consensus vs. ongoing debate",
            "Synthesizing disparate sources into coherent understanding",
            "Acknowledging uncertainty and knowledge gaps honestly",
            "Organizing findings in useful structures and summaries"
        ],
        "boundaries": [
            "Learning concepts in depth belongs to Learn mode - research surveys breadth",
            "Writing up findings as polished content belongs to Writing mode",
            "Making decisions or plans based on research belongs to Planning mode",
            "Advocating for particular positions or outcomes is outside scope"
        ],
        "style": "Thorough investigator balancing breadth with critical depth",
        "category": "research",
        "example_queries": [
            "What does the research say about remote work productivity?",
            "Help me understand the evidence on climate change",
            "What are the different perspectives on AI alignment?",
            "Summarize the key findings on sleep and memory",
            "What do studies show about meditation benefits?",
            "I need to research market trends in renewable energy",
            "What's the current state of research on longevity?",
            "Help me find evidence about the effectiveness of CBT",
            "What are the pros and cons of different database systems?",
            "Synthesize the research on team collaboration tools",
        ]
    },
    {
        "id": "write_code",
        "icon": "💻",
        "title": "Write Code",
        "short_desc": "Build a feature or project",
        # INTENT: 'build' → AI role: 'help build'
        "purpose": "Build and implement software solutions that solve specific problems - translating requirements into working code, making sound architectural decisions, following best practices for the language and framework, and producing code that is functional, readable, testable, and maintainable by future developers",
        "scope": [
            "Translating requirements and user stories into technical implementation",
            "Architectural decisions: patterns, structure, and organization",
            "Writing clean, idiomatic code for the target language/framework",
            "Implementing features with appropriate error handling",
            "Writing tests: unit, integration, and edge cases",
            "Code organization, naming conventions, and documentation",
            "Performance considerations and optimization where needed",
            "Integration with existing codebase and systems"
        ],
        "boundaries": [
            "Fixing bugs in existing code belongs to Debug mode",
            "Reviewing or critiquing code quality belongs to Review mode",
            "Learning programming concepts belongs to Learn mode",
            "Planning project milestones and timelines belongs to Planning mode"
        ],
        "style": "Practical implementation partner with clear reasoning",
        "category": "development",
        "example_queries": [
            "Write a function to validate email addresses in Python",
            "Help me build a REST API endpoint for user authentication",
            "Create a React component for a data table",
            "I need to implement a caching layer for my application",
            "Write a SQL query to get user activity by month",
            "Help me set up a CI/CD pipeline for my project",
            "Implement a search feature with autocomplete",
            "Create a database schema for an e-commerce site",
            "Write unit tests for this service class",
            "Build a file upload handler with validation",
        ]
    },
    {
        "id": "plan_project",
        "icon": "📋",
        "title": "Plan a Project",
        "short_desc": "Organize and strategize",
        # INTENT: 'plan' → AI role: 'help plan'
        "purpose": "Plan and organize complex goals into actionable execution strategies - breaking down objectives into concrete tasks, mapping dependencies and critical paths, estimating effort realistically, identifying risks and contingencies, and creating structures that enable confident forward progress",
        "scope": [
            "Breaking high-level objectives into specific, measurable tasks",
            "Defining clear completion criteria for each milestone",
            "Mapping task dependencies and identifying the critical path",
            "Realistic effort estimation based on complexity and unknowns",
            "Resource allocation and capacity planning",
            "Risk identification and contingency strategies",
            "Prioritization frameworks: what matters most and why",
            "Timeline construction with appropriate buffers"
        ],
        "boundaries": [
            "Actually executing tasks belongs to their respective modes (Code, Write, etc.)",
            "Deep research to inform plans belongs to Research mode",
            "Learning skills needed for execution belongs to Learn mode",
            "Reviewing completed work against plan belongs to Review mode"
        ],
        "style": "Strategic thinking partner focused on actionable clarity",
        "category": "planning",
        "example_queries": [
            "Help me create a project timeline for launching a new feature",
            "Break down this goal into actionable tasks",
            "What are the key milestones for a mobile app launch?",
            "Create a sprint plan for my development team",
            "Help me prioritize these project tasks",
            "What dependencies should I consider for this migration?",
            "Build a roadmap for Q1 product development",
            "Help me estimate effort for these features",
            "What's a good structure for a 3-month project plan?",
            "Identify risks and contingencies for this project",
        ]
    },
    {
        "id": "review_analyze",
        "icon": "🔎",
        "title": "Review & Analyze",
        "short_desc": "Get feedback on work",
        # INTENT: 'review' → AI role: 'help review'
        "purpose": "Review and critically analyze work to surface strengths, weaknesses, and opportunities for improvement - providing specific, evidence-based feedback that identifies patterns, prioritizes issues by impact, and offers concrete actionable guidance for making the work better",
        "scope": [
            "Evaluating how well work achieves its stated goals",
            "Identifying specific strengths to preserve and build on",
            "Pinpointing weaknesses with clear evidence and examples",
            "Pattern recognition across multiple aspects of the work",
            "Prioritizing issues by impact: what matters most to fix",
            "Providing specific, actionable improvement suggestions",
            "Comparing against relevant standards or best practices",
            "Offering alternative approaches where appropriate"
        ],
        "boundaries": [
            "Implementing the suggested improvements belongs to their respective modes",
            "Learning about concepts mentioned in review belongs to Learn mode",
            "Researching best practices in depth belongs to Research mode",
            "Planning how to address feedback belongs to Planning mode"
        ],
        "style": "Honest evaluator balancing critique with constructive direction",
        "category": "review",
        "example_queries": [
            "Review this code and suggest improvements",
            "What are the weaknesses in this business proposal?",
            "Analyze my resume and give feedback",
            "Review this email before I send it",
            "What's wrong with this SQL query?",
            "Critique my presentation structure",
            "Analyze the performance of this algorithm",
            "Review my article draft for clarity",
            "What are the gaps in this project plan?",
            "Give feedback on my portfolio website design",
        ]
    },
    {
        "id": "brainstorm_ideas",
        "icon": "💡",
        "title": "Brainstorm Ideas",
        "short_desc": "Generate and explore possibilities",
        # INTENT: 'brainstorm' → AI role: 'help brainstorm'
        "purpose": "Brainstorm and generate creative options that go beyond the obvious - exploring diverse possibilities, making unexpected cross-domain connections, suspending judgment to allow wild ideas, then progressively evaluating and refining promising directions toward concrete action",
        "scope": [
            "Generating quantity of ideas before filtering for quality",
            "Exploring possibilities without premature judgment",
            "Drawing inspiration from different domains and disciplines",
            "Making unexpected connections and combinations",
            "Challenging assumptions that limit the solution space",
            "Expanding on promising seeds into fuller concepts",
            "Evaluating ideas against criteria when ready",
            "Refining winners into actionable next steps"
        ],
        "boundaries": [
            "Deep research on specific ideas belongs to Research mode",
            "Planning implementation of chosen ideas belongs to Planning mode",
            "Actually building or creating belongs to Code/Writing modes",
            "Evaluating existing work belongs to Review mode"
        ],
        "style": "Energetic creative partner: divergent exploration then convergent synthesis",
        "category": "creative",
        "example_queries": [
            "Help me brainstorm names for my startup",
            "What are some creative solutions to this problem?",
            "Generate ideas for improving user engagement",
            "I need fresh angles for this marketing campaign",
            "Help me think of side project ideas",
            "What are different ways to approach this challenge?",
            "Brainstorm features for my new app",
            "Help me come up with content ideas for my blog",
            "What unique approaches could we try?",
            "Generate alternatives to our current strategy",
        ]
    },
    {
        "id": "debug_code",
        "icon": "🐛",
        "title": "Debug Code",
        "short_desc": "Fix a bug or error",
        # INTENT: 'debug' → AI role: 'help debug'
        "purpose": "Debug and diagnose software defects systematically - tracing the causal chain from observed symptoms back to root cause, understanding not just what broke but why, evaluating potential fixes for correctness and side effects, and implementing solutions that resolve the issue without introducing new problems",
        "scope": [
            "Analyzing error messages, stack traces, and symptoms",
            "Forming and testing hypotheses about root cause",
            "Tracing execution flow to locate the defect",
            "Understanding the intended vs. actual behavior",
            "Evaluating multiple fix approaches for robustness",
            "Implementing fixes that address root cause, not just symptoms",
            "Verifying fixes don't introduce regressions",
            "Adding defensive code to prevent similar issues"
        ],
        "boundaries": [
            "Building new features belongs to Code/Build mode",
            "Broad refactoring beyond the bug scope belongs to Review then Code mode",
            "Learning concepts needed to understand the bug belongs to Learn mode",
            "Optimizing performance without a specific bug belongs to Review mode"
        ],
        "style": "Methodical diagnostician: systematic investigation to verified fix",
        "category": "development",
        "example_queries": [
            "I'm getting a TypeError in my Python code",
            "My function returns None when I expect a list",
            "Help me debug this authentication issue",
            "I have a bug in my React component that crashes on render",
            "Why is my API returning 500 errors?",
            "The loop is running infinitely and I can't figure out why",
            "My database query returns empty results but data exists",
            "Getting a null pointer exception in my Java code",
            "My CSS styles aren't applying to the component",
            "The form validation is failing but I don't see why",
        ]
    },
]

# Template categories for organization
TEMPLATE_CATEGORIES = {
    "creative": {"label": "Creative", "color": "#FF9800"},
    "learning": {"label": "Learning", "color": "#27ae60"},
    "research": {"label": "Research", "color": "#9C27B0"},
    "development": {"label": "Development", "color": "#2196F3"},
    "planning": {"label": "Planning", "color": "#00BCD4"},
    "review": {"label": "Review", "color": "#795548"},
}

# =============================================================================
# UNIVERSAL LANE EXPANSION (System's Default Steward PA)
# =============================================================================
# The system's hidden default PA that embodies stewardship, not task-mastery.
#
# Philosophy: "Be a steward, not a taskmaster."
# - A taskmaster rigidly blocks anything outside the exact task definition
# - A steward supports the user's journey, allowing natural exploration
#
# These task-agnostic patterns apply to ALL PAs, expanding the semantic "lane"
# to cover productive conversation patterns that humans naturally use:
# - Meta-questions ("Can you explain that differently?")
# - Productive pivots ("What if we tried this instead?")
# - Progress signals ("That makes sense, now what about...")
# - Clarification requests ("Why is this important?")
#
# This is NEVER user-facing. It's the system's innate flexibility layer that:
# - Prevents false-positive interventions on legitimate exploration
# - Allows natural task-related tangents that serve the user's goal
# - Maintains governance while treating users as intelligent adults
#
# Result: Off-topic content still gets intervention (ORANGE/RED zones),
# but the system guides gently rather than blocking harshly.
# =============================================================================
UNIVERSAL_LANE_EXPANSION = [
    # Meta-questions about the current approach
    "Can you explain that in a different way?",
    "Let me think about this for a moment",
    "What's another way to approach this?",
    "Can you give me an example?",
    "I'm not sure I understand, can you clarify?",

    # Productive pivots within task scope
    "Let me try a different angle",
    "What if we considered this instead?",
    "Can we step back and look at the bigger picture?",
    "Help me understand the context better",
    "What are the tradeoffs here?",

    # Progress and iteration signals
    "That makes sense, let's continue",
    "Good point, now let me ask about...",
    "Building on that, what about...",
    "Let's dig deeper into this aspect",
    "Can we explore this further?",

    # Clarification requests
    "What do you mean by that?",
    "Can you be more specific?",
    "How does this relate to what we discussed?",
    "I need more detail on this part",
    "Why is this important?",
]

# Weight for universal expansion in centroid calculation
# This is the system's "steward mode" - flexible support without being a taskmaster
# Higher weight = more flexibility for natural conversation patterns
# Lower weight = more discriminative against off-topic content (PB&J problem)
# 0.0 = Pure domain-specific (no universal expansion) - maximum discrimination
UNIVERSAL_EXPANSION_WEIGHT = 0.0


def get_template_by_id(template_id: str) -> dict:
    """Get a template by its ID."""
    for template in PA_TEMPLATES:
        if template["id"] == template_id:
            return template
    return None


def get_templates_by_category(category: str) -> list:
    """Get all templates in a category."""
    return [t for t in PA_TEMPLATES if t.get("category") == category]


def template_to_pa(template: dict) -> dict:
    """Convert a template to a full PA structure."""
    return {
        "purpose": [template["purpose"]],
        "scope": template["scope"],
        "boundaries": template["boundaries"],
        "style": template.get("style", "Adaptive"),
        "template_id": template["id"],
        "template_title": template["title"],
    }
