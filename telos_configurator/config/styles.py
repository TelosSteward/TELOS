"""
TELOS Configurator - Brand & Style System
==========================================

This module codifies the TELOS visual language for the Corpus Configurator MVP.
All styling is extracted from TELOS Beta to maintain brand consistency.

Color System:
- Primary: Gold (#F4D03F) - TELOS signature color
- Status: Green/Yellow/Orange/Red gradient for alignment states
- Backgrounds: Dark theme with layered surfaces
- Text: High contrast hierarchy for readability

Design Pattern:
- Glassmorphism: Frosted glass effect with subtle gradients
- Cards: Elevated surfaces with gold borders
- Buttons: Primary (gold), secondary (muted), danger (red)
- Inputs: Dark backgrounds with gold accents

Usage:
    from config.styles import inject_custom_css, get_glassmorphism_css

    # In Streamlit app
    inject_custom_css()

    # For custom components
    st.markdown(get_card_css(), unsafe_allow_html=True)
"""

# ============================================================================
# COLOR PALETTE
# ============================================================================

# Primary Brand
GOLD = '#F4D03F'
GOLD_DARK = '#D4B136'
GOLD_LIGHT = '#F6DC6F'

# Status Colors (Alignment States)
STATUS_GOOD = '#27ae60'      # Green - Aligned
STATUS_MILD = '#f39c12'      # Yellow - Minor Drift
STATUS_MODERATE = '#e67e22'  # Orange - Drift Detected
STATUS_SEVERE = '#e74c3c'    # Red - Significant Drift

# Background Layers (Dark Theme)
BG_BASE = '#0d0d0d'          # Darkest base
BG_SURFACE = '#1a1a1a'       # Primary surface
BG_ELEVATED = '#2d2d2d'      # Cards/elevated
BG_OVERLAY = '#3a3a3a'       # Modals

# Text Hierarchy
TEXT_PRIMARY = '#e0e0e0'     # High contrast
TEXT_SECONDARY = '#b0b0b0'   # Medium contrast
TEXT_MUTED = '#808080'       # Low contrast
TEXT_DISABLED = '#4a4a4a'    # Disabled state

# Tier Colors
TIER_1 = '#e74c3c'           # Red - Mission Critical
TIER_2 = '#f39c12'           # Yellow - Important
TIER_3 = '#3498db'           # Blue - Standard

# State Colors
STATE_ACTIVE = '#27ae60'     # Green
STATE_INACTIVE = '#95a5a6'   # Gray
STATE_PENDING = '#f39c12'    # Yellow
STATE_ERROR = '#e74c3c'      # Red

# Accent Colors
ACCENT_BLUE = '#3498db'
ACCENT_PURPLE = '#9b59b6'
ACCENT_CYAN = '#1abc9c'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def with_opacity(color: str, opacity: float) -> str:
    """
    Convert hex color to rgba with specified opacity.

    Args:
        color: Hex color string (e.g., '#F4D03F')
        opacity: Float between 0.0 and 1.0

    Returns:
        RGBA color string (e.g., 'rgba(244, 208, 63, 0.5)')

    Example:
        >>> with_opacity('#F4D03F', 0.5)
        'rgba(244, 208, 63, 0.5)'
    """
    color = color.lstrip('#')
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {opacity})'


def get_tier_color(tier: int) -> str:
    """
    Get color for taxonomy tier (1, 2, or 3).

    Args:
        tier: Integer 1, 2, or 3

    Returns:
        Hex color string

    Example:
        >>> get_tier_color(1)
        '#e74c3c'
    """
    tier_map = {
        1: TIER_1,
        2: TIER_2,
        3: TIER_3
    }
    return tier_map.get(tier, TIER_3)


def get_status_color(status: str) -> str:
    """
    Get color for status state.

    Args:
        status: One of 'active', 'inactive', 'pending', 'error'

    Returns:
        Hex color string

    Example:
        >>> get_status_color('active')
        '#27ae60'
    """
    status_map = {
        'active': STATE_ACTIVE,
        'inactive': STATE_INACTIVE,
        'pending': STATE_PENDING,
        'error': STATE_ERROR,
        'good': STATUS_GOOD,
        'mild': STATUS_MILD,
        'moderate': STATUS_MODERATE,
        'severe': STATUS_SEVERE
    }
    return status_map.get(status.lower(), TEXT_MUTED)


def get_alignment_color(score: float) -> str:
    """
    Get color based on alignment score (0-100).

    Args:
        score: Float between 0 and 100

    Returns:
        Hex color string

    Example:
        >>> get_alignment_color(95.0)
        '#27ae60'
    """
    if score >= 90:
        return STATUS_GOOD
    elif score >= 75:
        return STATUS_MILD
    elif score >= 60:
        return STATUS_MODERATE
    else:
        return STATUS_SEVERE


# ============================================================================
# CSS COMPONENT GENERATORS
# ============================================================================

def get_glassmorphism_css(border_color: str = GOLD) -> str:
    """
    Generate glassmorphism effect CSS matching TELOS Beta.

    Creates frosted glass appearance with:
    - Subtle gradient overlay
    - Backdrop blur filter
    - Colored border with glow
    - Layered shadows (outer glow + depth + inner highlight)

    Args:
        border_color: Hex color for border (default: GOLD)

    Returns:
        CSS style string for use in st.markdown()

    Example:
        >>> css = get_glassmorphism_css()
        >>> st.markdown(f'<div style="{css}">Content</div>', unsafe_allow_html=True)
    """
    return f"""
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid {border_color};
        border-radius: 8px;
        box-shadow: 0 0 15px {with_opacity(border_color, 0.15)},
                    0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
    """.strip()


def get_card_css(border_color: str = GOLD, padding: str = '1.5rem') -> str:
    """
    Generate card container CSS with glassmorphism effect.

    Args:
        border_color: Hex color for border (default: GOLD)
        padding: CSS padding value (default: '1.5rem')

    Returns:
        Complete CSS for card containers

    Example:
        >>> css = get_card_css()
        >>> st.markdown(f'<div style="{css}">Card Content</div>', unsafe_allow_html=True)
    """
    glass = get_glassmorphism_css(border_color)
    return f"""
        {glass}
        padding: {padding};
        margin-bottom: 1rem;
    """.strip()


def get_button_css(variant: str = 'primary') -> str:
    """
    Generate button CSS for different variants.

    Variants:
    - primary: Gold background, dark text
    - secondary: Transparent with gold border
    - danger: Red background, light text
    - success: Green background, light text

    Args:
        variant: Button style variant

    Returns:
        CSS style string for buttons

    Example:
        >>> css = get_button_css('primary')
        >>> st.markdown(f'<button style="{css}">Click Me</button>', unsafe_allow_html=True)
    """
    base_css = """
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        font-size: 0.95rem;
    """

    variant_styles = {
        'primary': f"""
            {base_css}
            background: linear-gradient(135deg, {GOLD} 0%, {GOLD_DARK} 100%);
            color: {BG_BASE};
            box-shadow: 0 4px 12px {with_opacity(GOLD, 0.3)};
        """,
        'secondary': f"""
            {base_css}
            background: transparent;
            color: {GOLD};
            border: 1px solid {GOLD};
            box-shadow: 0 0 10px {with_opacity(GOLD, 0.1)};
        """,
        'danger': f"""
            {base_css}
            background: linear-gradient(135deg, {STATUS_SEVERE} 0%, #c0392b 100%);
            color: {TEXT_PRIMARY};
            box-shadow: 0 4px 12px {with_opacity(STATUS_SEVERE, 0.3)};
        """,
        'success': f"""
            {base_css}
            background: linear-gradient(135deg, {STATUS_GOOD} 0%, #229954 100%);
            color: {TEXT_PRIMARY};
            box-shadow: 0 4px 12px {with_opacity(STATUS_GOOD, 0.3)};
        """
    }

    return variant_styles.get(variant, variant_styles['primary']).strip()


def get_input_css() -> str:
    """
    Generate input field CSS with TELOS styling.

    Returns:
        CSS style string for text inputs

    Example:
        >>> css = get_input_css()
        >>> st.markdown(f'<input style="{css}" type="text" />', unsafe_allow_html=True)
    """
    return f"""
        background: {BG_ELEVATED};
        border: 1px solid {with_opacity(GOLD, 0.3)};
        border-radius: 6px;
        padding: 0.75rem 1rem;
        color: {TEXT_PRIMARY};
        font-size: 0.95rem;
        transition: all 0.3s ease;
    """.strip()


def get_slider_css() -> str:
    """
    Generate slider CSS with TELOS gold accent.

    Returns:
        CSS style string for range sliders
    """
    return f"""
        accent-color: {GOLD};
        background: {BG_ELEVATED};
        border-radius: 10px;
    """.strip()


def get_status_badge_css(status: str) -> str:
    """
    Generate status badge CSS with appropriate color.

    Args:
        status: Status type (good/mild/moderate/severe/active/inactive/etc.)

    Returns:
        CSS style string for status badges

    Example:
        >>> css = get_status_badge_css('good')
        >>> st.markdown(f'<span style="{css}">Aligned</span>', unsafe_allow_html=True)
    """
    color = get_status_color(status)
    return f"""
        background: {with_opacity(color, 0.15)};
        color: {color};
        border: 1px solid {with_opacity(color, 0.5)};
        border-radius: 12px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    """.strip()


def get_metric_card_css(tier: int = 3) -> str:
    """
    Generate metric card CSS with tier-based border color.

    Args:
        tier: Taxonomy tier (1, 2, or 3)

    Returns:
        CSS style string for metric cards

    Example:
        >>> css = get_metric_card_css(tier=1)
        >>> st.markdown(f'<div style="{css}">Metric: 95%</div>', unsafe_allow_html=True)
    """
    tier_color = get_tier_color(tier)
    return get_card_css(border_color=tier_color, padding='1rem')


# ============================================================================
# FULL PAGE CSS INJECTION
# ============================================================================

def inject_custom_css():
    """
    Inject comprehensive TELOS custom CSS into Streamlit app.

    This function should be called once at the start of your Streamlit app
    to apply TELOS branding globally, including the glassmorphism background effect.

    Styles applied:
    - Glassmorphism background with gold radial gradients
    - Global color scheme (dark theme)
    - Typography (font hierarchy)
    - Streamlit component overrides
    - Custom component classes
    - Responsive utilities

    Usage:
        import streamlit as st
        from config.styles import inject_custom_css

        inject_custom_css()
        st.title("TELOS Configurator")
    """
    import streamlit as st
    import streamlit.components.v1 as components

    css = f"""
    <style>
        /* ============================================
           GLOBAL VARIABLES
           ============================================ */
        :root {{
            --gold: {GOLD};
            --gold-dark: {GOLD_DARK};
            --gold-light: {GOLD_LIGHT};

            --status-good: {STATUS_GOOD};
            --status-mild: {STATUS_MILD};
            --status-moderate: {STATUS_MODERATE};
            --status-severe: {STATUS_SEVERE};

            --bg-base: {BG_BASE};
            --bg-surface: {BG_SURFACE};
            --bg-elevated: {BG_ELEVATED};
            --bg-overlay: {BG_OVERLAY};

            --text-primary: {TEXT_PRIMARY};
            --text-secondary: {TEXT_SECONDARY};
            --text-muted: {TEXT_MUTED};
        }}

        /* ============================================
           GLASSMORPHISM BACKGROUND - TELOS SIGNATURE
           ============================================ */
        .stApp {{
            background:
                radial-gradient(ellipse 90% 90% at 15% 5%, rgba(244, 208, 63, 0.7) 0%, transparent 40%),
                radial-gradient(ellipse 80% 80% at 90% 95%, rgba(200, 160, 40, 0.6) 0%, transparent 35%),
                radial-gradient(ellipse 60% 60% at 50% 50%, rgba(244, 180, 63, 0.4) 0%, transparent 50%),
                linear-gradient(135deg, #1a1510 0%, #0d0a05 50%, #0a0805 100%) !important;
            background-attachment: fixed !important;
            color: {TEXT_PRIMARY};
        }}

        /* Make containers transparent so background shows through */
        [data-testid="stAppViewContainer"],
        .main,
        section.main {{
            background: transparent !important;
            background-color: transparent !important;
        }}

        /* Main block container - transparent */
        .block-container {{
            background: transparent !important;
            max-width: 1100px !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {BG_SURFACE};
            border-right: 1px solid {with_opacity(GOLD, 0.2)};
        }}

        /* ============================================
           TYPOGRAPHY
           ============================================ */
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_PRIMARY};
            font-weight: 700;
        }}

        h1 {{
            color: {GOLD};
            text-shadow: 0 0 20px {with_opacity(GOLD, 0.3)};
        }}

        p {{
            color: {TEXT_SECONDARY};
            line-height: 1.6;
        }}

        /* ============================================
           STREAMLIT COMPONENT OVERRIDES
           ============================================ */

        /* Buttons - Muted gold with dark text for readability */
        .stButton > button {{
            background: {BG_ELEVATED};
            color: {GOLD};
            border: 1px solid {with_opacity(GOLD, 0.5)};
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 0 10px {with_opacity(GOLD, 0.1)};
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            background: {with_opacity(GOLD, 0.15)};
            border-color: {GOLD};
            box-shadow: 0 0 15px {with_opacity(GOLD, 0.2)};
        }}

        /* Primary buttons - slightly more prominent but not blinding */
        .stButton > button[kind="primary"] {{
            background: {with_opacity(GOLD, 0.2)};
            color: {GOLD};
            border: 1px solid {GOLD};
        }}

        .stButton > button[kind="primary"]:hover {{
            background: {with_opacity(GOLD, 0.3)};
        }}

        /* Text Inputs */
        .stTextInput > div > div > input {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            border: 1px solid {with_opacity(GOLD, 0.3)};
            border-radius: 6px;
            transition: all 0.3s ease;
        }}

        .stTextInput > div > div > input:focus {{
            border-color: {GOLD};
            box-shadow: 0 0 10px {with_opacity(GOLD, 0.2)};
        }}

        /* Text Areas */
        .stTextArea > div > div > textarea {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            border: 1px solid {with_opacity(GOLD, 0.3)};
            border-radius: 6px;
            transition: all 0.3s ease;
        }}

        .stTextArea > div > div > textarea:focus {{
            border-color: {GOLD};
            box-shadow: 0 0 10px {with_opacity(GOLD, 0.2)};
        }}

        /* Number Inputs */
        .stNumberInput > div > div > input {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            border: 1px solid {with_opacity(GOLD, 0.3)};
            border-radius: 6px;
        }}

        /* Select Boxes */
        .stSelectbox > div > div > div {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            border: 1px solid {with_opacity(GOLD, 0.3)};
            border-radius: 6px;
        }}

        /* Sliders */
        .stSlider > div > div > div > div {{
            background-color: {GOLD};
        }}

        .stSlider > div > div > div {{
            background-color: {BG_ELEVATED};
        }}

        /* Checkboxes */
        .stCheckbox > label > div {{
            background-color: {BG_ELEVATED};
            border: 1px solid {with_opacity(GOLD, 0.3)};
        }}

        .stCheckbox > label > div[data-checked="true"] {{
            background-color: {GOLD};
            border-color: {GOLD};
        }}

        /* Radio Buttons */
        .stRadio > label {{
            color: {TEXT_SECONDARY};
        }}

        /* Dataframes */
        .stDataFrame {{
            background-color: {BG_ELEVATED};
            border: 1px solid {with_opacity(GOLD, 0.2)};
            border-radius: 8px;
        }}

        /* Metrics */
        div[data-testid="stMetricValue"] {{
            color: {GOLD};
            font-size: 2rem;
            font-weight: 700;
        }}

        div[data-testid="stMetricLabel"] {{
            color: {TEXT_SECONDARY};
            font-size: 0.9rem;
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {BG_ELEVATED};
            color: {TEXT_PRIMARY};
            border: 1px solid {with_opacity(GOLD, 0.2)};
            border-radius: 6px;
        }}

        .streamlit-expanderHeader:hover {{
            border-color: {GOLD};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {BG_SURFACE};
            border-radius: 6px;
            padding: 0.5rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: {TEXT_SECONDARY};
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {BG_ELEVATED};
            color: {GOLD};
            border: 1px solid {with_opacity(GOLD, 0.3)};
        }}

        /* ============================================
           CUSTOM COMPONENT CLASSES
           ============================================ */

        /* Glassmorphism Card */
        .glass-card {{
            {get_glassmorphism_css()}
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}

        /* Status Badges */
        .badge-good {{
            {get_status_badge_css('good')}
        }}

        .badge-mild {{
            {get_status_badge_css('mild')}
        }}

        .badge-moderate {{
            {get_status_badge_css('moderate')}
        }}

        .badge-severe {{
            {get_status_badge_css('severe')}
        }}

        /* Tier Badges */
        .tier-1 {{
            color: {TIER_1};
            font-weight: 700;
        }}

        .tier-2 {{
            color: {TIER_2};
            font-weight: 700;
        }}

        .tier-3 {{
            color: {TIER_3};
            font-weight: 700;
        }}

        /* Section Headers */
        .section-header {{
            color: {GOLD};
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {with_opacity(GOLD, 0.3)};
        }}

        /* Info Boxes */
        .info-box {{
            background: {with_opacity(ACCENT_BLUE, 0.1)};
            border-left: 4px solid {ACCENT_BLUE};
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}

        .warning-box {{
            background: {with_opacity(STATUS_MODERATE, 0.1)};
            border-left: 4px solid {STATUS_MODERATE};
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}

        .error-box {{
            background: {with_opacity(STATUS_SEVERE, 0.1)};
            border-left: 4px solid {STATUS_SEVERE};
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}

        .success-box {{
            background: {with_opacity(STATUS_GOOD, 0.1)};
            border-left: 4px solid {STATUS_GOOD};
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}

        /* ============================================
           UTILITY CLASSES
           ============================================ */

        .text-gold {{
            color: {GOLD};
        }}

        .text-muted {{
            color: {TEXT_MUTED};
        }}

        .text-center {{
            text-align: center;
        }}

        .mt-1 {{
            margin-top: 0.5rem;
        }}

        .mt-2 {{
            margin-top: 1rem;
        }}

        .mb-1 {{
            margin-bottom: 0.5rem;
        }}

        .mb-2 {{
            margin-bottom: 1rem;
        }}

        /* ============================================
           ANIMATIONS
           ============================================ */

        @keyframes glow {{
            0%, 100% {{
                box-shadow: 0 0 15px {with_opacity(GOLD, 0.15)};
            }}
            50% {{
                box-shadow: 0 0 25px {with_opacity(GOLD, 0.3)};
            }}
        }}

        .glow-animation {{
            animation: glow 2s ease-in-out infinite;
        }}

        /* ============================================
           SCROLLBARS
           ============================================ */

        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}

        ::-webkit-scrollbar-track {{
            background: {BG_SURFACE};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {with_opacity(GOLD, 0.5)};
            border-radius: 5px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {GOLD};
        }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    # JavaScript injection to force glassmorphism (same approach as TELOS Beta main.py)
    components.html("""
    <script>
    (function() {
        var doc = window.parent.document;
        var gradientBg = `
            radial-gradient(ellipse 90% 90% at 15% 5%, rgba(244, 208, 63, 0.7) 0%, transparent 40%),
            radial-gradient(ellipse 80% 80% at 90% 95%, rgba(200, 160, 40, 0.6) 0%, transparent 35%),
            radial-gradient(ellipse 60% 60% at 50% 50%, rgba(244, 180, 63, 0.4) 0%, transparent 50%),
            linear-gradient(135deg, #1a1510 0%, #0d0a05 50%, #0a0805 100%)
        `;
        function applyGlassmorphism() {
            // Apply gradient DIRECTLY to .stApp
            var stApp = doc.querySelector('.stApp');
            if (stApp) {
                stApp.style.setProperty('background', gradientBg, 'important');
                stApp.style.setProperty('background-attachment', 'fixed', 'important');
            }
            // Only make the top-level containers transparent
            var topContainers = doc.querySelectorAll(`
                [data-testid="stAppViewContainer"],
                .main,
                section.main
            `);
            topContainers.forEach(function(el) {
                el.style.setProperty('background', 'transparent', 'important');
                el.style.setProperty('background-color', 'transparent', 'important');
            });
            // Sidebar styling - dark with subtle transparency
            var sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.setProperty('background-color', '#2a2a2a', 'important');
            }
        }
        // Apply immediately
        applyGlassmorphism();
        // Re-apply on DOM changes (Streamlit dynamic updates)
        var observer = new MutationObserver(function() {
            applyGlassmorphism();
        });
        observer.observe(doc.body, { childList: true, subtree: true });
        console.log('TELOS CONFIGURATOR: Glassmorphism styling applied');
    })();
    </script>
    """, height=0)


# ============================================================================
# MARKDOWN HELPERS
# ============================================================================

def render_section_header(title: str) -> str:
    """
    Render a styled section header.

    Args:
        title: Section title text

    Returns:
        HTML string for section header

    Example:
        >>> st.markdown(render_section_header("Configuration"), unsafe_allow_html=True)
    """
    return f'<div class="section-header">{title}</div>'


def render_status_badge(status: str, label: str) -> str:
    """
    Render a status badge with appropriate styling.

    Args:
        status: Status type (good/mild/moderate/severe)
        label: Badge text

    Returns:
        HTML string for status badge

    Example:
        >>> st.markdown(render_status_badge('good', 'Aligned'), unsafe_allow_html=True)
    """
    css = get_status_badge_css(status)
    return f'<span style="{css}">{label}</span>'


def render_tier_badge(tier: int) -> str:
    """
    Render a tier badge with appropriate color.

    Args:
        tier: Tier number (1, 2, or 3)

    Returns:
        HTML string for tier badge

    Example:
        >>> st.markdown(render_tier_badge(1), unsafe_allow_html=True)
    """
    return f'<span class="tier-{tier}">Tier {tier}</span>'


def render_info_box(message: str, box_type: str = 'info') -> str:
    """
    Render an info/warning/error/success box.

    Args:
        message: Message text
        box_type: One of 'info', 'warning', 'error', 'success'

    Returns:
        HTML string for info box

    Example:
        >>> st.markdown(render_info_box('Configuration saved', 'success'), unsafe_allow_html=True)
    """
    return f'<div class="{box_type}-box">{message}</div>'


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Colors
    'GOLD', 'GOLD_DARK', 'GOLD_LIGHT',
    'STATUS_GOOD', 'STATUS_MILD', 'STATUS_MODERATE', 'STATUS_SEVERE',
    'BG_BASE', 'BG_SURFACE', 'BG_ELEVATED', 'BG_OVERLAY',
    'TEXT_PRIMARY', 'TEXT_SECONDARY', 'TEXT_MUTED', 'TEXT_DISABLED',
    'TIER_1', 'TIER_2', 'TIER_3',
    'STATE_ACTIVE', 'STATE_INACTIVE', 'STATE_PENDING', 'STATE_ERROR',

    # Helpers
    'with_opacity', 'get_tier_color', 'get_status_color', 'get_alignment_color',

    # CSS Generators
    'get_glassmorphism_css', 'get_card_css', 'get_button_css',
    'get_input_css', 'get_slider_css', 'get_status_badge_css', 'get_metric_card_css',

    # Page Injection
    'inject_custom_css',

    # Markdown Helpers
    'render_section_header', 'render_status_badge', 'render_tier_badge', 'render_info_box'
]
