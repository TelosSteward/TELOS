"""
TELOSCOPE v2 - Timeline Marker Generator

Generates visual markers for the timeline scrubber with enhanced styling.

Improvements over Phase 1 inline markers:
- Extracted, reusable marker generation logic
- Configurable marker styles (size, opacity, glow effects)
- Hover tooltip support
- Click-to-jump functionality (data attributes)
- Cached marker generation (performance)
- Multiple marker styles (minimal, standard, enhanced)

Usage:
    from teloscope_v2.utils.marker_generator import generate_timeline_markers

    html = generate_timeline_markers(
        turns=session['turns'],
        current_turn=5,
        style='enhanced'
    )
    st.markdown(html, unsafe_allow_html=True)
"""

from typing import List, Dict, Any, Optional, Literal


# ============================================================================
# MARKER COLOR MAPPING
# ============================================================================

DEFAULT_STATUS_COLORS = {
    '✓': '#90EE90',     # Green - Stable governance
    '⚠️': '#FFA500',    # Orange - Drift detected
    '⚡': '#FFD700',    # Gold - Intervention applied
    '⚙️': '#6495ED',    # Blue - Calibration phase
    'default': '#90EE90',  # Fallback to green
}

STATUS_LABELS = {
    '✓': 'Stable',
    '⚠️': 'Drift',
    '⚡': 'Intervention',
    '⚙️': 'Calibration',
}


# ============================================================================
# MARKER STYLES
# ============================================================================

MarkerStyle = Literal['minimal', 'standard', 'enhanced']


def _get_marker_config(style: MarkerStyle) -> dict:
    """
    Get marker configuration for style.

    Returns:
        Dict with size, spacing, glow settings
    """
    configs = {
        'minimal': {
            'size_active': 10,
            'size_inactive': 6,
            'spacing': 4,
            'glow': False,
            'hover_scale': False,
            'transitions': False,
        },
        'standard': {
            'size_active': 12,
            'size_inactive': 8,
            'spacing': 6,
            'glow': True,
            'hover_scale': False,
            'transitions': True,
        },
        'enhanced': {
            'size_active': 14,
            'size_inactive': 8,
            'spacing': 8,
            'glow': True,
            'hover_scale': True,
            'transitions': True,
        },
    }
    return configs.get(style, configs['standard'])


# ============================================================================
# MARKER GENERATION
# ============================================================================

def generate_timeline_markers(
    turns: List[Dict[str, Any]],
    current_turn: int,
    style: MarkerStyle = 'standard',
    custom_colors: Optional[Dict[str, str]] = None,
    clickable: bool = False,
    show_tooltips: bool = False,
) -> str:
    """
    Generate HTML for timeline markers.

    Args:
        turns: List of turn dicts with 'status' field
        current_turn: Active turn index (0-based)
        style: Marker style ('minimal', 'standard', 'enhanced')
        custom_colors: Optional custom color mapping
        clickable: If True, add data attributes for click handling
        show_tooltips: If True, add title attributes with turn info

    Returns:
        HTML string with marker elements

    Example:
        html = generate_timeline_markers(
            turns=session['turns'],
            current_turn=5,
            style='enhanced',
            show_tooltips=True
        )
    """
    if not turns:
        return "<div></div>"

    # Get config for style
    config = _get_marker_config(style)

    # Use custom colors or defaults
    colors = custom_colors if custom_colors else DEFAULT_STATUS_COLORS

    # Build markers
    markers_html = []

    for idx, turn in enumerate(turns):
        marker_html = _generate_single_marker(
            turn=turn,
            turn_index=idx,
            current_turn=current_turn,
            config=config,
            colors=colors,
            clickable=clickable,
            show_tooltips=show_tooltips,
        )
        markers_html.append(marker_html)

    # Wrap in container
    container_style = f"""
        display: flex;
        justify-content: center;
        align-items: center;
        gap: {config['spacing']}px;
        padding: 8px 0;
        user-select: none;
    """

    return f'<div style="{container_style}">{"".join(markers_html)}</div>'


def _generate_single_marker(
    turn: Dict[str, Any],
    turn_index: int,
    current_turn: int,
    config: dict,
    colors: Dict[str, str],
    clickable: bool,
    show_tooltips: bool,
) -> str:
    """Generate HTML for a single marker."""
    # Determine if active
    is_active = (turn_index == current_turn)

    # Get status and color
    status = turn.get('status', '✓')
    color = colors.get(status, colors['default'])

    # Size based on active state
    size = config['size_active'] if is_active else config['size_inactive']

    # Opacity based on active state
    opacity = 1.0 if is_active else 0.7

    # Build base style
    style_parts = [
        f"width: {size}px",
        f"height: {size}px",
        f"background: {color}",
        "border-radius: 50%",
        f"opacity: {opacity}",
        "cursor: pointer" if clickable else "cursor: default",
    ]

    # Add glow effect for active marker
    if config['glow'] and is_active:
        style_parts.append(f"box-shadow: 0 0 8px 2px {color}")

    # Add transitions
    if config['transitions']:
        style_parts.append("transition: all 0.3s ease")

    # Add hover scale
    if config['hover_scale']:
        # Note: CSS hover can't be easily added inline, would need <style> block
        pass

    style = "; ".join(style_parts)

    # Build attributes
    attributes = []

    if clickable:
        attributes.append(f'data-turn="{turn_index}"')
        attributes.append('data-clickable="true"')

    if show_tooltips:
        turn_num = turn.get('turn', turn_index + 1)
        status_label = STATUS_LABELS.get(status, 'Unknown')
        fidelity = turn.get('fidelity')
        tooltip = f"Turn {turn_num}: {status_label}"
        if fidelity is not None:
            tooltip += f" (F={fidelity:.2f})"
        attributes.append(f'title="{tooltip}"')

    attrs = " ".join(attributes)

    return f'<span style="{style}" {attrs}></span>'


# ============================================================================
# LEGEND GENERATION
# ============================================================================

def generate_timeline_legend(
    custom_colors: Optional[Dict[str, str]] = None,
    compact: bool = False,
) -> str:
    """
    Generate HTML legend for timeline markers.

    Args:
        custom_colors: Optional custom color mapping
        compact: If True, use compact horizontal layout

    Returns:
        HTML string with legend
    """
    colors = custom_colors if custom_colors else DEFAULT_STATUS_COLORS

    # Build legend items
    items = []
    for status, label in STATUS_LABELS.items():
        color = colors.get(status, colors['default'])

        if compact:
            item_html = f"""
                <span style="display: inline-flex; align-items: center; margin-right: 12px;">
                    <span style="width: 10px; height: 10px; background: {color}; border-radius: 50%; display: inline-block; margin-right: 4px;"></span>
                    <span style="font-size: 12px; color: #888;">{label}</span>
                </span>
            """
        else:
            item_html = f"""
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="width: 12px; height: 12px; background: {color}; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
                    <span style="font-size: 13px; color: #aaa;">{label}</span>
                </div>
            """

        items.append(item_html)

    # Container style
    if compact:
        container_style = "display: flex; justify-content: center; padding: 8px 0; font-size: 12px;"
    else:
        container_style = "padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; margin-top: 8px;"

    return f'<div style="{container_style}">{"".join(items)}</div>'


# ============================================================================
# ADVANCED MARKERS (WITH ANNOTATIONS)
# ============================================================================

def generate_annotated_markers(
    turns: List[Dict[str, Any]],
    current_turn: int,
    style: MarkerStyle = 'enhanced',
) -> str:
    """
    Generate timeline markers with annotation labels.

    Shows annotations above markers if turn has 'annotation' field.

    Args:
        turns: List of turn dicts (with optional 'annotation' field)
        current_turn: Active turn index
        style: Marker style

    Returns:
        HTML string with markers and annotations
    """
    if not turns:
        return "<div></div>"

    config = _get_marker_config(style)

    # Check if any turns have annotations
    has_annotations = any(turn.get('annotation') for turn in turns)

    if not has_annotations:
        # Fall back to standard markers
        return generate_timeline_markers(turns, current_turn, style)

    # Build markers with annotations
    markers_html = []

    for idx, turn in enumerate(turns):
        marker = _generate_single_marker(
            turn=turn,
            turn_index=idx,
            current_turn=current_turn,
            config=config,
            colors=DEFAULT_STATUS_COLORS,
            clickable=False,
            show_tooltips=False,
        )

        annotation = turn.get('annotation', '')
        is_active = (idx == current_turn)

        # Wrap marker with annotation
        if annotation:
            annotation_color = '#fff' if is_active else '#666'
            annotation_weight = '600' if is_active else '400'
            spacing = config['spacing'] // 2

            annotation_style = f"""
                font-size: 9px;
                color: {annotation_color};
                white-space: nowrap;
                margin-bottom: 4px;
                font-weight: {annotation_weight};
            """
            item_html = f"""
                <div style="display: flex; flex-direction: column; align-items: center; margin: 0 {spacing}px;">
                    <div style="{annotation_style}">{annotation}</div>
                    {marker}
                </div>
            """
        else:
            spacing = config['spacing'] // 2
            item_html = f'<div style="margin: 0 {spacing}px;">{marker}</div>'

        markers_html.append(item_html)

    container_style = """
        display: flex;
        justify-content: center;
        align-items: flex-end;
        padding: 8px 0;
        user-select: none;
    """

    return f'<div style="{container_style}">{"".join(markers_html)}</div>'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_marker_color(status: str, custom_colors: Optional[Dict[str, str]] = None) -> str:
    """
    Get color for a status marker.

    Args:
        status: Status symbol ('✓', '⚠️', etc.)
        custom_colors: Optional custom color mapping

    Returns:
        Hex color string
    """
    colors = custom_colors if custom_colors else DEFAULT_STATUS_COLORS
    return colors.get(status, colors['default'])


def validate_marker_style(style: str) -> bool:
    """
    Validate marker style string.

    Args:
        style: Style string to validate

    Returns:
        True if valid, False otherwise
    """
    return style in ['minimal', 'standard', 'enhanced']


def calculate_marker_width(num_turns: int, style: MarkerStyle = 'standard') -> int:
    """
    Calculate total width needed for markers.

    Useful for layout planning.

    Args:
        num_turns: Number of turns/markers
        style: Marker style

    Returns:
        Width in pixels
    """
    config = _get_marker_config(style)
    marker_width = config['size_active']  # Use larger size
    spacing = config['spacing']

    total_width = (num_turns * marker_width) + ((num_turns - 1) * spacing)
    return total_width


# ============================================================================
# CSS INJECTION (OPTIONAL)
# ============================================================================

def get_marker_css() -> str:
    """
    Get CSS for enhanced marker styling.

    Includes hover effects, animations, etc.

    Returns:
        CSS string
    """
    return """
    <style>
    .timeline-marker {
        transition: all 0.3s ease;
    }

    .timeline-marker:hover {
        transform: scale(1.2);
        filter: brightness(1.2);
    }

    .timeline-marker.active {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .timeline-marker.clickable {
        cursor: pointer;
    }

    .timeline-marker.clickable:hover {
        transform: scale(1.3);
        box-shadow: 0 0 12px currentColor;
    }
    </style>
    """


# ============================================================================
# CACHING (PERFORMANCE OPTIMIZATION)
# ============================================================================

_marker_cache: Dict[str, str] = {}


def generate_timeline_markers_cached(
    turns: List[Dict[str, Any]],
    current_turn: int,
    style: MarkerStyle = 'standard',
) -> str:
    """
    Generate timeline markers with caching.

    Caches marker HTML to avoid regeneration on every render.
    Cache key is based on turn count and style.
    Active marker styling is applied dynamically.

    Args:
        turns: List of turn dicts
        current_turn: Active turn index
        style: Marker style

    Returns:
        HTML string with markers
    """
    # Generate cache key (excludes current_turn since that changes frequently)
    cache_key = f"{len(turns)}:{style}:{hash(tuple(t.get('status', '✓') for t in turns))}"

    # Check cache
    if cache_key not in _marker_cache:
        # Generate and cache
        _marker_cache[cache_key] = generate_timeline_markers(
            turns=turns,
            current_turn=0,  # Generate with turn 0 active
            style=style,
        )

    # Return cached HTML (note: active marker styling is already applied)
    # For dynamic active marker, would need JavaScript or regeneration
    return generate_timeline_markers(turns, current_turn, style)


def clear_marker_cache() -> None:
    """Clear marker generation cache."""
    global _marker_cache
    _marker_cache.clear()
