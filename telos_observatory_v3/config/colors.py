"""
TELOSCOPE Color Configuration
=============================

Centralized color definitions for consistency and accessibility.
Updated based on accessibility audit to reduce eye strain.
"""

# Primary Brand Colors
GOLD = '#F4D03F'  # Refined gold - less "neon", better for eyes
GOLD_BRIGHT = '#F4D03F'  # Legacy bright gold (deprecated - do not use)

# Status Colors (Fidelity Indicators) - Zone-based naming for user clarity
# Thresholds are internal implementation details, not user-facing
STATUS_GOOD = '#4CAF50'     # Green - "Aligned" zone
STATUS_MILD = '#F4D03F'      # Yellow/Gold - "Minor Drift" zone
STATUS_MODERATE = '#FF9E42'  # Orange - "Drift Detected" zone
STATUS_SEVERE = '#FF5757'    # Red - "Significant Drift" zone

# Background Colors
BG_BASE = '#0d0d0d'        # Darkest - base layer
BG_SURFACE = '#1a1a1a'     # Primary surface
BG_ELEVATED = '#2d2d2d'    # Elevated/card surface
BG_OVERLAY = '#3a3a3a'     # Modal/overlay surface

# Text Colors
TEXT_PRIMARY = '#e0e0e0'    # Primary text
TEXT_SECONDARY = '#b0b0b0'  # Secondary text
TEXT_MUTED = '#808080'      # Muted/disabled text

# Opacity Variations (for visual hierarchy)
def with_opacity(color: str, opacity: float) -> str:
    """Add opacity to a hex color."""
    # Convert hex to RGB
    hex_color = color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {opacity})'

# Gold variations for hierarchy
GOLD_100 = GOLD                           # Headers
GOLD_80 = with_opacity(GOLD, 0.8)        # Borders
GOLD_60 = with_opacity(GOLD, 0.6)        # Accents
GOLD_40 = with_opacity(GOLD, 0.4)        # Subtle hints

# Legacy mappings (for backward compatibility during migration)
COLOR_MAPPINGS = {
    '#F4D03F': GOLD,      # Map old bright gold to refined gold
    '#FF4444': STATUS_SEVERE,  # Map old red to improved red
    '#FFA500': STATUS_MODERATE,  # Map old orange to refined orange
}

# Goldilocks zone thresholds (internal - derived from mathematical optimization)
# These are implementation details, users see zone names not numbers
_ZONE_ALIGNED = 0.76      # "Aligned" zone threshold
_ZONE_MINOR_DRIFT = 0.73  # "Minor Drift" zone threshold
_ZONE_DRIFT = 0.67        # "Drift Detected" zone threshold
# Below _ZONE_DRIFT = "Significant Drift" zone


def get_fidelity_color(score: float) -> str:
    """Get the appropriate color for a fidelity score.

    Uses Goldilocks zone thresholds derived from mathematical optimization.
    """
    if score >= _ZONE_ALIGNED:
        return STATUS_GOOD
    elif score >= _ZONE_MINOR_DRIFT:
        return STATUS_MILD
    elif score >= _ZONE_DRIFT:
        return STATUS_MODERATE
    else:
        return STATUS_SEVERE


def get_color_name(score: float) -> str:
    """Get the color name for a fidelity score.

    Uses Goldilocks zone thresholds derived from mathematical optimization.
    """
    if score >= _ZONE_ALIGNED:
        return "green"
    elif score >= _ZONE_MINOR_DRIFT:
        return "gold"
    elif score >= _ZONE_DRIFT:
        return "orange"
    else:
        return "red"


def get_zone_name(score: float) -> str:
    """Get the human-friendly zone name for a fidelity score.

    Returns semantic zone names instead of threshold numbers for better UX.
    """
    if score >= _ZONE_ALIGNED:
        return "Aligned"
    elif score >= _ZONE_MINOR_DRIFT:
        return "Minor Drift"
    elif score >= _ZONE_DRIFT:
        return "Drift Detected"
    else:
        return "Significant Drift"