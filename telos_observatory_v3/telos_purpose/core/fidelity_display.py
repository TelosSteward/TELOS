"""
Fidelity Display Normalization
==============================

Maps SentenceTransformer raw scores to a normalized display scale that matches
user expectations (0.70+ = GREEN, 0.60+ = YELLOW, 0.50+ = ORANGE, <0.50 = RED).

CALIBRATION (2025-12-27):
-------------------------
Based on empirical testing with "Learn a Concept" PA template, actual raw
similarity scores from SentenceTransformer (all-MiniLM-L6-v2) are:

    Query Type                              Raw Score   Expected Zone
    ---------                               ---------   -------------
    "What is recursion?" (direct)           0.762       GREEN (100%)
    "Recursive functions" (highly relevant) 0.726       GREEN (95%)
    "Base cases in recursion"               0.647       GREEN (85%)
    "Recursion vs iteration" (tangent)      0.676       GREEN (88%)
    "Loops in programming" (related)        0.430       YELLOW (65%)
    "Function calls in Python"              0.364       ORANGE (55%)
    "What language to learn?"               0.391       ORANGE (58%)
    "How do I debug code?"                  0.206       RED (35%)
    "Pizza recipe?" (off-topic)             0.097       RED (15%)
    "Weather today?" (completely off)       -0.002      RED (0%)

NEW ANCHOR POINTS (collinear):
    Raw 0.62 → Display 0.70 (GREEN threshold)
    Raw 0.50 → Display 0.60 (YELLOW threshold)
    Raw 0.38 → Display 0.50 (ORANGE threshold)

Formula: display = (0.70 - 0.50) / (0.62 - 0.38) × (raw - 0.38) + 0.50
       = 0.833 × raw + 0.183

This spreads the useful range (0.35-0.75 raw) across the full display scale,
preventing the previous "all 100% or all <50%" binary distribution.
"""

from telos_purpose.core.constants import (
    ST_FIDELITY_GREEN, ST_FIDELITY_YELLOW, ST_FIDELITY_ORANGE
)

# Display thresholds (universal - what users expect)
DISPLAY_GREEN = 0.70
DISPLAY_YELLOW = 0.60
DISPLAY_ORANGE = 0.50

# NEW Linear transformation coefficients (2025-12-27 recalibration)
# Anchor points: raw 0.62→0.70, raw 0.50→0.60, raw 0.38→0.50
# m = (0.70 - 0.50) / (0.62 - 0.38) = 0.20 / 0.24 = 0.833
# b = 0.70 - 0.833 × 0.62 = 0.183
LINEAR_SLOPE = 0.833
LINEAR_INTERCEPT = 0.183


def normalize_st_fidelity(raw_fidelity: float) -> float:
    """
    Transform SentenceTransformer raw fidelity to display scale.

    Uses linear transformation calibrated 2025-12-27 based on empirical testing:
        Raw 0.76 → Display 0.82 (direct on-topic)
        Raw 0.65 → Display 0.72 (clearly aligned - GREEN)
        Raw 0.62 → Display 0.70 (GREEN threshold)
        Raw 0.50 → Display 0.60 (YELLOW threshold)
        Raw 0.43 → Display 0.54 (tangentially related)
        Raw 0.38 → Display 0.50 (ORANGE threshold)
        Raw 0.20 → Display 0.35 (off-topic programming)
        Raw 0.10 → Display 0.27 (clearly off-topic - RED)

    Args:
        raw_fidelity: Raw cosine similarity from SentenceTransformer

    Returns:
        Normalized fidelity on display scale (0.0 to 1.0)

    Examples:
        >>> normalize_st_fidelity(0.76)   # "What is recursion?" - direct on-topic
        0.82  # GREEN zone - aligned
        >>> normalize_st_fidelity(0.43)   # "Loops in programming" - related
        0.54  # ORANGE zone - minor drift
        >>> normalize_st_fidelity(0.10)   # "Pizza recipe" - off-topic
        0.27  # RED zone - triggers intervention
    """
    display = LINEAR_SLOPE * raw_fidelity + LINEAR_INTERCEPT
    return max(0.0, min(1.0, display))  # Clamp to [0, 1]


def normalize_fidelity_for_display(
    raw_fidelity: float,
    model_type: str
) -> float:
    """
    Normalize any model's raw fidelity to the universal display scale.

    Both SentenceTransformer and Mistral embeddings produce similar raw
    cosine similarity ranges (~0.35-0.60 for typical content), so we apply
    the same linear transformation to both.

    CHANGE (2025-12-26): Previously Mistral scores were passed through raw,
    causing off-topic content (raw ~0.35-0.45) to display as 78-88% instead
    of the expected ~40% (RED zone). Now both models use normalization.

    Args:
        raw_fidelity: Raw cosine similarity from embedding model
        model_type: 'sentence_transformer' or 'mistral'

    Returns:
        Normalized fidelity for display (0.0 to 1.0)
    """
    # Apply same normalization to all embedding models
    # Both ST and Mistral produce similar raw similarity ranges
    return normalize_st_fidelity(raw_fidelity)


def get_display_thresholds() -> dict:
    """
    Get the universal display thresholds.
    """
    return {
        'green': DISPLAY_GREEN,
        'yellow': DISPLAY_YELLOW,
        'orange': DISPLAY_ORANGE,
    }


def raw_to_display_mapping_table() -> str:
    """
    Generate a mapping table for verification.
    """
    lines = [
        "SentenceTransformer Raw → Display Mapping (RECALIBRATED 2025-12-27)",
        "=" * 60,
        f"{'Raw':<10} {'Display':<10} {'Zone':<10}",
        "-" * 60,
    ]

    # Test values spanning the realistic range based on empirical testing
    test_values = [0.76, 0.72, 0.68, 0.65, 0.62, 0.55, 0.50, 0.45, 0.43, 0.38, 0.35, 0.30, 0.20, 0.10, 0.00]

    for raw in test_values:
        display = normalize_st_fidelity(raw)
        if display >= DISPLAY_GREEN:
            zone = "GREEN"
        elif display >= DISPLAY_YELLOW:
            zone = "YELLOW"
        elif display >= DISPLAY_ORANGE:
            zone = "ORANGE"
        else:
            zone = "RED"
        lines.append(f"{raw:<10.3f} {display:<10.3f} {zone:<10}")

    lines.append("-" * 60)
    lines.append("Anchor points verified (2025-12-27):")
    lines.append(f"  ST 0.62 → {normalize_st_fidelity(0.62):.2f} (expect 0.70 GREEN threshold)")
    lines.append(f"  ST 0.50 → {normalize_st_fidelity(0.50):.2f} (expect 0.60 YELLOW threshold)")
    lines.append(f"  ST 0.38 → {normalize_st_fidelity(0.38):.2f} (expect 0.50 ORANGE threshold)")
    lines.append("Test cases from empirical data:")
    lines.append(f"  ST 0.76 → {normalize_st_fidelity(0.76):.2f} ('What is recursion?' - direct on-topic)")
    lines.append(f"  ST 0.43 → {normalize_st_fidelity(0.43):.2f} ('Loops in programming' - related)")
    lines.append(f"  ST 0.10 → {normalize_st_fidelity(0.10):.2f} ('Pizza recipe' - off-topic)")

    return "\n".join(lines)


if __name__ == "__main__":
    print(raw_to_display_mapping_table())
