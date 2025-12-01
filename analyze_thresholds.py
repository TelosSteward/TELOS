#!/usr/bin/env python3
"""
Statistical analysis to derive optimal thresholds from test data.
"""
import numpy as np

# Actual scores from the 100-question test (grouped by intended category)
GREEN_EXPECTED = [
    0.796, 0.828, 0.884, 0.715, 0.785, 0.624, 0.827, 0.781, 0.747, 0.732,
    0.787, 0.826, 0.669, 0.865, 0.804, 0.780, 0.788, 0.789, 0.759, 0.850,
    0.768, 0.757, 0.724, 0.752, 0.791
]

YELLOW_EXPECTED = [
    0.641, 0.782, 0.706, 0.754, 0.740, 0.752, 0.716, 0.781, 0.746, 0.746,
    0.661, 0.728, 0.820, 0.797, 0.744, 0.737, 0.716, 0.715, 0.808, 0.767
]

ORANGE_EXPECTED = [
    0.675, 0.734, 0.723, 0.662, 0.719, 0.725, 0.695, 0.650, 0.646, 0.671,
    0.738, 0.712, 0.688, 0.637, 0.724, 0.729, 0.688, 0.716, 0.714, 0.659
]

RED_EXPECTED = [
    0.581, 0.651, 0.625, 0.651, 0.603, 0.642, 0.640, 0.655, 0.583, 0.609,
    0.646, 0.666, 0.672, 0.618, 0.640, 0.628, 0.653, 0.654, 0.649, 0.638,
    0.604, 0.608, 0.635, 0.630, 0.662, 0.604, 0.659, 0.657, 0.634, 0.603,
    0.675, 0.580, 0.582, 0.643, 0.611
]

all_scores = GREEN_EXPECTED + YELLOW_EXPECTED + ORANGE_EXPECTED + RED_EXPECTED

print("=" * 70)
print("STATISTICAL THRESHOLD ANALYSIS")
print("=" * 70)

# Overall distribution
print("\n1. OVERALL DISTRIBUTION")
print(f"   Total samples: {len(all_scores)}")
print(f"   Mean: {np.mean(all_scores):.4f}")
print(f"   Std Dev: {np.std(all_scores):.4f}")
print(f"   Min: {min(all_scores):.4f}")
print(f"   Max: {max(all_scores):.4f}")

# Percentiles
percentiles = [10, 25, 50, 75, 90, 95]
print(f"\n   Percentiles:")
for p in percentiles:
    val = np.percentile(all_scores, p)
    print(f"     {p}th: {val:.4f}")

# Per-category statistics
print("\n2. CATEGORY STATISTICS")
categories = [
    ("GREEN (on-topic TELOS)", GREEN_EXPECTED),
    ("YELLOW (related AI)", YELLOW_EXPECTED),
    ("ORANGE (tangential tech)", ORANGE_EXPECTED),
    ("RED (unrelated)", RED_EXPECTED)
]

for name, scores in categories:
    print(f"\n   {name}:")
    print(f"     N: {len(scores)}")
    print(f"     Mean: {np.mean(scores):.4f}")
    print(f"     Std: {np.std(scores):.4f}")
    print(f"     Min: {min(scores):.4f}")
    print(f"     Max: {max(scores):.4f}")
    print(f"     25th percentile: {np.percentile(scores, 25):.4f}")
    print(f"     75th percentile: {np.percentile(scores, 75):.4f}")

# Derive thresholds mathematically
print("\n3. MATHEMATICALLY DERIVED THRESHOLDS")
print("\n   Method 1: Midpoint between category means")
green_mean = np.mean(GREEN_EXPECTED)
yellow_mean = np.mean(YELLOW_EXPECTED)
orange_mean = np.mean(ORANGE_EXPECTED)
red_mean = np.mean(RED_EXPECTED)

threshold_gy = (green_mean + yellow_mean) / 2
threshold_yo = (yellow_mean + orange_mean) / 2
threshold_or = (orange_mean + red_mean) / 2

print(f"     GREEN/YELLOW boundary: {threshold_gy:.4f}")
print(f"     YELLOW/ORANGE boundary: {threshold_yo:.4f}")
print(f"     ORANGE/RED boundary: {threshold_or:.4f}")

print("\n   Method 2: Based on category 25th/75th percentiles (minimize overlap)")
green_25 = np.percentile(GREEN_EXPECTED, 25)
yellow_75 = np.percentile(YELLOW_EXPECTED, 75)
yellow_25 = np.percentile(YELLOW_EXPECTED, 25)
orange_75 = np.percentile(ORANGE_EXPECTED, 75)
orange_25 = np.percentile(ORANGE_EXPECTED, 25)
red_75 = np.percentile(RED_EXPECTED, 75)

print(f"     GREEN 25th: {green_25:.4f}, YELLOW 75th: {yellow_75:.4f}")
print(f"     -> GREEN/YELLOW threshold: {(green_25 + yellow_75) / 2:.4f}")
print(f"     YELLOW 25th: {yellow_25:.4f}, ORANGE 75th: {orange_75:.4f}")
print(f"     -> YELLOW/ORANGE threshold: {(yellow_25 + orange_75) / 2:.4f}")
print(f"     ORANGE 25th: {orange_25:.4f}, RED 75th: {red_75:.4f}")
print(f"     -> ORANGE/RED threshold: {(orange_25 + red_75) / 2:.4f}")

print("\n   Method 3: Standard deviation bands from overall mean")
overall_mean = np.mean(all_scores)
overall_std = np.std(all_scores)
print(f"     Mean: {overall_mean:.4f}, Std: {overall_std:.4f}")
print(f"     Mean + 0.5*Std (GREEN): {overall_mean + 0.5*overall_std:.4f}")
print(f"     Mean (YELLOW): {overall_mean:.4f}")
print(f"     Mean - 0.5*Std (ORANGE): {overall_mean - 0.5*overall_std:.4f}")
print(f"     Mean - 1.0*Std (RED): {overall_mean - 1.0*overall_std:.4f}")

# Final recommendation
print("\n4. RECOMMENDED THRESHOLDS (combining methods)")
# Use Method 1 (midpoints) as primary, rounded to clean values
rec_green = round((green_mean + yellow_mean) / 2, 2)
rec_yellow = round((yellow_mean + orange_mean) / 2, 2)
rec_orange = round((orange_mean + red_mean) / 2, 2)

print(f"\n   GREEN  >= {rec_green}")
print(f"   YELLOW >= {rec_yellow} (range {rec_yellow}-{rec_green})")
print(f"   ORANGE >= {rec_orange} (range {rec_orange}-{rec_yellow})")
print(f"   RED    <  {rec_orange}")

# Validate
print("\n5. VALIDATION WITH RECOMMENDED THRESHOLDS")
def classify(score, g, y, o):
    if score >= g: return "GREEN"
    elif score >= y: return "YELLOW"
    elif score >= o: return "ORANGE"
    else: return "RED"

correct = 0
for s in GREEN_EXPECTED:
    if classify(s, rec_green, rec_yellow, rec_orange) == "GREEN": correct += 1
print(f"   GREEN accuracy: {correct}/{len(GREEN_EXPECTED)} ({100*correct/len(GREEN_EXPECTED):.1f}%)")

correct = 0
for s in YELLOW_EXPECTED:
    if classify(s, rec_green, rec_yellow, rec_orange) == "YELLOW": correct += 1
print(f"   YELLOW accuracy: {correct}/{len(YELLOW_EXPECTED)} ({100*correct/len(YELLOW_EXPECTED):.1f}%)")

correct = 0
for s in ORANGE_EXPECTED:
    if classify(s, rec_green, rec_yellow, rec_orange) == "ORANGE": correct += 1
print(f"   ORANGE accuracy: {correct}/{len(ORANGE_EXPECTED)} ({100*correct/len(ORANGE_EXPECTED):.1f}%)")

correct = 0
for s in RED_EXPECTED:
    if classify(s, rec_green, rec_yellow, rec_orange) == "RED": correct += 1
print(f"   RED accuracy: {correct}/{len(RED_EXPECTED)} ({100*correct/len(RED_EXPECTED):.1f}%)")
