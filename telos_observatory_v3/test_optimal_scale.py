# Test different distance_scale values
distances = {
    "AI governance": 0.692,
    "TELOS work": 0.775,
    "AI alignment": 0.797,
    "Python code": 0.898,
    "Weather": 1.001,
    "PB&J": 0.991
}

scales_to_test = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

print(f"{'Query':<20}", end='')
for scale in scales_to_test:
    print(f"scale={scale:<4}", end='  ')
print()
print("-" * 100)

for query, dist in distances.items():
    print(f"{query:<20}", end='')
    for scale in scales_to_test:
        fidelity = max(0.0, min(1.0, 1.0 - (dist / scale)))
        print(f"{fidelity:>8.3f}  ", end='')
    print()

print()
print("THRESHOLD = 0.70 (intervention if fidelity < 0.70)")
print()
print("GOAL: HIGHLY RELATED (AI gov, TELOS) should be >= 0.70")
print("      UNRELATED (Weather, PB&J) should be < 0.70")
