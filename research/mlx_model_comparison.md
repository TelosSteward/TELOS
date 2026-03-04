# MLX Model Comparison Experiment

**Date:** 2026-03-01
**Benchmark:** Nearmap Counterfactual v1 (235 scenarios)
**Hardware:** Apple Silicon (MLX native), ONNX Runtime (CPU)

---

## Results

| Model | Overall | Cat A | Cat B | Cat C | Cat D | Cat E | FPR | Time |
|-------|---------|-------|-------|-------|-------|-------|-----|------|
| **MiniLM-ONNX (baseline)** | **84.7%** | 90.0% | 90.5% | 89.8% | 100.0% | 60.0% | 10.2% | 2s |
| MiniLM-MLX (4-bit) | 78.3% | 94.0% | 88.1% | 73.9% | 70.0% | 62.2% | 26.1% | 4s |
| MPNet-ONNX | 71.5% | 98.0% | 69.0% | 62.5% | 60.0% | 64.4% | 37.5% | 6s |
| E5-base-MLX | 23.4% | 0.0% | 16.7% | 51.1% | 10.0% | 4.4% | 48.9% | 6s |

**Critical caveat:** All models use MiniLM-calibrated thresholds. Non-MiniLM models are **UNCALIBRATED** — their cosine similarity distributions differ, so thresholds tuned for MiniLM produce incorrect decision boundaries.

---

## Analysis

### MiniLM-ONNX (baseline) — 84.7% overall
Current production model. All thresholds calibrated for this model. Best overall accuracy with balanced performance across all categories.

### MiniLM-MLX (4-bit) — 78.3% overall
Same architecture, 4-bit quantized. Key observations:
- **Cat A improved** (90.0% → 94.0%): Quantization may sharpen boundary detection by reducing noise in mid-range similarities
- **Cat C degraded** (89.8% → 73.9%): Quantization shifts cosine distributions, causing legitimate requests to be over-flagged (FPR 10.2% → 26.1%)
- **Cat E improved** (60.0% → 62.2%): Slight adversarial detection improvement
- **Conclusion:** 4-bit quantization changes cosine distributions enough to require recalibration. The underlying model quality is preserved (Cat A proves boundary separation is intact), but thresholds need adjustment.

### MPNet-ONNX — 71.5% overall (UNCALIBRATED)
Larger model (109M vs 22M, 768-dim vs 384-dim). Key observations:
- **Cat A: 98.0%** — Near-perfect boundary detection. The larger embedding space separates violations significantly better than MiniLM.
- **Cat B: 69.0%** — Off-topic detection degraded. MiniLM thresholds are too permissive in MPNet's embedding space.
- **Cat C: 62.5%** — Massive false positive rate (37.5%). MPNet's cosine distributions sit lower, so MiniLM's "execute" threshold lets almost nothing through.
- **Cat E: 64.4%** — Best adversarial detection of all models.
- **Conclusion:** MPNet has genuinely superior boundary detection (+8pp Cat A, +4pp Cat E) but needs calibrated thresholds. The 49.3% score reported previously was also uncalibrated — this 71.5% represents improvement from the A20 Advisory engine fixes.

### E5-base-MLX — 23.4% overall (UNCALIBRATED)
Instruction-tuned multilingual model. Catastrophically mismatched with MiniLM thresholds. The embedding space is structured fundamentally differently (instruction-tuned models produce different similarity distributions). Not viable without complete recalibration.

---

## Key Finding: Cat A (Boundary Detection)

| Model | Cat A |
|-------|-------|
| MiniLM-ONNX | 90.0% |
| MiniLM-MLX (4-bit) | 94.0% |
| **MPNet-ONNX** | **98.0%** |
| E5-base-MLX | 0.0% |

MPNet's 98% Cat A strongly suggests the larger embedding space separates boundary violations better. If thresholds were calibrated for MPNet, we would likely see:
- Maintained ~98% Cat A (boundary detection)
- Recovered Cat B/C (off-topic and legitimate) via calibrated thresholds
- Best-in-class Cat E (adversarial) at ~65%+

---

## Recommendations

1. **Production:** Keep MiniLM-ONNX as default. It's calibrated, fast (2s), and 84.7% overall.

2. **Next experiment:** Run the governance optimizer with `--model mpnet` to calibrate MPNet thresholds. The 98% Cat A signal is too strong to ignore — if calibrated thresholds recover Cat B/C while maintaining Cat A, MPNet becomes the production model.

3. **MLX backend:** Useful for development speed on Apple Silicon but not a production differentiator. 4-bit quantization changes cosine distributions enough to require separate calibration. ONNX remains the portable default.

4. **E5-base:** Eliminate from consideration. Instruction-tuned embedding spaces are fundamentally incompatible with the cosine-threshold governance approach without architectural changes.

---

## Files

- `validation/nearmap/run_model_comparison.py` — Experiment script
- `validation/nearmap/model_comparison_results.json` — Raw results
- `telos_core/embedding_provider.py` — MlxEmbeddingProvider class
- `tests/unit/test_mlx_embedding.py` — 20 MLX provider tests
